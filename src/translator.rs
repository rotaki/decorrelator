use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    catalog::{self, Catalog, CatalogRef},
    col_id_generator::ColIdGeneratorRef,
    expressions::prelude::*,
    field::Field,
    rules::RulesRef,
};

type EnvironmentRef = Rc<Environment>;

#[derive(Debug, Clone)]
struct Environment {
    outer: Option<EnvironmentRef>,
    columns: RefCell<HashMap<String, usize>>,
}

impl Environment {
    fn new() -> Environment {
        Environment {
            outer: None,
            columns: RefCell::new(HashMap::new()),
        }
    }

    fn new_with_outer(outer: EnvironmentRef) -> Environment {
        Environment {
            outer: Some(outer),
            columns: RefCell::new(HashMap::new()),
        }
    }

    fn get(&self, name: &str) -> Option<usize> {
        if let Some(index) = self.columns.borrow().get(name) {
            return Some(*index);
        }

        if let Some(outer) = &self.outer {
            return outer.get(name);
        }

        None
    }

    fn get_at(&self, distance: usize, name: &str) -> Option<usize> {
        if distance == 0 {
            if let Some(index) = self.columns.borrow().get(name) {
                return Some(*index);
            } else {
                return None;
            }
        }

        if let Some(outer) = &self.outer {
            return outer.get_at(distance - 1, name);
        }

        None
    }

    fn set(&self, name: &str, index: usize) {
        self.columns.borrow_mut().insert(name.to_string(), index);
    }

    fn get_names(&self, col_id: usize) -> Vec<String> {
        let mut names = Vec::new();
        for (name, index) in self.columns.borrow().iter() {
            if *index == col_id {
                names.push(name.clone());
            }
        }
        names
    }
}

struct Translator {
    catalog_ref: CatalogRef,
    enabled_rules: RulesRef,
    col_id_gen: ColIdGeneratorRef,
    env: EnvironmentRef, // Variables in the current scope
}

struct Query {
    env: EnvironmentRef,
    plan: RelExpr,
}

#[derive(Debug)]
pub enum TranslatorError {
    ColumnNotFound(String),
    TableNotFound(String),
    InvalidSQL(String),
    UnsupportedSQL(String),
}

macro_rules! translation_err {
    (ColumnNotFound, $($arg:tt)*) => {
        TranslatorError::ColumnNotFound(format!($($arg)*))
    };
    (TableNotFound, $($arg:tt)*) => {
        TranslatorError::TableNotFound(format!($($arg)*))
    };
    (InvalidSQL, $($arg:tt)*) => {
        TranslatorError::InvalidSQL(format!($($arg)*))
    };
    (UnsupportedSQL, $($arg:tt)*) => {
        TranslatorError::UnsupportedSQL(format!($($arg)*))
    };
}

impl Translator {
    pub fn new(
        catalog: &CatalogRef,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
    ) -> Translator {
        Translator {
            catalog_ref: catalog.clone(),
            enabled_rules: enabled_rules.clone(),
            col_id_gen: col_id_gen.clone(),
            env: Rc::new(Environment::new()),
        }
    }

    fn new_with_outer(
        catalog: &CatalogRef,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        outer: &EnvironmentRef,
    ) -> Translator {
        Translator {
            col_id_gen: col_id_gen.clone(),
            enabled_rules: enabled_rules.clone(),
            catalog_ref: catalog.clone(),
            env: Rc::new(Environment::new_with_outer(outer.clone())),
        }
    }

    pub fn process_query(
        &mut self,
        query: &sqlparser::ast::Query,
    ) -> Result<Query, TranslatorError> {
        let select = match query.body.as_ref() {
            sqlparser::ast::SetExpr::Select(select) => select,
            _ => {
                return Err(translation_err!(
                    UnsupportedSQL,
                    "Only SELECT queries are supported"
                ))
            }
        };

        let plan = self.process_from(&select.from)?;
        let plan = self.process_where(plan, &select.selection)?;
        let plan = self.process_projection(
            plan,
            &select.projection,
            &select.from,
            &query.order_by,
            &query.limit,
            &select.group_by,
            &select.having,
            &select.distinct,
        )?;

        Ok(Query {
            env: self.env.clone(),
            plan,
        })
    }

    fn process_from(
        &mut self,
        from: &[sqlparser::ast::TableWithJoins],
    ) -> Result<RelExpr, TranslatorError> {
        if from.is_empty() {
            return Err(translation_err!(InvalidSQL, "FROM clause is empty"));
        }

        let mut join_exprs = Vec::with_capacity(from.len());
        for table_with_joins in from {
            let join_expr = self.process_table_with_joins(table_with_joins)?;
            join_exprs.push(join_expr);
        }
        let (mut plan, _) = join_exprs.remove(0);
        for (join_expr, is_subquery) in join_exprs.into_iter() {
            plan = if is_subquery {
                plan.flatmap(true, &self.enabled_rules, &self.col_id_gen, join_expr)
            } else {
                plan.join(
                    true,
                    &self.enabled_rules,
                    &self.col_id_gen,
                    JoinType::CrossJoin,
                    join_expr,
                    vec![],
                )
            }
        }
        Ok(plan)
    }

    fn process_table_with_joins(
        &mut self,
        table_with_joins: &sqlparser::ast::TableWithJoins,
    ) -> Result<(RelExpr, bool), TranslatorError> {
        let (mut plan, is_sbqry) = self.process_table_factor(&table_with_joins.relation)?;
        for join in &table_with_joins.joins {
            let (right, is_subquery) = self.process_table_factor(&join.relation)?;
            // If it is a subquery, we use flat_map + condition
            // Other wise we use a join
            let (join_type, condition) = self.process_join_operator(&join.join_operator)?;
            plan = if is_subquery {
                if matches!(
                    join_type,
                    JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter
                ) {
                    return Err(translation_err!(
                        UnsupportedSQL,
                        "Unsupported join type with subquery"
                    ));
                }
                plan.flatmap(true, &self.enabled_rules, &self.col_id_gen, right)
                    .select(
                        true,
                        &self.enabled_rules,
                        &self.col_id_gen,
                        condition.into_iter().collect(),
                    )
            } else {
                plan.join(
                    true,
                    &self.enabled_rules,
                    &self.col_id_gen,
                    join_type,
                    right,
                    condition.into_iter().collect(),
                )
            }
        }
        Ok((plan, is_sbqry))
    }

    fn process_join_operator(
        &self,
        join_operator: &sqlparser::ast::JoinOperator,
    ) -> Result<(JoinType, Option<Expr>), TranslatorError> {
        use sqlparser::ast::{JoinConstraint, JoinOperator::*};
        match join_operator {
            Inner(JoinConstraint::On(cond)) => {
                Ok((JoinType::Inner, Some(self.process_expr(cond, None)?)))
            }
            LeftOuter(JoinConstraint::On(cond)) => {
                Ok((JoinType::LeftOuter, Some(self.process_expr(cond, None)?)))
            }
            RightOuter(JoinConstraint::On(cond)) => {
                Ok((JoinType::RightOuter, Some(self.process_expr(cond, None)?)))
            }
            FullOuter(JoinConstraint::On(cond)) => {
                Ok((JoinType::FullOuter, Some(self.process_expr(cond, None)?)))
            }
            CrossJoin => Ok((JoinType::CrossJoin, None)),
            _ => Err(translation_err!(
                UnsupportedSQL,
                "Unsupported join operator: {:?}",
                join_operator
            )),
        }
    }

    // Out: (RelExpr, is_subquery: bool)
    fn process_table_factor(
        &mut self,
        table_factor: &sqlparser::ast::TableFactor,
    ) -> Result<(RelExpr, bool), TranslatorError> {
        match table_factor {
            sqlparser::ast::TableFactor::Table { name, alias, .. } => {
                // Find the actual name from the catalog
                // If name exists in the catalog, then add the columns to the environment
                // Otherwise return an error
                let table_name = get_name(name);
                if self.catalog_ref.is_valid_table(&table_name) {
                    let cols = self.catalog_ref.get_cols(&table_name);
                    let plan =
                        RelExpr::scan(table_name.clone(), cols.iter().map(|(_, id)| *id).collect());
                    let att = plan.att();
                    let (plan, mut new_col_ids) =
                        plan.rename(&self.enabled_rules, &self.col_id_gen);

                    for i in att {
                        let new_col_id = new_col_ids.remove(&i).unwrap();
                        // get the name of the column
                        let col_name = cols.iter().find(|(_, id)| *id == i).cloned().unwrap().0;
                        self.env.set(&col_name, new_col_id);
                        self.env
                            .set(&format!("{}.{}", table_name, col_name), new_col_id);

                        // If there is an alias, set the alias in the current environment
                        if let Some(alias) = alias {
                            if is_valid_alias(&alias.name.value) {
                                self.env.set(&format!("{}.{}", alias, col_name), new_col_id);
                            } else {
                                return Err(translation_err!(
                                    InvalidSQL,
                                    "Invalid alias name: {}",
                                    alias.name.value
                                ));
                            }
                        }
                    }

                    Ok((plan, false))
                } else {
                    Err(translation_err!(TableNotFound, "{}", table_name))
                }
            }
            sqlparser::ast::TableFactor::Derived {
                subquery, alias, ..
            } => {
                let mut translator = Translator::new_with_outer(
                    &self.catalog_ref,
                    &self.enabled_rules,
                    &self.col_id_gen,
                    &self.env,
                );
                let subquery = translator.process_query(subquery)?;
                let plan = subquery.plan;
                let att = plan.att();

                for i in att {
                    // get the name of the column from env
                    let names = subquery.env.get_names(i);
                    for name in &names {
                        self.env.set(&name, i);
                    }
                    // If there is an alias, set the alias in the current environment
                    if let Some(alias) = alias {
                        if is_valid_alias(&alias.name.value) {
                            for name in &names {
                                self.env.set(&format!("{}.{}", alias, name), i);
                            }
                        } else {
                            return Err(translation_err!(
                                InvalidSQL,
                                "Invalid alias name: {}",
                                alias.name.value
                            ));
                        }
                    }
                }
                Ok((plan, true))
            }
            _ => Err(translation_err!(UnsupportedSQL, "Unsupported table factor")),
        }
    }

    fn process_where(
        &mut self,
        plan: RelExpr,
        where_clause: &Option<sqlparser::ast::Expr>,
    ) -> Result<RelExpr, TranslatorError> {
        if let Some(expr) = where_clause {
            match self.process_expr(expr, Some(0)) {
                Ok(expr) => {
                    match expr {
                        Expr::Subquery { expr } => {
                            if expr.att().len() != 1 {
                                panic!("Subquery in WHERE clause returns more than one column")
                            }
                            // Add map first
                            let col_id = self.col_id_gen.next();
                            let plan = plan.map(
                                true,
                                &self.enabled_rules,
                                &self.col_id_gen,
                                [(col_id, Expr::subquery(*expr))],
                            );
                            // Add select
                            Ok(plan.select(
                                true,
                                &self.enabled_rules,
                                &self.col_id_gen,
                                vec![Expr::col_ref(col_id)],
                            ))
                        }
                        _ => Ok(plan.select(
                            true,
                            &self.enabled_rules,
                            &self.col_id_gen,
                            vec![expr],
                        )),
                    }
                }
                Err(TranslatorError::ColumnNotFound(_)) => {
                    // Search globally.
                    let expr = self.process_expr(expr, None)?;
                    let col_id = self.col_id_gen.next();
                    Ok(plan
                        .map(
                            true,
                            &self.enabled_rules,
                            &self.col_id_gen,
                            [(col_id, expr)],
                        )
                        .select(
                            true,
                            &self.enabled_rules,
                            &self.col_id_gen,
                            vec![Expr::col_ref(col_id)],
                        ))
                }
                Err(e) => Err(e),
            }
        } else {
            Ok(plan)
        }
    }

    fn process_projection(
        &mut self,
        mut plan: RelExpr,
        projection: &Vec<sqlparser::ast::SelectItem>,
        from: &Vec<sqlparser::ast::TableWithJoins>,
        order_by: &Vec<sqlparser::ast::OrderByExpr>,
        limit: &Option<sqlparser::ast::Expr>,
        group_by: &sqlparser::ast::GroupByExpr,
        having: &Option<sqlparser::ast::Expr>,
        distinct: &Option<sqlparser::ast::Distinct>,
    ) -> Result<RelExpr, TranslatorError> {
        let mut projected_cols = HashSet::new();
        let mut aggregations = Vec::new();
        let mut maps = Vec::new();
        for item in projection {
            match item {
                sqlparser::ast::SelectItem::Wildcard(_) => {
                    // Add all the environment variables to the projected columns
                    for (_, col_id) in self.env.columns.borrow().iter() {
                        projected_cols.insert(*col_id);
                    }
                }
                sqlparser::ast::SelectItem::UnnamedExpr(expr) => {
                    if !has_agg(expr) {
                        match self.process_expr(expr, Some(0)) {
                            Ok(expr) => {
                                let col_id = if let Expr::ColRef { id } = expr {
                                    id
                                } else {
                                    // create a new col_id for the expression
                                    let col_id = self.col_id_gen.next();
                                    plan = plan.map(
                                        true,
                                        &self.enabled_rules,
                                        &self.col_id_gen,
                                        [(col_id, expr)],
                                    );
                                    col_id
                                };
                                projected_cols.insert(col_id);
                            }
                            Err(TranslatorError::ColumnNotFound(_)) => {
                                // Search globally.
                                let expr = self.process_expr(expr, None)?;
                                // Add a map to the plan
                                let col_id = self.col_id_gen.next();
                                plan = plan.map(
                                    true,
                                    &self.enabled_rules,
                                    &self.col_id_gen,
                                    [(col_id, expr)],
                                );
                                projected_cols.insert(col_id);
                            }
                            Err(e) => return Err(e),
                        }
                    } else {
                        // The most complicated case will be:
                        // Agg(a + b) + Agg(c + d) + 4
                        // if we ignore nested aggregation.
                        //
                        // In this case,
                        // Level1: | map a + b to col_id1
                        //         | map c + d to col_id2
                        // Level2: |Agg(col_id1) to col_id3
                        //         |Agg(col_id2) to col_id4
                        // Level3: |map col_id3 + col_id4 + 4 to col_id5

                        let mut aggs = Vec::new();
                        let res = self.process_aggregation_arguments(plan, expr, &mut aggs);
                        plan = res.0;
                        let expr = res.1;
                        let col_id = if let Expr::ColRef { id } = expr {
                            id
                        } else {
                            // create a new col_id for the expression
                            let col_id = self.col_id_gen.next();
                            maps.push((col_id, expr));
                            col_id
                        };
                        aggregations.append(&mut aggs);
                        projected_cols.insert(col_id);
                    }
                }
                sqlparser::ast::SelectItem::ExprWithAlias { expr, alias } => {
                    // create a new col_id for the expression
                    let col_id = if !has_agg(expr) {
                        let col_id = match self.process_expr(expr, Some(0)) {
                            Ok(expr) => {
                                if let Expr::ColRef { id } = expr {
                                    id
                                } else {
                                    let col_id = self.col_id_gen.next();
                                    plan = plan.map(
                                        true,
                                        &self.enabled_rules,
                                        &self.col_id_gen,
                                        [(col_id, expr)],
                                    );
                                    col_id
                                }
                            }
                            Err(TranslatorError::ColumnNotFound(_)) => {
                                // Search globally.
                                let expr = self.process_expr(expr, None)?;
                                let col_id = self.col_id_gen.next();
                                plan = plan.map(
                                    true,
                                    &self.enabled_rules,
                                    &self.col_id_gen,
                                    [(col_id, expr)],
                                );
                                col_id
                            }
                            Err(e) => return Err(e),
                        };
                        projected_cols.insert(col_id);
                        col_id
                    } else {
                        // The most complicated case will be:
                        // Agg(a + b) + Agg(c + d) + 4
                        // if we ignore nested aggregation.
                        //
                        // In this case,
                        // Level1: | map a + b to col_id1
                        //         | map c + d to col_id2
                        // Level2: |Agg(col_id1) to col_id3
                        //         |Agg(col_id2) to col_id4
                        // Level3: |map col_id3 + col_id4 + 4 to col_id5

                        let mut aggs = Vec::new();
                        let res = self.process_aggregation_arguments(plan, expr, &mut aggs);
                        plan = res.0;
                        let expr = res.1;
                        let col_id = if let Expr::ColRef { id } = expr {
                            id
                        } else {
                            // create a new col_id for the expression
                            let col_id = self.col_id_gen.next();
                            maps.push((col_id, expr));
                            col_id
                        };
                        aggregations.append(&mut aggs);
                        projected_cols.insert(col_id);
                        col_id
                    };

                    // Add the alias to the aliases map
                    let alias_name = alias.value.clone();
                    if is_valid_alias(&alias_name) {
                        self.env.set(&alias_name, col_id);
                    } else {
                        return Err(translation_err!(
                            InvalidSQL,
                            "Invalid alias name: {}",
                            alias_name
                        ));
                    }
                }
                _ => {
                    return Err(translation_err!(
                        UnsupportedSQL,
                        "Unsupported select item: {:?}",
                        item
                    ))
                }
            }
        }

        if !aggregations.is_empty() {
            let group_by = match group_by {
                sqlparser::ast::GroupByExpr::All => Err(translation_err!(
                    UnsupportedSQL,
                    "GROUP BY ALL is not supported"
                ))?,
                sqlparser::ast::GroupByExpr::Expressions(exprs) => {
                    let mut group_by = Vec::new();
                    for expr in exprs {
                        let expr = self.process_expr(expr, None)?;
                        let col_id = if let Expr::ColRef { id } = expr {
                            id
                        } else {
                            // create a new col_id for the expression
                            let col_id = self.col_id_gen.next();
                            plan = plan.map(
                                true,
                                &self.enabled_rules,
                                &self.col_id_gen,
                                [(col_id, expr)],
                            );
                            col_id
                        };
                        group_by.push(col_id);
                    }
                    group_by
                }
            };
            plan = plan.aggregate(group_by, aggregations);
            plan = self.process_where(plan, having)?;
        }
        plan = plan.map(true, &self.enabled_rules, &self.col_id_gen, maps); // This map corresponds to the Level3 in the comment above
        plan = plan.project(true, &self.enabled_rules, &self.col_id_gen, projected_cols);
        Ok(plan)
    }

    // DFS until we find an aggregation function
    // If we find an aggregation function, then add the aggregation argument to the plan
    // and put the aggregation function in the aggregation list, return the modified plan with the expression.
    // For example, if SUM(a+b) + AVG(c+d) + 4, then
    // a+b -> col_id1, c+d -> col_id2 will be added to the plan
    // SUM(col_id1) -> col_id3, AVG(col_id2) -> col_id4 will be added to the aggregation list
    // col_id3 + col_id4 + 4 -> col_id5 will be returned with the plan
    fn process_aggregation_arguments(
        &self,
        mut plan: RelExpr,
        expr: &sqlparser::ast::Expr,
        aggs: &mut Vec<(usize, (usize, AggOp))>,
    ) -> (RelExpr, Expr) {
        match expr {
            sqlparser::ast::Expr::Identifier(_) | sqlparser::ast::Expr::CompoundIdentifier(_) => {
                unreachable!(
                    "Identifier and compound identifier should be processed in the Function branch"
                )
            }
            sqlparser::ast::Expr::Value(_) | sqlparser::ast::Expr::TypedString { .. } => {
                let expr = self.process_expr(expr, Some(0)).unwrap();
                (plan, expr)
            }
            sqlparser::ast::Expr::BinaryOp { left, op, right } => {
                let (plan, left) = self.process_aggregation_arguments(plan, left, aggs);
                let (plan, right) = self.process_aggregation_arguments(plan, right, aggs);
                let bin_op = match op {
                    sqlparser::ast::BinaryOperator::And => BinaryOp::And,
                    sqlparser::ast::BinaryOperator::Or => BinaryOp::Or,
                    sqlparser::ast::BinaryOperator::Plus => BinaryOp::Add,
                    sqlparser::ast::BinaryOperator::Minus => BinaryOp::Sub,
                    sqlparser::ast::BinaryOperator::Multiply => BinaryOp::Mul,
                    sqlparser::ast::BinaryOperator::Divide => BinaryOp::Div,
                    sqlparser::ast::BinaryOperator::Eq => BinaryOp::Eq,
                    sqlparser::ast::BinaryOperator::NotEq => BinaryOp::Neq,
                    sqlparser::ast::BinaryOperator::Lt => BinaryOp::Lt,
                    sqlparser::ast::BinaryOperator::Gt => BinaryOp::Gt,
                    sqlparser::ast::BinaryOperator::LtEq => BinaryOp::Le,
                    sqlparser::ast::BinaryOperator::GtEq => BinaryOp::Ge,
                    _ => unimplemented!("Unsupported binary operator: {:?}", op),
                };
                (plan, Expr::binary(bin_op, left, right))
            }
            sqlparser::ast::Expr::Function(function) => {
                let name = get_name(&function.name).to_uppercase();
                let agg_op = match name.as_str() {
                    "COUNT" => AggOp::Count,
                    "SUM" => AggOp::Sum,
                    "AVG" => AggOp::Avg,
                    "MIN" => AggOp::Min,
                    "MAX" => AggOp::Max,
                    _ => unimplemented!("Unsupported aggregation function: {:?}", function),
                };
                let args = match &function.args {
                    sqlparser::ast::FunctionArguments::List(args) => &args.args,
                    _ => unimplemented!("Unsupported aggregation function: {:?}", function),
                };
                if args.len() != 1 {
                    unimplemented!("Unsupported aggregation function: {:?}", function);
                }
                let function_arg_expr = match &args[0] {
                    sqlparser::ast::FunctionArg::Named { arg, .. } => arg,
                    sqlparser::ast::FunctionArg::Unnamed(arg) => arg,
                };

                let agg_col_id = self.col_id_gen.next();
                match function_arg_expr {
                    sqlparser::ast::FunctionArgExpr::Expr(expr) => {
                        match self.process_expr(&expr, Some(0)) {
                            Ok(expr) => {
                                if let Expr::ColRef { id } = expr {
                                    aggs.push((agg_col_id, (id, agg_op)));
                                    (plan, Expr::col_ref(agg_col_id))
                                } else {
                                    plan = plan.map(
                                        true,
                                        &self.enabled_rules,
                                        &self.col_id_gen,
                                        [(agg_col_id, expr)],
                                    );
                                    aggs.push((agg_col_id, (agg_col_id, agg_op)));
                                    (plan, Expr::col_ref(agg_col_id))
                                }
                            }
                            Err(TranslatorError::ColumnNotFound(_)) => {
                                // Search globally.
                                let expr = self.process_expr(&expr, None).unwrap();
                                let col_id = self.col_id_gen.next();
                                plan = plan.map(
                                    true,
                                    &self.enabled_rules,
                                    &self.col_id_gen,
                                    [(col_id, expr)],
                                );
                                aggs.push((agg_col_id, (col_id, agg_op)));
                                (plan, Expr::col_ref(agg_col_id))
                            }
                            _ => unimplemented!("Unsupported expression: {:?}", expr),
                        }
                    }
                    sqlparser::ast::FunctionArgExpr::QualifiedWildcard(_) => {
                        unimplemented!("QualifiedWildcard is not supported yet")
                    }
                    sqlparser::ast::FunctionArgExpr::Wildcard => {
                        // Wildcard is only supported for COUNT
                        // If wildcard, just need to return Int(1) as it returns the count of rows
                        if matches!(agg_op, AggOp::Count) {
                            let col_id = self.col_id_gen.next();
                            plan = plan.map(
                                true,
                                &self.enabled_rules,
                                &self.col_id_gen,
                                [(col_id, Expr::int(1))],
                            );
                            aggs.push((agg_col_id, (col_id, agg_op)));
                            (plan, Expr::col_ref(agg_col_id))
                        } else {
                            panic!("Wildcard is only supported for COUNT");
                        }
                    }
                }
            }
            sqlparser::ast::Expr::Nested(expr) => {
                self.process_aggregation_arguments(plan, expr, aggs)
            }
            _ => unimplemented!("Unsupported expression: {:?}", expr),
        }
    }

    fn process_expr(
        &self,
        expr: &sqlparser::ast::Expr,
        distance: Option<usize>,
    ) -> Result<Expr, TranslatorError> {
        match expr {
            sqlparser::ast::Expr::Identifier(ident) => {
                let id = if let Some(distance) = distance {
                    self.env.get_at(distance, &ident.value)
                } else {
                    self.env.get(&ident.value)
                };
                let id = id.ok_or(translation_err!(
                    ColumnNotFound,
                    "{}, env: {}",
                    ident.value,
                    self.env
                        .columns
                        .borrow()
                        .iter()
                        .map(|(k, v)| format!("{}:{}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))?;
                Ok(Expr::col_ref(id))
            }
            sqlparser::ast::Expr::CompoundIdentifier(idents) => {
                let name = idents
                    .iter()
                    .map(|i| i.value.clone())
                    .collect::<Vec<_>>()
                    .join(".");
                let id = if let Some(distance) = distance {
                    self.env.get_at(distance, &name)
                } else {
                    self.env.get(&name)
                };
                let id = id.ok_or(translation_err!(
                    ColumnNotFound,
                    "{}, env: {}",
                    name,
                    self.env
                        .columns
                        .borrow()
                        .iter()
                        .map(|(k, v)| format!("{}:{}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))?;
                Ok(Expr::col_ref(id))
            }
            sqlparser::ast::Expr::BinaryOp { left, op, right } => {
                use sqlparser::ast::BinaryOperator::*;
                let left = self.process_expr(left, distance)?;
                let right = self.process_expr(right, distance)?;
                let bin_op = match op {
                    And => BinaryOp::And,
                    Or => BinaryOp::Or,
                    Plus => BinaryOp::Add,
                    Minus => BinaryOp::Sub,
                    Multiply => BinaryOp::Mul,
                    Divide => BinaryOp::Div,
                    Eq => BinaryOp::Eq,
                    NotEq => BinaryOp::Neq,
                    Lt => BinaryOp::Lt,
                    Gt => BinaryOp::Gt,
                    LtEq => BinaryOp::Le,
                    GtEq => BinaryOp::Ge,
                    _ => {
                        return Err(translation_err!(
                            UnsupportedSQL,
                            "Unsupported binary operator: {:?}",
                            op
                        ));
                    }
                };
                Ok(Expr::binary(bin_op, left, right))
            }
            sqlparser::ast::Expr::Value(value) => match value {
                sqlparser::ast::Value::Number(num, _) => Ok(Expr::int(num.parse().unwrap())),
                _ => Err(translation_err!(
                    UnsupportedSQL,
                    "Unsupported value: {:?}",
                    value
                )),
            },
            sqlparser::ast::Expr::Exists { subquery, negated } => {
                let mut translator = Translator::new_with_outer(
                    &self.catalog_ref,
                    &self.enabled_rules,
                    &self.col_id_gen,
                    &self.env,
                );
                let subquery = translator.process_query(subquery)?;
                let mut plan = subquery.plan;
                // Add count(*) to the subquery
                let col_id1 = translator.col_id_gen.next();
                plan = plan.map(
                    true,
                    &translator.enabled_rules,
                    &translator.col_id_gen,
                    [(col_id1, Expr::int(1))],
                );
                let col_id2 = translator.col_id_gen.next();
                plan = plan.aggregate(vec![], vec![(col_id2, (col_id1, AggOp::Count))]);
                // Add count(*) > 0  to the subquery
                let exists_expr = if *negated {
                    Expr::binary(BinaryOp::Le, Expr::col_ref(col_id2), Expr::int(0))
                } else {
                    Expr::binary(BinaryOp::Gt, Expr::col_ref(col_id2), Expr::int(0))
                };
                let col_id3 = self.col_id_gen.next();
                plan = plan.map(
                    true,
                    &translator.enabled_rules,
                    &translator.col_id_gen,
                    [(col_id3, exists_expr)],
                );
                // Add project col 'count(*) > 0' to the subquery
                plan = plan.project(
                    true,
                    &translator.enabled_rules,
                    &translator.col_id_gen,
                    [col_id3].into_iter().collect(),
                );
                Ok(Expr::subquery(plan))
            }
            sqlparser::ast::Expr::AnyOp {
                left,
                compare_op,
                right,
            } => {
                let left = self.process_expr(left, distance)?;
                let right = self.process_expr(right, distance)?;
                let bin_op = match compare_op {
                    sqlparser::ast::BinaryOperator::Eq => BinaryOp::Eq,
                    sqlparser::ast::BinaryOperator::NotEq => BinaryOp::Neq,
                    sqlparser::ast::BinaryOperator::Lt => BinaryOp::Lt,
                    sqlparser::ast::BinaryOperator::Gt => BinaryOp::Gt,
                    sqlparser::ast::BinaryOperator::LtEq => BinaryOp::Le,
                    sqlparser::ast::BinaryOperator::GtEq => BinaryOp::Ge,
                    _ => {
                        unreachable!()
                    }
                };
                match right {
                    Expr::Subquery { expr } => {
                        let mut plan = *expr;
                        let att = plan.att();
                        if att.len() != 1 {
                            panic!("Subquery in ANY should return only one column")
                        }
                        let col_id = att.iter().next().unwrap();
                        // Super dirty hack to deal with ANY subquery without introducing a new join rule (mark join)
                        // First, we append two columns to the result of the subquery
                        // Expr::int(1) and the result of the binary operation
                        let col_id0 = self.col_id_gen.next();
                        let col_id1 = self.col_id_gen.next();
                        // Case when col is NULL then MIN_INT/2 (Large negative number)
                        //      when left is NULL then MIN_INT/4 (Second large negative number)
                        //      when (left bin_op col) then 1
                        //      else 0
                        let case_expr1 = Expr::Case {
                            expr: None,
                            whens: vec![
                                (
                                    Expr::col_ref(*col_id).eq(Expr::null()),
                                    Expr::int(i64::MIN / 2),
                                ),
                                (left.clone().eq(Expr::null()), Expr::int(i64::MIN / 4)),
                                (
                                    Expr::binary(bin_op, left, Expr::col_ref(*col_id)),
                                    Expr::int(1),
                                ),
                            ],
                            else_expr: Box::new(Expr::int(0)),
                        };
                        plan = plan.map(
                            true,
                            &self.enabled_rules,
                            &self.col_id_gen,
                            [(col_id0, Expr::int(1)), (col_id1, case_expr1)],
                        );
                        // Take the count, max, min of the columns
                        let col_id2 = self.col_id_gen.next();
                        let col_id3 = self.col_id_gen.next();
                        let col_id4 = self.col_id_gen.next();
                        plan = plan.aggregate(
                            vec![],
                            vec![
                                (col_id2, (col_id0, AggOp::Count)),
                                (col_id3, (col_id1, AggOp::Max)),
                                (col_id4, (col_id1, AggOp::Min)),
                            ],
                        );
                        // Case when count(1) == 0 then FALSE
                        //      when max == 1 then TRUE
                        //      when min == MIN_INT/2 then NULL
                        //      when min == MIN_INT/4 then NULL
                        //      else FALSE
                        let case_expr2 = Expr::Case {
                            expr: None,
                            whens: vec![
                                (Expr::col_ref(col_id2).eq(Expr::int(0)), Expr::bool(false)),
                                (Expr::col_ref(col_id3).eq(Expr::int(1)), Expr::bool(true)),
                                (
                                    Expr::col_ref(col_id4).eq(Expr::int(i64::MIN / 2)),
                                    Expr::null(),
                                ),
                                (
                                    Expr::col_ref(col_id4).eq(Expr::int(i64::MIN / 4)),
                                    Expr::null(),
                                ),
                            ],
                            else_expr: Box::new(Expr::int(0)),
                        };
                        let col_id5 = self.col_id_gen.next();
                        plan = plan
                            .map(
                                true,
                                &self.enabled_rules,
                                &self.col_id_gen,
                                [(col_id5, case_expr2)],
                            )
                            .project(
                                true,
                                &self.enabled_rules,
                                &self.col_id_gen,
                                [col_id5].into_iter().collect(),
                            );
                        Ok(Expr::subquery(plan))
                    }
                    _ => {
                        unimplemented!("AnyOp should have a subquery on the right side")
                    }
                }
            }
            sqlparser::ast::Expr::Subquery(query) => {
                let mut translator = Translator::new_with_outer(
                    &self.catalog_ref,
                    &self.enabled_rules,
                    &self.col_id_gen,
                    &self.env,
                );
                let subquery = translator.process_query(query)?;
                let plan = subquery.plan;
                let att = plan.att();
                if att.len() != 1 {
                    panic!("Subquery returns more than one column")
                }
                Ok(Expr::subquery(plan))
            }
            _ => Err(translation_err!(
                UnsupportedSQL,
                "Unsupported expression: {:?}",
                expr
            )),
        }
    }
}

// Helper functions
fn get_name(name: &sqlparser::ast::ObjectName) -> String {
    name.0
        .iter()
        .map(|i| i.value.clone())
        .collect::<Vec<_>>()
        .join(".")
}

fn is_valid_alias(alias: &str) -> bool {
    alias.chars().all(|c| c.is_alphanumeric() || c == '_')
}

fn has_agg(expr: &sqlparser::ast::Expr) -> bool {
    use sqlparser::ast::Expr::*;
    match expr {
        Identifier(_) => false,
        CompoundIdentifier(_) => false,
        Value(_) => false,
        TypedString { .. } => false,

        BinaryOp { left, op: _, right } => has_agg(left) || has_agg(right),
        Function(function) => match get_name(&function.name).to_uppercase().as_str() {
            "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" => true,
            _ => false,
        },
        Nested(expr) => has_agg(expr),
        _ => unimplemented!("Unsupported expression: {:?}", expr),
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use sqlparser::dialect::{DuckDbDialect, PostgreSqlDialect};

    use crate::{
        catalog::{Catalog, Column, DataType, Schema, Table},
        col_id_generator::ColIdGenerator,
        rules::{Rule, Rules},
        translator::Translator,
    };

    fn get_test_catalog() -> Catalog {
        let catalog = Catalog::new();
        catalog.add_table(Table::new(
            "t1",
            Schema::new(vec![
                Column::new("a", DataType::Int),
                Column::new("b", DataType::Int),
                Column::new("p", DataType::Int),
                Column::new("q", DataType::Int),
                Column::new("r", DataType::Int),
            ]),
        ));
        catalog.add_table(Table::new(
            "t2",
            Schema::new(vec![
                Column::new("c", DataType::Int),
                Column::new("d", DataType::Int),
            ]),
        ));
        catalog.add_table(Table::new(
            "t3",
            Schema::new(vec![
                Column::new("e", DataType::Int),
                Column::new("f", DataType::Int),
            ]),
        ));
        catalog
    }

    fn parse_sql(sql: &str) -> sqlparser::ast::Query {
        use sqlparser::dialect::GenericDialect;
        use sqlparser::parser::Parser;

        let dialect = DuckDbDialect {};
        let statements = Parser::new(&dialect)
            .try_with_sql(&sql)
            .unwrap()
            .parse_statements()
            .unwrap();
        let query = {
            let statement = statements.into_iter().next().unwrap();
            if let sqlparser::ast::Statement::Query(query) = statement {
                query
            } else {
                panic!("Expected a query");
            }
        };
        *query
    }

    fn get_translator() -> Translator {
        let catalog = Rc::new(get_test_catalog());
        let enabled_rules = Rc::new(Rules::default());
        // enabled_rules.disable(Rule::Decorrelate);
        // enabled_rules.disable(Rule::Hoist);
        // enabled_rules.disable(Rule::ProjectionPushdown);
        let col_id_gen = Rc::new(ColIdGenerator::new());
        Translator::new(&catalog, &enabled_rules, &col_id_gen)
    }

    fn get_plan(sql: &str) -> String {
        let query = parse_sql(sql);
        let mut translator = get_translator();
        let query = translator.process_query(&query).unwrap();
        query.plan.pretty_string()
    }

    #[test]
    fn parse_from_clause() {
        let sql = "SELECT a FROM t1 WHERE a = 1 AND b = 2";
        println!("{}", get_plan(sql));
    }

    #[test]
    #[should_panic]
    fn parse_from_with_subquery() {
        let sql = "SELECT a FROM (SELECT a FROM t1) WHERE a = 1 AND b = 2";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_from_with_subquery_2() {
        let sql = "SELECT a FROM (SELECT a, b FROM t1) AS t WHERE a = 1 AND b = 2";
        println!("{}", get_plan(sql));
    }

    #[test]
    #[should_panic]
    fn parse_from_with_subquery_and_alias() {
        let sql = "SELECT a FROM (SELECT a FROM t1) AS t WHERE a = 1 AND b = 2";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_from_with_join() {
        let sql = "SELECT a FROM t1 JOIN t2 ON t1.a = t2.c WHERE a = 1 AND b = 2";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_from_with_multiple_joins() {
        let sql =
            "SELECT a FROM t1 JOIN t2 ON t1.a = t2.c JOIN t3 ON t2.d = t3.e WHERE a = 1 AND b = 2";
        println!("{}", get_plan(sql));
    }

    #[test]
    #[should_panic]
    fn parse_from_with_subquery_joins() {
        let sql = "SELECT a FROM (SELECT a FROM t1) AS t1 JOIN (SELECT c FROM t2) AS t2 ON t1.a = t2.c WHERE a = 1 AND b = 2";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_from_with_subquery_joins_2() {
        let sql = "SELECT a FROM (SELECT a, b FROM t1) AS t1 JOIN (SELECT c, d FROM t2) AS t2 ON t1.a = t2.c WHERE a = 1 AND b = 2";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_where_clause() {
        let sql = "SELECT a FROM t1 WHERE a = 1 AND b = 2";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_subquery() {
        let sql = "SELECT a, x, y FROM t1, (SELECT COUNT(*) AS x, SUM(c) as y FROM t2 WHERE c = a)";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_subquery_2() {
        // This actually makes sense if we consider AVG(b) as AVG(0+b)
        let sql = "SELECT a, x, y FROM t1, (SELECT AVG(b) AS x, SUM(d) as y FROM t2 WHERE c = a)";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_subquery_3() {
        let sql = "SELECT a, k, x, y FROM t1, (SELECT b as k, c as x, d as y FROM t2 WHERE c = a)";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_subquery_with_alias() {
        let sql =
            "SELECT a, x, y FROM t1, (SELECT COUNT(a) AS x, SUM(b) as y FROM t2 WHERE c = a) AS t2";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_subquery_with_same_tables() {
        let sql = "SELECT x, y FROM (SELECT a as x FROM t1), (SELECT a as y FROM t1) WHERE x = y";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_joins_with_same_name() {
        let sql = "SELECT * FROM t1 as a, t1 as b";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parser_aggregate() {
        let sql = "SELECT COUNT(a), SUM(b) FROM t1";
        println!("{}", get_plan(sql));
    }

    #[test]
    fn parse_subquery_where() {
        let sql = "SELECT a FROM t1 WHERE exists (SELECT * FROM t2 WHERE c = a)";
        println!("{}", get_plan(sql));
    }

    // #[test]
    // fn parse_subquery_where_any() {
    //     let sql = "SELECT * FROM t1 WHERE a IN (SELECT * FROM t2 WHERE c = a)";
    //     println!("{}", get_plan(sql));
    // }
}

// Subquery types
// 1. Select clause
//   a. Scalar subquery. A subquery that returns a single row.
//   b. EXISTS subquery. Subquery can return multiple rows.
//   c. ANY subquery. Subquery can return multiple rows.
// 2. From clause
//   a. Subquery can return multiple rows.
//
