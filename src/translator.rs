use std::{
    collections::{HashMap, HashSet},
    vec,
};

use crate::{
    catalog::CatalogRef,
    expressions::{AggOp, BinaryOp, Expr, JoinType, OptimizerCtx, RelExpr},
};

#[derive(Debug)]
pub enum TranslatorError {
    ColumnNotFound(String),
    TableNotFound(String),
    InvalidSQL(String),
    UnsupportedSQL(String),
}

/// This struct will be used to translate SQL expression into a query plan.
/// The translator will be responsible for doing the semantic analysis of the SQL expression
/// and validating the expression.
pub struct Translator {
    catalog_ref: CatalogRef,
    table_aliases: HashMap<String, String>, // table_alias -> table_name
    column_aliases: HashMap<String, usize>, // column_alias -> col_id
    referenced_tables: HashSet<String>, // all the tables referenced in the query (FROM, JOIN, Subquery)
    opt_ctx: OptimizerCtx,
}

impl Translator {
    pub fn new(catalog_ref: CatalogRef) -> Self {
        let opt_ctx = OptimizerCtx::new(catalog_ref.get_col_ids());
        Translator {
            catalog_ref,
            table_aliases: HashMap::new(),
            column_aliases: HashMap::new(),
            referenced_tables: HashSet::new(),
            opt_ctx,
        }
    }

    // Input: table_name or alias
    // Output: table_name
    fn disambiguate_table_name(&self, name: &str) -> Result<String, TranslatorError> {
        let name = {
            if let Some(name) = self.table_aliases.get(name) {
                // If there is an alias, then we prioritize the alias
                name
            } else {
                name
            }
        };
        if self.catalog_ref.is_valid_table(name) {
            Ok(name.into())
        } else {
            Err(TranslatorError::TableNotFound(name.into()))
        }
    }

    // Input: table_name.column_name or alias_table.column_name or column_name or alias_column
    //        (note that table_name.alias_column, using period in alias_column is not supported)
    // Output: col_id
    fn disambiguate_column(&self, name: &str) -> Result<usize, TranslatorError> {
        // Split the name into parts
        let parts: Vec<&str> = name.split('.').collect();
        match parts.len() {
            1 => {
                // If there is only one part, then it is the column name or alias
                if let Some(col_id) = self.column_aliases.get(parts[0]) {
                    // If there is an alias, then we prioritize the alias
                    Ok(*col_id)
                } else {
                    // We need to find the table name for this column
                    // We will iterate over all the referenced tables and find the column
                    for table_name in &self.referenced_tables {
                        if self.catalog_ref.is_valid_column(table_name, parts[0]) {
                            let disambiguated_name = format!("{}.{}", table_name, parts[0]);
                            return Ok(self.catalog_ref.get_col_id(&disambiguated_name).unwrap());
                        }
                    }
                    Err(TranslatorError::ColumnNotFound(format!(
                        "{} not found in any of the referenced tables",
                        parts[0]
                    )))
                }
            }
            2 => {
                // If there are two parts, then the first part is the alias or table name
                // and the second part is the column name
                // We need to find the table name for this alias
                // If the first part is an alias, then we will use the alias to find the table name
                // If the first part is a table name, then we will use the table name as is
                let table_name = self.disambiguate_table_name(parts[0])?;
                // We need to check if we reference the table in the query
                // and if the column is valid
                if self.referenced_tables.contains(&table_name)
                    && self.catalog_ref.is_valid_column(&table_name, parts[1])
                {
                    let disambiguated_name = format!("{}.{}", table_name, parts[1]);
                    Ok(self.catalog_ref.get_col_id(&disambiguated_name).unwrap())
                } else {
                    return Err(TranslatorError::ColumnNotFound(name.to_string()));
                }
            }
            _ => Err(TranslatorError::UnsupportedSQL(format!(
                "3 or more parts in column name: {}",
                name
            ))),
        }
    }

    pub fn process_query(
        &mut self,
        query: &sqlparser::ast::Query,
    ) -> Result<RelExpr, TranslatorError> {
        let select = match query.body.as_ref() {
            sqlparser::ast::SetExpr::Select(ref select) => select,
            _ => {
                return Err(TranslatorError::UnsupportedSQL(
                    "Only SELECT queries are supported".to_string(),
                ))
            }
        };

        self.add_tables_to_referenced_tables(&select.from)?;

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

        Ok(plan)
    }

    fn process_from(
        &mut self,
        from: &[sqlparser::ast::TableWithJoins],
    ) -> Result<RelExpr, TranslatorError> {
        if from.is_empty() {
            return Err(TranslatorError::InvalidSQL(
                "FROM clause is required".to_string(),
            ));
        }
        let mut join_exprs = vec![];
        for table_with_joins in from {
            let join_expr = self.process_table_with_joins(table_with_joins)?;
            join_exprs.push(join_expr);
        }
        // Join all the tables together using cross join
        let mut plan = join_exprs.pop().unwrap();
        for join_expr in join_exprs {
            plan = plan.join(&self.opt_ctx, JoinType::CrossJoin, join_expr, vec![]);
        }
        Ok(plan)
    }

    fn process_where(
        &mut self,
        plan: RelExpr,
        where_clause: &Option<sqlparser::ast::Expr>,
    ) -> Result<RelExpr, TranslatorError> {
        if let Some(expr) = where_clause {
            let expr = self.process_expr(expr)?;
            Ok(plan.select(&self.opt_ctx, vec![expr]))
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
                    for table_name in &self.referenced_tables {
                        let col_ids = self.catalog_ref.get_col_ids_of_table(&table_name);
                        projected_cols.extend(col_ids);
                    }
                }
                sqlparser::ast::SelectItem::UnnamedExpr(expr) => {
                    if !has_agg(expr) {
                        let expr = self.process_expr(expr)?;
                        let col_id = if let Expr::ColRef { id } = expr {
                            id
                        } else {
                            // create a new col_id for the expression
                            let col_id = self.opt_ctx.next();
                            plan = plan.map(&self.opt_ctx, [(col_id, expr)]);
                            col_id
                        };
                        projected_cols.insert(col_id);
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
                            let col_id = self.opt_ctx.next();
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
                        let expr = self.process_expr(expr)?;
                        let col_id = if let Expr::ColRef { id } = expr {
                            id
                        } else {
                            // create a new col_id for the expression
                            let col_id = self.opt_ctx.next();
                            plan = plan.map(&self.opt_ctx, [(col_id, expr)]);
                            col_id
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
                            let col_id = self.opt_ctx.next();
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
                        self.column_aliases.insert(alias_name, col_id);
                    } else {
                        return Err(TranslatorError::InvalidSQL(format!(
                            "Invalid alias name: {}",
                            alias_name
                        )));
                    }
                }
                _ => {
                    return Err(TranslatorError::UnsupportedSQL(format!(
                        "Unsupported select item: {:?}",
                        item
                    )))
                }
            }
        }

        if !aggregations.is_empty() {
            let group_by = match group_by {
                sqlparser::ast::GroupByExpr::All => Err(TranslatorError::UnsupportedSQL(
                    "GROUP BY ALL is not supported".to_string(),
                ))?,
                sqlparser::ast::GroupByExpr::Expressions(exprs) => {
                    let mut group_by = Vec::new();
                    for expr in exprs {
                        let expr = self.process_expr(expr)?;
                        let col_id = if let Expr::ColRef { id } = expr {
                            id
                        } else {
                            // create a new col_id for the expression
                            let col_id = self.opt_ctx.next();
                            plan = plan.map(&self.opt_ctx, [(col_id, expr)]);
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
        plan = plan.map(&self.opt_ctx, maps); // This map corresponds to the Level3 in the comment above
        plan = plan.project(&self.opt_ctx, projected_cols);
        Ok(plan)
    }

    fn process_table_with_joins(
        &mut self,
        table_with_joins: &sqlparser::ast::TableWithJoins,
    ) -> Result<RelExpr, TranslatorError> {
        let mut plan = self.process_table_factor(&table_with_joins.relation)?;
        for join in &table_with_joins.joins {
            let right = self.process_table_factor(&join.relation)?;
            let (join_type, condition) = self.process_join_operator(&join.join_operator)?;
            plan = plan.join(
                &self.opt_ctx,
                join_type,
                right,
                condition.into_iter().collect(),
            );
        }
        Ok(plan)
    }

    fn process_table_factor(
        &mut self,
        table_factor: &sqlparser::ast::TableFactor,
    ) -> Result<RelExpr, TranslatorError> {
        match table_factor {
            sqlparser::ast::TableFactor::Table { name, .. } => {
                let disambiguated_name = self.disambiguate_table_name(&get_name(&name))?;
                let column_names = self.catalog_ref.get_col_ids_of_table(&disambiguated_name);
                Ok(RelExpr::Scan {
                    table_name: disambiguated_name,
                    column_names: column_names.into_iter().collect(),
                })
            }
            _ => Err(TranslatorError::UnsupportedSQL(format!(
                "Unsupported table factor: {:?}",
                table_factor
            ))),
        }
    }

    fn process_join_operator(
        &self,
        join_operator: &sqlparser::ast::JoinOperator,
    ) -> Result<(JoinType, Option<Expr>), TranslatorError> {
        use sqlparser::ast::{JoinConstraint, JoinOperator::*};
        match join_operator {
            Inner(JoinConstraint::On(cond)) => {
                Ok((JoinType::Inner, Some(self.process_expr(cond)?)))
            }
            LeftOuter(JoinConstraint::On(cond)) => {
                Ok((JoinType::LeftOuter, Some(self.process_expr(cond)?)))
            }
            RightOuter(JoinConstraint::On(cond)) => {
                Ok((JoinType::RightOuter, Some(self.process_expr(cond)?)))
            }
            FullOuter(JoinConstraint::On(cond)) => {
                Ok((JoinType::FullOuter, Some(self.process_expr(cond)?)))
            }
            CrossJoin => Ok((JoinType::CrossJoin, None)),
            _ => Err(TranslatorError::UnsupportedSQL(format!(
                "Unsupported join operator: {:?}",
                join_operator
            ))),
        }
    }

    fn process_expr(&self, expr: &sqlparser::ast::Expr) -> Result<Expr, TranslatorError> {
        match expr {
            sqlparser::ast::Expr::Identifier(ident) => {
                let id = self.disambiguate_column(&ident.value)?;
                Ok(Expr::col_ref(id))
            }
            sqlparser::ast::Expr::CompoundIdentifier(idents) => {
                let name = idents
                    .iter()
                    .map(|i| i.value.clone())
                    .collect::<Vec<_>>()
                    .join(".");
                let id = self.disambiguate_column(&name)?;
                Ok(Expr::col_ref(id))
            }
            sqlparser::ast::Expr::BinaryOp { left, op, right } => {
                use sqlparser::ast::BinaryOperator::*;
                let left = self.process_expr(left)?;
                let right = self.process_expr(right)?;
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
                        return Err(TranslatorError::UnsupportedSQL(format!(
                            "Unsupported binary operator: {:?}",
                            op
                        )))
                    }
                };
                Ok(Expr::binary(bin_op, left, right))
            }
            sqlparser::ast::Expr::Value(value) => match value {
                sqlparser::ast::Value::Number(num, _) => Ok(Expr::int(num.parse().unwrap())),
                _ => Err(TranslatorError::UnsupportedSQL(format!(
                    "Unsupported value: {:?}",
                    value
                ))),
            },
            _ => Err(TranslatorError::UnsupportedSQL(format!(
                "Unsupported expression: {:?}",
                expr
            ))),
        }
    }

    // Add all table names to the referenced_tables set
    fn add_tables_to_referenced_tables(
        &mut self,
        from: &Vec<sqlparser::ast::TableWithJoins>,
    ) -> Result<(), TranslatorError> {
        for table_with_joins in from {
            self.add_table_factor_to_referenced_tables(&table_with_joins.relation)?;

            for join in &table_with_joins.joins {
                self.add_table_factor_to_referenced_tables(&join.relation)?;
            }
        }
        Ok(())
    }

    fn add_table_factor_to_referenced_tables(
        &mut self,
        table_factor: &sqlparser::ast::TableFactor,
    ) -> Result<(), TranslatorError> {
        let table_name = match &table_factor {
            sqlparser::ast::TableFactor::Table { name, alias, .. } => {
                let name = get_name(name);
                // Add the alias to the aliases map
                if let Some(alias) = alias {
                    let alias_name = alias.name.value.clone();
                    if is_valid_alias(&alias_name) {
                        return Err(TranslatorError::InvalidSQL(format!(
                            "Invalid alias name: {}",
                            alias_name
                        )));
                    }
                    self.table_aliases.insert(alias_name, name.clone());
                }
                name
            }
            _ => Err(TranslatorError::InvalidSQL(format!(
                "Unsupported table factor: {:?}",
                table_factor
            )))?,
        };
        let table_name = self.disambiguate_table_name(&table_name)?;
        self.referenced_tables.insert(table_name);

        Ok(())
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
            sqlparser::ast::Expr::Identifier(_)
            | sqlparser::ast::Expr::CompoundIdentifier(_)
            | sqlparser::ast::Expr::Value(_)
            | sqlparser::ast::Expr::TypedString { .. } => {
                let expr = self.process_expr(expr).unwrap();
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
                if function.args.len() != 1 {
                    unimplemented!("Unsupported aggregation function: {:?}", function);
                }
                let function_arg_expr = match &function.args[0] {
                    sqlparser::ast::FunctionArg::Named { arg, .. } => arg,
                    sqlparser::ast::FunctionArg::Unnamed(arg) => arg,
                };

                let agg_col_id = self.opt_ctx.next();
                match function_arg_expr {
                    sqlparser::ast::FunctionArgExpr::Expr(expr) => {
                        let expr = self.process_expr(&expr).unwrap();
                        if let Expr::ColRef { id } = expr {
                            aggs.push((agg_col_id, (id, agg_op)));
                        } else {
                            let col_id = self.opt_ctx.next();
                            plan = plan.map(&self.opt_ctx, [(col_id, expr)]);
                            aggs.push((agg_col_id, (col_id, agg_op)));
                        }
                        (plan, Expr::col_ref(agg_col_id))
                    }
                    sqlparser::ast::FunctionArgExpr::QualifiedWildcard(_) => {
                        unimplemented!("QualifiedWildcard is not supported yet")
                    }
                    sqlparser::ast::FunctionArgExpr::Wildcard => {
                        // Wildcard is only supported for COUNT
                        // If wildcard, just need to return Int(1) as it returns the count of rows
                        if matches!(agg_op, AggOp::Count) {
                            let col_id = self.opt_ctx.next();
                            plan = plan.map(&self.opt_ctx, [(col_id, Expr::int(1))]);
                            aggs.push((col_id, (col_id, agg_op)));
                            (plan, Expr::col_ref(col_id))
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

    use crate::{
        catalog::{Catalog, Column, DataType, Schema, Table},
        translator::Translator,
    };

    fn get_test_catalog() -> Catalog {
        let catalog = Catalog::new();
        catalog.add_table(Table::new(
            "t1",
            Schema::new(vec![
                Column::new("a", DataType::Int),
                Column::new("b", DataType::Int),
            ]),
        ));
        catalog.add_table(Table::new(
            "t2",
            Schema::new(vec![
                Column::new("a", DataType::Int),
                Column::new("b", DataType::Int),
            ]),
        ));
        catalog.add_table(Table::new(
            "t3",
            Schema::new(vec![
                Column::new("c", DataType::Int),
                Column::new("d", DataType::Int),
            ]),
        ));
        catalog
    }

    fn parse_sql(sql: &str) -> sqlparser::ast::Query {
        use sqlparser::dialect::GenericDialect;
        use sqlparser::parser::Parser;

        let dialect = GenericDialect {};
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

    #[test]
    fn parse_where_clause() {
        let sql = "SELECT a FROM t1 WHERE a = 1 AND b = 2";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_join_1() {
        let sql = "SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.a = t2.b";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_join_2() {
        let sql = "SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.a = t2.b AND t1.b = t2.a AND t1.a = 1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_join_3() {
        let sql = "SELECT t1.a, t2.b FROM t1, t2";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_where_1() {
        let sql = "SELECT a FROM t1 WHERE a = 1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_where_2() {
        let sql = "SELECT a FROM t1 WHERE a = 1 AND b = 2";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_join_with_where_1() {
        let sql = "SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.a = t2.b WHERE t1.b = 2";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_join_with_where_2() {
        let sql = "SELECT t1.a, t2.b FROM t1, t2 WHERE t1.a = t2.b AND t1.b = 2";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_projection_1() {
        let sql = "SELECT a, b FROM t1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_projection_2() {
        let sql = "SELECT a + b FROM t1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_projection_3() {
        let sql = "SELECT a + b, a - b FROM t1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_projection_4() {
        let sql = "SELECT a, b, a + b, a - b FROM t1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_projection_5() {
        let sql = "SELECT a, b, a + b, a - b, a = b, a < b, a > b, a <= b, a >= b FROM t1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_projection_6() {
        let sql = "SELECT * FROM t1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_projection_7() {
        let sql = "SELECT a, b FROM t2";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_aggregation_1() {
        let sql = "SELECT COUNT(a), SUM(b), AVG(a + b), MIN(a - b), MAX(a * b) FROM t1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_aggregation_2() {
        let sql = "SELECT COUNT(*) + SUM(a + 4) + 4, AVG(b) FROM t1";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_aggregation_3() {
        let sql =
            "SELECT COUNT(*) + SUM(a + 4) + 4, AVG(b) FROM t1 GROUP BY a, b HAVING a > 1 AND b < 2";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }

    #[test]
    fn parse_aggregation_4() {
        let sql = "SELECT COUNT(*) + SUM(a + d + 3) + 4, AVG(b) FROM t1, t3 GROUP BY a, b HAVING a > 1 AND b < 2";
        let query = parse_sql(sql);

        let catalog = Rc::new(get_test_catalog());
        let mut translator = Translator::new(catalog);
        let plan = translator.process_query(&query).unwrap();
        plan.pretty_print();
    }
}
