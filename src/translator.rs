use std::{borrow::Borrow, collections::HashMap};

use crate::{
    catalog::CatalogRef,
    expressions::{self, BinaryOp, Expr, JoinType, RelExpr},
};

enum TranslatorError {
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
    aliases: HashMap<String, String>,
}

impl Translator {
    // Input: table_name or alias
    // Output: table_name
    fn disambiguate_table_name(&self, name: &str) -> Result<String, TranslatorError> {
        let name = {
            if let Some(name) = self.aliases.get(name) {
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

    // Input: table_name.column_name or alias.column_name or column_name
    // Output: table_name.column_name
    fn disambiguate_column_name(&self, name: &str) -> Result<String, TranslatorError> {
        // Split the name into parts
        let parts: Vec<&str> = name.split('.').collect();
        match parts.len() {
            1 => {
                // If there is only one part, then it is the column name
                // We need to find the table name for this column
                // We will iterate over all the tables and check if the column exists in any of the tables
                let (table, col) = self
                    .catalog_ref
                    .find_column(parts[0])
                    .ok_or(TranslatorError::ColumnNotFound(name.to_string()))?;
                Ok(format!("{}.{}", table.name, col.name))
            }
            2 => {
                // If there are two parts, then the first part is the alias or table name
                // and the second part is the column name
                // We need to find the table name for this alias
                // If the first part is an alias, then we will use the alias to find the table name
                // If the first part is a table name, then we will use the table name as is
                let table_name = self.disambiguate_table_name(parts[0])?;
                // We need to check if the column exists in the table
                if self.catalog_ref.is_valid_column(&table_name, parts[1]) {
                    return Ok(format!("{}.{}", table_name, parts[1]));
                } else {
                    return Err(TranslatorError::ColumnNotFound(name.to_string()));
                }
            }
            _ => Err(TranslatorError::UnsupportedSQL(
                "3 or more parts in a column name is not supported".to_string(),
            )),
        }
    }

    fn process_query(&mut self, query: &sqlparser::ast::Query) -> Result<RelExpr, TranslatorError> {
        let select = match query.body.as_ref() {
            sqlparser::ast::SetExpr::Select(ref select) => select,
            _ => {
                return Err(TranslatorError::UnsupportedSQL(
                    "Only SELECT queries are supported".to_string(),
                ))
            }
        };

        let plan = self.process_from(&select.from)?;

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
        // Join all the tables together
        let mut plan = join_exprs.pop().unwrap();
        for join_expr in join_exprs {
            plan = plan.join(JoinType::CrossJoin, join_expr, vec![]);
        }
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
            // Seperate the condition by and
            let condition = condition.map(|cond| cond.split_conjunction());
            plan = plan.join(join_type, right, condition.unwrap_or(vec![]));
        }
        Ok(plan)
    }

    fn process_table_factor(
        &mut self,
        table_factor: &sqlparser::ast::TableFactor,
    ) -> Result<RelExpr, TranslatorError> {
        match table_factor {
            sqlparser::ast::TableFactor::Table { name, alias, .. } => {
                let disambiguated_name = self.disambiguate_table_name(&get_name(&name))?;
                // Add the alias to the aliases map
                if let Some(alias) = alias {
                    self.aliases
                        .insert(alias.name.value.clone(), disambiguated_name.clone());
                }
                let column_names = self.catalog_ref.get_col_ids_of_table(&disambiguated_name);
                Ok(RelExpr::Scan {
                    table_name: disambiguated_name,
                    column_names,
                })
            }
            _ => Err(TranslatorError::UnsupportedSQL(
                "Only table names are supported".to_string(),
            )),
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
            _ => Err(TranslatorError::UnsupportedSQL(
                "Unsupported join operator".to_string(),
            )),
        }
    }

    fn process_expr(&self, expr: &sqlparser::ast::Expr) -> Result<Expr, TranslatorError> {
        match expr {
            sqlparser::ast::Expr::Identifier(ident) => {
                let disambiguated_name = self.disambiguate_column_name(&ident.value)?;
                let id = self.catalog_ref.get_col_id(&disambiguated_name).unwrap();
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
                        return Err(TranslatorError::UnsupportedSQL(
                            "Unsupported binary operator".to_string(),
                        ))
                    }
                };
                Ok(Expr::binary(bin_op, left, right))
            }
            sqlparser::ast::Expr::Value(value) => match value {
                sqlparser::ast::Value::Number(num, _) => Ok(Expr::int(num.parse().unwrap())),
                _ => Err(TranslatorError::UnsupportedSQL(
                    "Unsupported value".to_string(),
                )),
            },
            _ => Err(TranslatorError::UnsupportedSQL(
                "Unsupported expression".to_string(),
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
