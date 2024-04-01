use super::prelude::*;
use crate::col_id_generator::ColIdGeneratorRef;
use crate::rules::{Rule, RulesRef};
use std::collections::HashMap;

impl RelExpr {
    // Make subquery into a FlatMap
    // FlatMap is sometimes called "Apply", "Dependent Join", or "Lateral Join"
    //
    // SQL Query:
    // Table x: a, b
    // Table y: c
    //
    // SELECT x.a, x.b, 4 + (SELECT x.a + y.c FROM y) FROM x
    //
    // Before:
    // ---------------------------------------
    // |  Map to @4                          |
    // |            ------------------------ |
    // |            |  Subquery: @3 + @1   | |
    // |    4 +     |  Scan @3             | |
    // |            ------------------------ |
    // ---------------------------------------
    //                 |
    // ---------------------------------------
    // |  Scan  @1, @2                       |
    // ---------------------------------------

    // After:
    // -------------------------------------------
    // |  Project @1, @2, @4                     |
    // -------------------------------------------
    //                  |
    // -------------------------------------------
    // |  Map to @4                              |
    // |     @lhs_id + @rhs_id                   |
    // -------------------------------------------
    //                  |
    // -------------------------------------------
    // |  FlatMap (@rhs_id <- @3 + @1)           |
    // -------------------------------------------
    //              /                   \
    // ---------------------------     -----------
    // |  Join (@lhs_id <- 4)    |     | @3 + @1 |
    // ---------------------------     -----------
    //          /         \
    // ----------------
    // |  Scan @1, @2 |     4
    // ----------------

    pub(crate) fn hoist(
        self,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        id: usize,
        expr: Expr,
    ) -> RelExpr {
        match expr {
            Expr::Subquery { expr } => {
                let att = expr.att();
                assert!(att.len() == 1);
                let input_col_id = att.iter().next().unwrap();
                if att.len() != 1 {
                    panic!("Subquery has more than one column");
                }
                // Give the column the name that's expected
                let rhs: RelExpr = expr.map(
                    true,
                    enabled_rules,
                    col_id_gen,
                    vec![(id, Expr::col_ref(*input_col_id))],
                );
                self.flatmap(true, enabled_rules, col_id_gen, rhs)
            }
            Expr::Binary { op, left, right } => {
                // Hoist the left, hoist the right, then perform the binary operation
                let lhs_id = col_id_gen.next();
                let rhs_id = col_id_gen.next();
                let att = self.att();
                self.hoist(enabled_rules, col_id_gen, lhs_id, *left)
                    .hoist(enabled_rules, col_id_gen, rhs_id, *right)
                    .map(
                        true,
                        enabled_rules,
                        col_id_gen,
                        [(
                            id,
                            Expr::Binary {
                                op,
                                left: Box::new(Expr::col_ref(lhs_id)),
                                right: Box::new(Expr::col_ref(rhs_id)),
                            },
                        )],
                    )
                    .project(
                        true,
                        enabled_rules,
                        col_id_gen,
                        att.into_iter().chain([id].into_iter()).collect(),
                    )
            }
            Expr::Field { .. } | Expr::ColRef { .. } => {
                self.map(true, enabled_rules, col_id_gen, vec![(id, expr)])
            }
            Expr::Case { .. } => {
                panic!("Case expression is not supported in hoist")
            }
        }
    }
}
