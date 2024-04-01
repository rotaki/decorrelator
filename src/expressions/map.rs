use super::prelude::*;
use crate::col_id_generator::ColIdGeneratorRef;
use crate::rules::{Rule, RulesRef};
use std::collections::HashSet;

impl RelExpr {
    pub fn map(
        self,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        exprs: impl IntoIterator<Item = (usize, Expr)>,
    ) -> RelExpr {
        let mut exprs: Vec<(usize, Expr)> = exprs.into_iter().collect();

        if exprs.is_empty() {
            return self;
        }

        if enabled_rules.is_enabled(&Rule::Hoist) {
            for i in 0..exprs.len() {
                // Only hoist expressions with subqueries
                if exprs[i].1.has_subquery() {
                    let (id, expr) = exprs.swap_remove(i);
                    return self.map(enabled_rules, col_id_gen, exprs).hoist(
                        enabled_rules,
                        col_id_gen,
                        id,
                        expr,
                    );
                }
            }
        }

        match self {
            RelExpr::Map {
                input,
                exprs: mut existing_exprs,
            } => {
                // If the free variables of the new exprs (exprs) does not intersect with the attrs
                // of the existing exprs (existing_exprs), then we can push the new exprs to existing_exprs.
                let atts = existing_exprs
                    .iter()
                    .map(|(id, _)| *id)
                    .collect::<HashSet<_>>();
                let (mut push_down, keep): (Vec<_>, Vec<_>) = exprs
                    .into_iter()
                    .partition(|(_, expr)| expr.free().is_disjoint(&atts));
                existing_exprs.append(&mut push_down);
                RelExpr::Map {
                    input: Box::new(input.map(enabled_rules, col_id_gen, existing_exprs)),
                    exprs: keep,
                }
            }
            _ => RelExpr::Map {
                input: Box::new(self),
                exprs,
            },
        }
    }
}
