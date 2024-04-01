use super::prelude::*;
use crate::col_id_generator::ColIdGeneratorRef;
use crate::field::Field;
use crate::rules::{Rule, RulesRef};
use std::collections::{HashMap, HashSet};

impl RelExpr {
    pub fn flatmap(
        self,
        optimize: bool,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        func: RelExpr,
    ) -> RelExpr {
        if optimize && enabled_rules.is_enabled(&Rule::Decorrelate) {
            // Not correlated!
            if func.free().is_empty() {
                return self.join(
                    true,
                    enabled_rules,
                    col_id_gen,
                    JoinType::CrossJoin,
                    func,
                    vec![],
                );
            }

            // Pull up Project
            if let RelExpr::Project { src, mut cols } = func {
                cols.extend(self.att());
                return self.flatmap(true, enabled_rules, col_id_gen, *src).project(
                    false,
                    enabled_rules,
                    col_id_gen,
                    cols,
                );
            }

            // Pull up Maps
            if let RelExpr::Map { input, exprs } = func {
                return self.flatmap(true, enabled_rules, col_id_gen, *input).map(
                    false,
                    enabled_rules,
                    col_id_gen,
                    exprs,
                );
            }

            // Pull up Selects
            if let RelExpr::Select { src, predicates } = func {
                return self.flatmap(true, enabled_rules, col_id_gen, *src).select(
                    false,
                    enabled_rules,
                    col_id_gen,
                    predicates,
                );
            }

            // Pull up Aggregates
            if let RelExpr::Aggregate {
                src,
                group_by,
                aggrs,
            } = func
            {
                // Return result should be self.att() + func.att()
                // func.att() is group_by + aggrs
                let counts: Vec<usize> = aggrs
                    .iter()
                    .filter_map(|(id, (src_id, op))| {
                        if let AggOp::Count = op {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                    .collect();
                if counts.is_empty() {
                    let att = self.att();
                    let group_by: HashSet<usize> = group_by
                        .iter()
                        .cloned()
                        .chain(att.iter().cloned())
                        .collect();
                    return self
                        .flatmap(true, enabled_rules, col_id_gen, *src)
                        .aggregate(group_by.into_iter().collect(), aggrs);
                } else {
                    // Deal with the COUNT BUG
                    let orig = self.clone();

                    // Create a copy of the original plan and rename it. Left join the copy with the src.
                    // Need to replace the free variables in the src with the new column ids.
                    let (mut copy, new_col_ids) = self.rename(&enabled_rules, &col_id_gen);
                    let copy_att = copy.att();
                    let src = src.replace_variables(&new_col_ids);
                    copy = copy
                        .flatmap(true, enabled_rules, col_id_gen, src)
                        .aggregate(
                            group_by
                                .into_iter()
                                .chain(copy_att.iter().cloned())
                                .collect(),
                            aggrs,
                        );
                    // Join the original plan with the copy with the shared columns.
                    let plan = orig.join(
                        false,
                        enabled_rules,
                        col_id_gen,
                        JoinType::LeftOuter,
                        copy,
                        new_col_ids
                            .iter()
                            .map(|(src, dest)| Expr::col_ref(*src).eq(Expr::col_ref(*dest)))
                            .collect(),
                    );
                    // Project the columns except the columns of the copy
                    let att = plan.att();
                    let project_att = att.difference(&copy_att).cloned().collect();
                    return plan
                        .project(false, enabled_rules, col_id_gen, project_att)
                        .map(
                            false,
                            enabled_rules,
                            col_id_gen,
                            counts.into_iter().map(|id| {
                                (
                                    id,
                                    Expr::Case {
                                        expr: Box::new(Expr::col_ref(id)),
                                        whens: [(Expr::Field { val: Field::Null }, Expr::int(0))]
                                            .to_vec(),
                                        else_expr: Box::new(Expr::col_ref(id)),
                                    },
                                )
                            }),
                        );
                }
            }
        }
        RelExpr::FlatMap {
            input: Box::new(self),
            func: Box::new(func),
        }
    }
}
