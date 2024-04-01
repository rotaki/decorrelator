use super::prelude::*;
use crate::col_id_generator::ColIdGeneratorRef;
use crate::rules::{Rule, RulesRef};
use std::collections::{HashMap, HashSet};

impl RelExpr {
    pub fn project(
        self,
        optimize: bool,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        cols: HashSet<usize>,
    ) -> RelExpr {
        if self.att() == cols {
            return self;
        }

        let outer_refs = self.free();

        if optimize && enabled_rules.is_enabled(&Rule::ProjectionPushdown) {
            match self {
                RelExpr::Project {
                    src,
                    cols: _no_need_cols,
                } => src.project(true, enabled_rules, col_id_gen, cols),
                RelExpr::Map {
                    input,
                    exprs: mut existing_exprs,
                } => {
                    // Remove the mappings that are not used in the projection.
                    existing_exprs.retain(|(id, _)| cols.contains(id));
                    // Pushdown the projection to the source. Note that we don't push
                    // down the projection of outer columns.
                    let mut free: HashSet<usize> = existing_exprs
                        .iter()
                        .flat_map(|(_, expr)| expr.free())
                        .collect();
                    free = free.difference(&outer_refs).cloned().collect();
                    let new_cols = cols.union(&free).cloned().collect();

                    input
                        .project(true, enabled_rules, col_id_gen, new_cols)
                        .map(true, enabled_rules, col_id_gen, existing_exprs)
                        .project(false, enabled_rules, col_id_gen, cols)
                }
                RelExpr::Select { src, predicates } => {
                    // The necessary columns are the free variables of the predicates and the projection columns
                    let free: HashSet<usize> =
                        predicates.iter().flat_map(|pred| pred.free()).collect();
                    let new_cols = cols.union(&free).cloned().collect();
                    src.project(true, enabled_rules, col_id_gen, new_cols)
                        .select(true, enabled_rules, col_id_gen, predicates)
                        .project(false, enabled_rules, col_id_gen, cols)
                }
                RelExpr::Join {
                    join_type,
                    left,
                    right,
                    predicates,
                } => {
                    // The necessary columns are the free variables of the predicates and the projection columns
                    let free: HashSet<usize> =
                        predicates.iter().flat_map(|pred| pred.free()).collect();
                    let new_cols = cols.union(&free).cloned().collect();
                    let left_proj = left.att().intersection(&new_cols).cloned().collect();
                    let right_proj = right.att().intersection(&new_cols).cloned().collect();
                    left.project(true, enabled_rules, col_id_gen, left_proj)
                        .join(
                            true,
                            enabled_rules,
                            col_id_gen,
                            join_type,
                            right.project(true, enabled_rules, col_id_gen, right_proj),
                            predicates,
                        )
                        .project(false, enabled_rules, col_id_gen, cols)
                }
                RelExpr::Rename {
                    src,
                    src_to_dest: mut existing_rename,
                } => {
                    // Remove the mappings that are not used in the projection.
                    existing_rename.retain(|_, dest| cols.contains(dest));
                    // Pushdown the projection to the source. First we need to rewrite the column names
                    let existing_rename_rev: HashMap<usize, usize> = existing_rename
                        .iter()
                        .map(|(src, dest)| (*dest, *src))
                        .collect(); // dest -> src
                    let mut new_cols = HashSet::new();
                    for col in cols {
                        new_cols.insert(existing_rename_rev.get(&col).unwrap_or(&col).clone());
                    }
                    src.project(true, enabled_rules, &col_id_gen, new_cols)
                        .rename_to(existing_rename)
                }
                RelExpr::Scan {
                    table_name,
                    mut column_names,
                } => {
                    column_names.retain(|col| cols.contains(col));
                    RelExpr::scan(table_name, column_names)
                }
                _ => self.project(false, enabled_rules, col_id_gen, cols),
            }
        } else {
            RelExpr::Project {
                src: Box::new(self),
                cols,
            }
        }
    }
}
