use super::prelude::*;
use crate::col_id_generator::ColIdGeneratorRef;
use crate::rules::{Rule, RulesRef};
use std::collections::{HashMap, HashSet};

impl RelExpr {
    pub fn project(
        self,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        cols: HashSet<usize>,
    ) -> RelExpr {
        if self.att() == cols {
            return self;
        }

        if enabled_rules.is_enabled(&Rule::ProjectionPushdown) {
            match self {
                RelExpr::Project {
                    src,
                    cols: mut existing_cols,
                } => {
                    // Merge the columns that are projected
                    existing_cols.extend(cols);
                    RelExpr::Project {
                        src,
                        cols: existing_cols,
                    }
                }
                RelExpr::Map {
                    input,
                    exprs: mut existing_exprs,
                } => {
                    // Remove the mappings that are not used in the projection.
                    existing_exprs.retain(|(id, _)| cols.contains(id));
                    let src = if existing_exprs.is_empty() {
                        *input
                    } else {
                        RelExpr::Map {
                            input,
                            exprs: existing_exprs,
                        }
                    };
                    RelExpr::Project {
                        src: Box::new(src),
                        cols,
                    }
                }
                RelExpr::Select { src, predicates } => {
                    // The necessary columns are the free variables of the predicates and the projection columns
                    let free: HashSet<usize> =
                        predicates.iter().flat_map(|pred| pred.free()).collect();
                    let new_cols = cols.union(&free).cloned().collect();
                    RelExpr::Project {
                        src: Box::new(RelExpr::Select {
                            src: Box::new(src.project(enabled_rules, col_id_gen, new_cols)),
                            predicates,
                        }),
                        cols,
                    }
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
                    RelExpr::Project {
                        src: Box::new(RelExpr::Join {
                            join_type,
                            left: Box::new(left.project(enabled_rules, col_id_gen, left_proj)),
                            right: Box::new(right.project(enabled_rules, col_id_gen, right_proj)),
                            predicates,
                        }),
                        cols,
                    }
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
                        .collect();
                    let mut new_cols = HashSet::new();
                    for col in cols {
                        if let Some(src) = existing_rename_rev.get(&col) {
                            new_cols.insert(*src);
                        } else {
                            new_cols.insert(col);
                        }
                    }
                    src.project(enabled_rules, &col_id_gen, new_cols)
                        .rename_to(existing_rename)
                }
                RelExpr::Scan {
                    table_name,
                    mut column_names,
                } => {
                    column_names.retain(|col| cols.contains(col));
                    RelExpr::Scan {
                        table_name,
                        column_names,
                    }
                }
                _ => RelExpr::Project {
                    src: Box::new(self),
                    cols,
                },
            }
        } else {
            RelExpr::Project {
                src: Box::new(self),
                cols,
            }
        }
    }
}
