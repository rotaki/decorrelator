use super::prelude::*;
use crate::col_id_generator::ColIdGeneratorRef;
use crate::rules::{Rule, RulesRef};
use std::collections::{HashMap, HashSet};

impl RelExpr {
    pub fn join(
        self,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        join_type: JoinType,
        other: RelExpr,
        mut predicates: Vec<Expr>,
    ) -> RelExpr {
        predicates = predicates
            .into_iter()
            .flat_map(|expr| expr.split_conjunction())
            .collect();

        if !predicates.is_empty()
            && matches!(
                join_type,
                JoinType::Inner | JoinType::LeftOuter | JoinType::CrossJoin
            )
        {
            let (push_down, keep): (Vec<_>, Vec<_>) =
                predicates.iter().partition(|pred| pred.bound_by(&self));
            if !push_down.is_empty() {
                // This condition is necessary to avoid infinite recursion
                let push_down = push_down.into_iter().map(|expr| expr.clone()).collect();
                let keep = keep.into_iter().map(|expr| expr.clone()).collect();
                return self.select(enabled_rules, col_id_gen, push_down).join(
                    enabled_rules,
                    col_id_gen,
                    join_type,
                    other,
                    keep,
                );
            }
        }

        if !predicates.is_empty()
            && matches!(
                join_type,
                JoinType::Inner | JoinType::RightOuter | JoinType::CrossJoin
            )
        {
            let (push_down, keep): (Vec<_>, Vec<_>) =
                predicates.iter().partition(|pred| pred.bound_by(&other));
            if !push_down.is_empty() {
                // This condition is necessary to avoid infinite recursion
                let push_down = push_down.into_iter().map(|expr| expr.clone()).collect();
                let keep = keep.into_iter().map(|expr| expr.clone()).collect();
                return self.join(
                    enabled_rules,
                    col_id_gen,
                    join_type,
                    other.select(enabled_rules, col_id_gen, push_down),
                    keep,
                );
            }
        }

        // If the remaining predicates are bound by the left and right sides
        // (don't contain any free variables), we can turn the cross join into an inner join.
        // If they contain free variables, then there is a subquery that needs to be evaluated
        // for each row of the cross join.
        if matches!(join_type, JoinType::CrossJoin) && !predicates.is_empty() {
            let free = predicates
                .iter()
                .flat_map(|expr| expr.free())
                .collect::<HashSet<_>>();
            let atts = self.att().union(&other.att()).cloned().collect();
            if free.is_subset(&atts) {
                return self.join(
                    enabled_rules,
                    col_id_gen,
                    JoinType::Inner,
                    other,
                    predicates,
                );
            }
        }

        RelExpr::Join {
            join_type,
            left: Box::new(self),
            right: Box::new(other),
            predicates,
        }
    }
}
