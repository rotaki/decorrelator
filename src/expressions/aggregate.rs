use super::prelude::*;

impl RelExpr {
    pub fn aggregate(self, group_by: Vec<usize>, aggrs: Vec<(usize, (usize, AggOp))>) -> RelExpr {
        RelExpr::Aggregate {
            src: Box::new(self),
            group_by,
            aggrs,
        }
    }
}
