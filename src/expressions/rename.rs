use super::prelude::*;
use crate::col_id_generator::ColIdGeneratorRef;
use crate::rules::{Rule, RulesRef};
use std::collections::HashMap;

impl RelExpr {
    // Rename the output columns of the relational expression
    // Output: RelExpr, HashMap<old_col_id, new_col_id>
    pub fn rename(
        self,
        _enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
    ) -> (RelExpr, HashMap<usize, usize>) {
        let atts = self.att();
        let cols: HashMap<usize, usize> = atts
            .into_iter()
            .map(|old_col_id| (old_col_id, col_id_gen.next()))
            .collect();
        (self.rename_to(cols.clone()), cols)
    }

    pub(crate) fn rename_to(self, src_to_dest: HashMap<usize, usize>) -> RelExpr {
        if let RelExpr::Rename {
            src,
            src_to_dest: mut existing_rename,
        } = self
        {
            for value in existing_rename.values_mut() {
                *value = *src_to_dest.get(value).unwrap_or(value);
            }
            src.rename_to(existing_rename)
        } else {
            RelExpr::Rename {
                src: Box::new(self.clone()),
                src_to_dest,
            }
        }
    }
}
