use super::prelude::*;
use crate::col_id_generator::ColIdGeneratorRef;
use crate::rules::{Rule, RulesRef};
use std::collections::{HashMap, HashSet};

impl RelExpr {
    pub fn scan(table_name: String, column_names: Vec<usize>) -> RelExpr {
        RelExpr::Scan {
            table_name,
            column_names,
        }
    }
}
