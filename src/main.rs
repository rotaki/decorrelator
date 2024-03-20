use std::collections::HashSet;

enum Expr {
    ColRef { id: usize },
    Int { val: i64 },
    Eq { left: Box<Expr>, right: Box<Expr> },
}

impl Expr {
    fn col_ref(id: usize) -> Expr {
        Expr::ColRef { id }
    }

    fn int(val: i64) -> Expr {
        Expr::Int { val }
    }

    fn eq(self, other: Expr) -> Expr {
        Expr::Eq {
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    fn print(&self, indent: usize, out: &mut String) {
        match self {
            Expr::ColRef { id } => {
                out.push_str(&format!("@{}", id));
            }
            Expr::Int { val } => {
                out.push_str(&format!("{}", val));
            }
            Expr::Eq { left, right } => {
                left.print(indent, out);
                out.push_str("=");
                right.print(indent, out);
            }
        }
    }
}

// Free variables
// * A column in an expression that is not bound.

// Bound variables
// * A column that gives its values within an expression and is not a
//   parameter that comes from some other context

// Example:
// function(x) {x + y}
// * x is a bound variable
// * y is a free variable

impl Expr {
    fn free(&self) -> HashSet<usize> {
        match self {
            Expr::ColRef { id } => {
                let mut set = HashSet::new();
                set.insert(*id);
                set
            }
            Expr::Int { val: _ } => HashSet::new(),
            Expr::Eq { left, right } => {
                let mut set = left.free();
                set.extend(right.free());
                set
            }
        }
    }

    fn bound_by(&self, rel: &RelExpr) -> bool {
        self.free().is_subset(&rel.att())
    }
}

enum RelExpr {
    Scan {
        table_name: String,
        column_names: Vec<usize>,
    },
    Select {
        src: Box<RelExpr>,
        predicates: Vec<Expr>,
    },
    Join {
        left: Box<RelExpr>,
        right: Box<RelExpr>,
        predicates: Vec<Expr>,
    },
    Project {
        src: Box<RelExpr>,
        cols: HashSet<usize>,
    },
}

impl RelExpr {
    fn scan(table_name: String, column_names: Vec<usize>) -> RelExpr {
        RelExpr::Scan {
            table_name,
            column_names,
        }
    }

    fn select(self, mut predicates: Vec<Expr>) -> RelExpr {
        if let RelExpr::Select {
            src,
            predicates: mut preds,
        } = self
        {
            preds.append(&mut predicates);
            return src.select(preds);
        }

        RelExpr::Select {
            src: Box::new(self),
            predicates,
        }
    }

    fn join(self, other: RelExpr, mut predicates: Vec<Expr>) -> RelExpr {
        // RelExpr::Join {left: Box::new(self), right: Box::new(other), predicates}
        for i in 0..predicates.len() {
            if predicates[i].bound_by(&self) {
                // We can push this predicate to the left side
                let predicate = predicates.swap_remove(i);
                return self.select(vec![predicate]).join(other, predicates);
            }

            if predicates[i].bound_by(&other) {
                // We can push this predicate to the right side
                let predicate = predicates.swap_remove(i);
                return self.join(other.select(vec![predicate]), predicates);
            }
        }

        RelExpr::Join {
            left: Box::new(self),
            right: Box::new(other),
            predicates,
        }
    }

    fn project(self, cols: HashSet<usize>) -> RelExpr {
        RelExpr::Project {
            src: Box::new(self),
            cols,
        }
    }
}

impl RelExpr {
    // Free set of relational expression
    // * The set of columns that are not bound in the expression
    // * From all the columns required to compute the result, remove the columns that are
    //   internally bound.
    //
    // * Examples of internally bound columns:
    //   * The columns that are bound by the source of the expression (e.g. the columns of a table)
    //   * The columns that are bound by the projection of the expression
    //   * The columns that are bound by evaluating an expression
    fn free(&self) -> HashSet<usize> {
        match self {
            RelExpr::Scan { .. } => HashSet::new(),
            RelExpr::Select { src, predicates } => {
                // For each predicate, identify the free columns.
                // Take the set difference of the free columns and the src attribute set.
                let mut set = src.free();
                for pred in predicates {
                    set.extend(pred.free());
                }
                set.difference(&src.att()).cloned().collect()
            }
            RelExpr::Join {
                left,
                right,
                predicates,
            } => {
                let mut set = left.free();
                set.extend(right.free());
                for pred in predicates {
                    set.extend(pred.free());
                }
                set.difference(&left.att().union(&right.att()).cloned().collect())
                    .cloned()
                    .collect()
            }
            RelExpr::Project { src, cols } => {
                // We assume that the cols are in the src attribute set.
                // Otherwise we can't project the columns.
                // Therefore, we don't need to check the free set of the cols.
                src.free()
            }
        }
    }

    // Attribute set of relational expression
    // * The set of columns that are in the result of the expression.
    // * Attribute changes when we do a projection or map the columns to a different name.
    //
    // Difference between "free" and "att"
    // * "free" is the set of columns that we need to evaluate the expression
    // * "att" is the set of columns that we have (the column names of the result of RelExpr)
    fn att(&self) -> HashSet<usize> {
        match self {
            RelExpr::Scan {
                table_name: _,
                column_names,
            } => column_names.iter().cloned().collect(),
            RelExpr::Select { src, .. } => src.att(),
            RelExpr::Join { left, right, .. } => {
                let mut set = left.att();
                set.extend(right.att());
                set
            }
            RelExpr::Project { cols, .. } => cols.clone(),
        }
    }
}

impl RelExpr {
    fn print(&self, indent: usize, out: &mut String) {
        match self {
            RelExpr::Scan {
                table_name,
                column_names,
            } => out.push_str(&format!(
                "{}-> scan({:?}, {:?})\n",
                " ".repeat(indent),
                table_name,
                column_names
            )),
            RelExpr::Select { src, predicates } => {
                out.push_str(&format!("{}-> select(", " ".repeat(indent)));
                let mut split = "";
                for pred in predicates {
                    out.push_str(split);
                    pred.print(0, out);
                    split = " && ";
                }
                out.push_str(")\n");
                src.print(indent + 2, out);
            }
            RelExpr::Join {
                left,
                right,
                predicates,
            } => {
                out.push_str(&format!("{}-> join(", " ".repeat(indent)));
                let mut split = "";
                for pred in predicates {
                    out.push_str(split);
                    pred.print(0, out);
                    split = " && ";
                }
                out.push_str(")\n");
                left.print(indent + 2, out);
                right.print(indent + 2, out);
            }
            RelExpr::Project { src, cols } => {
                out.push_str(&format!("{}-> project({:?})\n", " ".repeat(indent), cols));
                src.print(indent + 2, out);
            }
        }
    }
}

fn main() {
    let left = RelExpr::scan("a".into(), vec![0, 1]);
    let right = RelExpr::scan("b".into(), vec![2, 3]);

    let join = left.join(
        right,
        vec![
            Expr::col_ref(0).eq(Expr::col_ref(2)),
            Expr::col_ref(1).eq(Expr::int(100)),
            Expr::col_ref(2).eq(Expr::int(200)),
        ],
    );

    let mut out = String::new();
    join.print(0, &mut out);
    println!("{}", out);
}
