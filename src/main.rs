// Query Decorrelator
// References:
// * https://buttondown.email/jaffray/archive/the-little-planner-chapter-4-a-pushdown-party/
// * https://buttondown.email/jaffray/archive/a-very-basic-decorrelator/

use std::{cell::RefCell, collections::HashSet, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Rule {
    Hoist,
    Decorrelate,
}

struct State {
    next_id: Rc<RefCell<usize>>,
    enabled_rules: Rc<RefCell<HashSet<Rule>>>,
}

impl State {
    fn new() -> Self {
        State {
            next_id: Rc::new(RefCell::new(0)),
            enabled_rules: Rc::new(RefCell::new(HashSet::new())),
        }
    }

    fn next(&self) -> usize {
        let id = *self.next_id.borrow();
        *self.next_id.borrow_mut() += 1;
        id
    }

    fn enable(&self, rule: Rule) {
        self.enabled_rules.borrow_mut().insert(rule);
    }

    fn enabled(&self, rule: Rule) -> bool {
        self.enabled_rules.borrow().contains(&rule)
    }
}

#[derive(Debug)]
enum Expr {
    ColRef { id: usize },
    Int { val: i64 },
    Eq { left: Box<Expr>, right: Box<Expr> },
    Add { left: Box<Expr>, right: Box<Expr> },
    Subquery { expr: Box<RelExpr> },
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

    fn add(self, other: Expr) -> Expr {
        Expr::Add {
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    fn has_subquery(&self) -> bool {
        match self {
            Expr::ColRef { id: _ } => false,
            Expr::Int { val: _ } => false,
            Expr::Eq { left, right } => left.has_subquery() || right.has_subquery(),
            Expr::Add { left, right } => left.has_subquery() || right.has_subquery(),
            Expr::Subquery { expr: _ } => true,
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
            Expr::Add { left, right } => {
                left.print(indent, out);
                out.push_str("+");
                right.print(indent, out);
            }
            Expr::Subquery { expr } => {
                out.push_str("λ.(\n");
                expr.print(indent + 6, out);
                out.push_str(&format!("{})", " ".repeat(indent + 4)));
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
            Expr::Add { left, right } => {
                let mut set = left.free();
                set.extend(right.free());
                set
            }
            Expr::Subquery { expr } => expr.free(),
        }
    }

    fn bound_by(&self, rel: &RelExpr) -> bool {
        self.free().is_subset(&rel.att())
    }
}

#[derive(Debug)]
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
        // Reduces the number of columns in the result
        src: Box<RelExpr>,
        cols: HashSet<usize>,
    },
    Map {
        // Appends new columns to the result
        input: Box<RelExpr>,
        exprs: Vec<(usize, Expr)>,
    },
    FlatMap {
        // For each row in the input, call func and append the result to the output
        input: Box<RelExpr>,
        func: Box<RelExpr>,
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

    fn project(self, _state: &State, cols: HashSet<usize>) -> RelExpr {
        RelExpr::Project {
            src: Box::new(self),
            cols,
        }
    }

    fn map(self, state: &State, exprs: impl IntoIterator<Item = (usize, Expr)>) -> RelExpr {
        let mut exprs: Vec<(usize, Expr)> = exprs.into_iter().collect();

        if exprs.is_empty() {
            return self;
        }

        if state.enabled(Rule::Hoist) {
            for i in 0..exprs.len() {
                // Only hoist expressions with subqueries
                if exprs[i].1.has_subquery() {
                    let (id, expr) = exprs.swap_remove(i);
                    return self.map(state, exprs).hoist(state, id, expr);
                }
            }
        }

        RelExpr::Map {
            input: Box::new(self),
            exprs,
        }
    }

    fn flatmap(self, state: &State, func: RelExpr) -> RelExpr {
        if state.enabled(Rule::Decorrelate) {
            // Not correlated!
            if func.free().is_empty() {
                return self.join(func, vec![]);
            }

            // Pull up Project
            if let RelExpr::Project { src, mut cols } = func {
                cols.extend(self.att());
                return self.flatmap(state, *src).project(state, cols);
            }

            // Pull up Maps
            if let RelExpr::Map { input, exprs } = func {
                return self.flatmap(state, *input).map(state, exprs);
            }
        }
        RelExpr::FlatMap {
            input: Box::new(self),
            func: Box::new(func),
        }
    }

    // Make subquery into a FlatMap
    // FlatMap is sometimes called "Apply", "Dependent Join", or "Lateral Join"
    fn hoist(self, state: &State, id: usize, expr: Expr) -> RelExpr {
        match expr {
            Expr::Subquery { expr } => {
                let att = expr.att();
                assert!(att.len() == 1);
                let input_col_id = att.iter().next().unwrap();
                // Give the column the name that's expected
                let rhs = expr.map(state, vec![(id, Expr::col_ref(*input_col_id))]);
                self.flatmap(state, rhs)
            }
            Expr::Add { left, right } => {
                // Hoist the left, hoist the right, then perform the addition
                let lhs_id = state.next();
                let rhs_id = state.next();
                let att = self.att();
                self.hoist(state, lhs_id, *left)
                    .hoist(state, rhs_id, *right)
                    .map(
                        state,
                        [(
                            id,
                            Expr::Add {
                                left: Box::new(Expr::col_ref(lhs_id)),
                                right: Box::new(Expr::col_ref(rhs_id)),
                            },
                        )],
                    )
                    .project(state, att.into_iter().chain([id].into_iter()).collect())
            }
            Expr::Int { .. } | Expr::ColRef { .. } => self.map(state, vec![(id, expr)]),
            x => unimplemented!("{:?}", x),
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
            RelExpr::Map { input, exprs } => {
                let mut set = input.free();
                for (_, expr) in exprs {
                    set.extend(expr.free());
                }
                set.difference(&input.att()).cloned().collect()
            }
            RelExpr::FlatMap { input, func } => {
                let mut set = input.free();
                set.extend(func.free());
                set.difference(&input.att()).cloned().collect()
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
            RelExpr::Map { input, exprs } => {
                let mut set = input.att();
                set.extend(exprs.iter().map(|(id, _)| *id));
                set
            }
            RelExpr::FlatMap { input, func } => {
                let mut set = input.att();
                set.extend(func.att());
                set
            }
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
            RelExpr::Map { input, exprs } => {
                out.push_str(&format!("{}-> map(\n", " ".repeat(indent)));
                for (id, expr) in exprs {
                    out.push_str(&format!("{}    @{} <- ", " ".repeat(indent), id));
                    expr.print(indent, out);
                    out.push_str(",\n");
                }
                out.push_str(&format!("{})\n", " ".repeat(indent + 2)));
                input.print(indent + 2, out);
            }
            RelExpr::FlatMap { input, func } => {
                out.push_str(&format!("{}-> flatmap\n", " ".repeat(indent)));
                input.print(indent + 2, out);
                out.push_str(&format!("{}  λ.{:?}\n", " ".repeat(indent), func.free()));
                func.print(indent + 2, out);
            }
        }
    }
}
fn main() {
    let state = State::new();
    state.enable(Rule::Hoist);
    state.enable(Rule::Decorrelate);

    let a = state.next();
    let b = state.next();
    let x = state.next();
    let y = state.next();

    let sum_col = state.next();

    let v = RelExpr::scan("a".into(), vec![a, b]).map(
        &state,
        vec![
            // (
            //     state.next(),
            //     Expr::int(3).plus(Expr::Subquery {
            //         expr: Box::new(
            //             RelExpr::scan("x".into(), vec![x, y]).project([x].into_iter().collect()),
            //         ),
            //     }),
            // ),
            (
                state.next(),
                Expr::int(4).add(Expr::Subquery {
                    expr: Box::new(
                        RelExpr::scan("x".into(), vec![x, y])
                            .project(&state, [x].into_iter().collect())
                            .map(&state, [(sum_col, Expr::col_ref(x).add(Expr::col_ref(a)))])
                            .project(&state, [sum_col].into_iter().collect()),
                    ),
                }),
            ),
        ],
    );

    // let v = RelExpr::scan("a".into(), vec![a, b]).map(
    //     &state,
    //     vec![(
    //         state.next(),
    //         Expr::Subquery {
    //             expr: Box::new(
    //                 RelExpr::scan("x".into(), vec![x, y])
    //                     .project(&state, [x].into_iter().collect()),
    //             ),
    //         },
    //     )],
    // );

    let mut out = String::new();
    v.print(0, &mut out);

    println!("{}", out);
}
