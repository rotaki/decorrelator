// Query Decorrelator
// References:
// * https://buttondown.email/jaffray/archive/the-little-planner-chapter-4-a-pushdown-party/
// * https://buttondown.email/jaffray/archive/a-very-basic-decorrelator/

use crate::field::Field;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

impl std::fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Eq => write!(f, "="),
            BinaryOp::Neq => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::Le => write!(f, "<="),
            BinaryOp::Ge => write!(f, ">="),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AggOp {
    Sum,
    Count,
    Avg,
    Min,
    Max,
}

impl std::fmt::Display for AggOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggOp::Sum => write!(f, "sum"),
            AggOp::Count => write!(f, "count"),
            AggOp::Avg => write!(f, "avg"),
            AggOp::Min => write!(f, "min"),
            AggOp::Max => write!(f, "max"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    ColRef {
        id: usize,
    },
    Field {
        val: Field,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Case {
        expr: Option<Box<Expr>>,
        whens: Vec<(Expr, Expr)>,
        else_expr: Box<Expr>,
    },
    Subquery {
        expr: Box<RelExpr>,
    },
}

impl Expr {
    pub fn col_ref(id: usize) -> Expr {
        Expr::ColRef { id }
    }

    pub fn int(val: i64) -> Expr {
        Expr::Field {
            val: Field::Int(val),
        }
    }

    pub fn bool(val: bool) -> Expr {
        Expr::Field {
            val: Field::Bool(val),
        }
    }

    pub fn null() -> Expr {
        Expr::Field { val: Field::Null }
    }

    pub fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn eq(self, other: Expr) -> Expr {
        Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    pub fn gt(self, other: Expr) -> Expr {
        Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    pub fn add(self, other: Expr) -> Expr {
        Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    pub fn subquery(expr: RelExpr) -> Expr {
        Expr::Subquery {
            expr: Box::new(expr),
        }
    }

    pub fn has_subquery(&self) -> bool {
        match self {
            Expr::ColRef { id: _ } => false,
            Expr::Field { val: _ } => false,
            Expr::Binary { left, right, .. } => left.has_subquery() || right.has_subquery(),
            Expr::Case { .. } => {
                // Currently, we don't support subqueries in the case expression
                false
            }
            Expr::Subquery { expr: _ } => true,
        }
    }

    pub fn split_conjunction(self) -> Vec<Expr> {
        match self {
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                let mut left = left.split_conjunction();
                let mut right = right.split_conjunction();
                left.append(&mut right);
                left
            }
            _ => vec![self],
        }
    }

    // Rewrite the variables in the expression tree
    fn replace_variables(self, src_to_dest: &HashMap<usize, usize>) -> Expr {
        match self {
            Expr::ColRef { id } => {
                if let Some(dest) = src_to_dest.get(&id) {
                    Expr::ColRef { id: *dest }
                } else {
                    Expr::ColRef { id }
                }
            }
            Expr::Field { val } => Expr::Field { val },
            Expr::Binary { op, left, right } => Expr::Binary {
                op,
                left: Box::new(left.replace_variables(src_to_dest)),
                right: Box::new(right.replace_variables(src_to_dest)),
            },
            Expr::Case {
                expr,
                whens,
                else_expr,
            } => Expr::Case {
                expr: expr.map(|expr| Box::new(expr.replace_variables(src_to_dest))),
                whens: whens
                    .into_iter()
                    .map(|(when, then)| {
                        (
                            when.replace_variables(src_to_dest),
                            then.replace_variables(src_to_dest),
                        )
                    })
                    .collect(),
                else_expr: Box::new(else_expr.replace_variables(src_to_dest)),
            },
            Expr::Subquery { expr } => Expr::Subquery {
                expr: Box::new(expr.replace_variables(src_to_dest)),
            },
        }
    }

    pub(crate) fn replace_variables_with_exprs(self, src_to_dest: &HashMap<usize, Expr>) -> Expr {
        match self {
            Expr::ColRef { id } => {
                if let Some(expr) = src_to_dest.get(&id) {
                    expr.clone()
                } else {
                    Expr::ColRef { id }
                }
            }
            Expr::Field { val } => Expr::Field { val },
            Expr::Binary { op, left, right } => Expr::Binary {
                op,
                left: Box::new(left.replace_variables_with_exprs(src_to_dest)),
                right: Box::new(right.replace_variables_with_exprs(src_to_dest)),
            },
            Expr::Case {
                expr,
                whens,
                else_expr,
            } => Expr::Case {
                expr: expr.map(|expr| Box::new(expr.replace_variables_with_exprs(src_to_dest))),
                whens: whens
                    .into_iter()
                    .map(|(when, then)| {
                        (
                            when.replace_variables_with_exprs(src_to_dest),
                            then.replace_variables_with_exprs(src_to_dest),
                        )
                    })
                    .collect(),
                else_expr: Box::new(else_expr.replace_variables_with_exprs(src_to_dest)),
            },
            Expr::Subquery { expr } => Expr::Subquery {
                // Do nothing for subquery
                expr,
            },
        }
    }

    pub fn pretty_print(&self) {
        println!("{}", self.pretty_string());
    }

    pub fn pretty_string(&self) -> String {
        let mut out = String::new();
        self.print_inner(0, &mut out);
        out
    }

    fn print_inner(&self, indent: usize, out: &mut String) {
        match self {
            Expr::ColRef { id } => {
                out.push_str(&format!("@{}", id));
            }
            Expr::Field { val } => {
                out.push_str(&format!("{}", val));
            }
            Expr::Binary { op, left, right } => {
                left.print_inner(indent, out);
                out.push_str(&format!("{}", op));
                right.print_inner(indent, out);
            }
            Expr::Case {
                expr,
                whens,
                else_expr,
            } => {
                out.push_str("case ");
                expr.as_ref().map(|expr| expr.print_inner(indent, out));
                for (when, then) in whens {
                    out.push_str(" when ");
                    when.print_inner(indent, out);
                    out.push_str(" then ");
                    then.print_inner(indent, out);
                }
                out.push_str(" else ");
                else_expr.print_inner(indent, out);
                out.push_str(" end");
            }
            Expr::Subquery { expr } => {
                out.push_str(&format!("λ.{:?}(\n", expr.free()));
                expr.print_inner(indent + 6, out);
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
    pub fn free(&self) -> HashSet<usize> {
        match self {
            Expr::ColRef { id } => {
                let mut set = HashSet::new();
                set.insert(*id);
                set
            }
            Expr::Field { val: _ } => HashSet::new(),
            Expr::Binary { left, right, .. } => {
                let mut set = left.free();
                set.extend(right.free());
                set
            }
            Expr::Case {
                expr,
                whens,
                else_expr,
            } => {
                let mut set = expr
                    .as_ref()
                    .map(|expr| expr.free())
                    .unwrap_or(HashSet::new());
                for (when, then) in whens {
                    set.extend(when.free());
                    set.extend(then.free());
                }
                set.extend(else_expr.free());
                set
            }
            Expr::Subquery { expr } => expr.free(),
        }
    }

    pub fn bound_by(&self, rel: &RelExpr) -> bool {
        self.free().is_subset(&rel.att())
    }
}

#[derive(Debug, Clone)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    FullOuter,
    CrossJoin,
}

impl std::fmt::Display for JoinType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JoinType::Inner => write!(f, "inner"),
            JoinType::LeftOuter => write!(f, "left_outer"),
            JoinType::RightOuter => write!(f, "right_outer"),
            JoinType::FullOuter => write!(f, "full_outer"),
            JoinType::CrossJoin => write!(f, "cross"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RelExpr {
    Scan {
        table_name: String,
        column_names: Vec<usize>,
    },
    Select {
        // Evaluate the predicate for each row in the source
        src: Box<RelExpr>,
        predicates: Vec<Expr>,
    },
    Join {
        join_type: JoinType,
        left: Box<RelExpr>,
        right: Box<RelExpr>,
        predicates: Vec<Expr>,
    },
    Project {
        // Reduces the number of columns in the result
        src: Box<RelExpr>,
        cols: HashSet<usize>,
    },
    OrderBy {
        src: Box<RelExpr>,
        cols: Vec<(usize, bool, bool)>, // (column_id, asc, nulls_first)
    },
    Aggregate {
        src: Box<RelExpr>,
        group_by: Vec<usize>,
        aggrs: Vec<(usize, (usize, AggOp))>, // (dest_column_id, (src_column_id, agg_op)
    },
    Map {
        // Appends new columns to the result
        // This is the only operator that can have a reference to the columns of
        // the outer scope
        input: Box<RelExpr>,
        exprs: Vec<(usize, Expr)>,
    },
    FlatMap {
        // For each row in the input, call func and append the result to the output
        input: Box<RelExpr>,
        func: Box<RelExpr>,
    },
    Rename {
        src: Box<RelExpr>,
        src_to_dest: HashMap<usize, usize>, // (src_column_id, dest_column_id)
    },
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
    pub fn free(&self) -> HashSet<usize> {
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
                ..
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
                let mut set = src.free();
                for col in cols {
                    set.insert(*col);
                }
                set.difference(&src.att()).cloned().collect()
            }
            RelExpr::OrderBy { src, cols } => {
                let mut set = src.free();
                for (id, _, _) in cols {
                    set.insert(*id);
                }
                set.difference(&src.att()).cloned().collect()
            }
            RelExpr::Aggregate {
                src,
                group_by,
                aggrs,
                ..
            } => {
                let mut set = src.free();
                for id in group_by {
                    set.insert(*id);
                }
                for (_, (src_id, _)) in aggrs {
                    set.insert(*src_id);
                }
                set.difference(&src.att()).cloned().collect()
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
            RelExpr::Rename { src, .. } => src.free(),
        }
    }

    // Attribute set of relational expression
    // * The set of columns that are in the result of the expression.
    // * Attribute changes when we do a projection or map the columns to a different name.
    //
    // Difference between "free" and "att"
    // * "free" is the set of columns that we need to evaluate the expression
    // * "att" is the set of columns that we have (the column names of the result of RelExpr)
    pub fn att(&self) -> HashSet<usize> {
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
            RelExpr::Project { cols, .. } => cols.iter().cloned().collect(),
            RelExpr::OrderBy { src, .. } => src.att(),
            RelExpr::Aggregate {
                group_by, aggrs, ..
            } => {
                let mut set: HashSet<usize> = group_by.iter().cloned().collect();
                set.extend(aggrs.iter().map(|(id, _)| *id));
                set
            }
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
            RelExpr::Rename {
                src, src_to_dest, ..
            } => {
                let mut set = src.att();
                // rewrite the column names
                for (src, dest) in src_to_dest {
                    set.remove(src);
                    set.insert(*dest);
                }
                set
            }
        }
    }

    // Replace the column names in the relational expression
    pub(crate) fn replace_variables(self, src_to_dest: &HashMap<usize, usize>) -> RelExpr {
        match self {
            RelExpr::Scan {
                table_name,
                column_names,
            } => {
                let column_names = column_names
                    .into_iter()
                    .map(|col| *src_to_dest.get(&col).unwrap_or(&col))
                    .collect();
                RelExpr::Scan {
                    table_name,
                    column_names,
                }
            }
            RelExpr::Select { src, predicates } => RelExpr::Select {
                src: Box::new(src.replace_variables(src_to_dest)),
                predicates: predicates
                    .into_iter()
                    .map(|pred| pred.replace_variables(src_to_dest))
                    .collect(),
            },
            RelExpr::Join {
                join_type,
                left,
                right,
                predicates,
            } => RelExpr::Join {
                join_type,
                left: Box::new(left.replace_variables(src_to_dest)),
                right: Box::new(right.replace_variables(src_to_dest)),
                predicates: predicates
                    .into_iter()
                    .map(|pred| pred.replace_variables(src_to_dest))
                    .collect(),
            },
            RelExpr::Project { src, cols } => RelExpr::Project {
                src: Box::new(src.replace_variables(src_to_dest)),
                cols: cols
                    .into_iter()
                    .map(|col| *src_to_dest.get(&col).unwrap_or(&col))
                    .collect(),
            },
            RelExpr::OrderBy { src, cols } => RelExpr::OrderBy {
                src: Box::new(src.replace_variables(src_to_dest)),
                cols: cols
                    .into_iter()
                    .map(|(id, asc, nulls_first)| {
                        (*src_to_dest.get(&id).unwrap_or(&id), asc, nulls_first)
                    })
                    .collect(),
            },
            RelExpr::Aggregate {
                src,
                group_by,
                aggrs,
            } => RelExpr::Aggregate {
                src: Box::new(src.replace_variables(src_to_dest)),
                group_by: group_by
                    .into_iter()
                    .map(|id| *src_to_dest.get(&id).unwrap_or(&id))
                    .collect(),
                aggrs: aggrs
                    .into_iter()
                    .map(|(id, (src_id, op))| {
                        (
                            *src_to_dest.get(&id).unwrap_or(&id),
                            (*src_to_dest.get(&src_id).unwrap_or(&src_id), op),
                        )
                    })
                    .collect(),
            },
            RelExpr::Map { input, exprs } => RelExpr::Map {
                input: Box::new(input.replace_variables(src_to_dest)),
                exprs: exprs
                    .into_iter()
                    .map(|(id, expr)| {
                        (
                            *src_to_dest.get(&id).unwrap_or(&id),
                            expr.replace_variables(src_to_dest),
                        )
                    })
                    .collect(),
            },
            RelExpr::FlatMap { input, func } => RelExpr::FlatMap {
                input: Box::new(input.replace_variables(src_to_dest)),
                func: Box::new(func.replace_variables(src_to_dest)),
            },
            RelExpr::Rename {
                src,
                src_to_dest: column_mappings,
            } => RelExpr::Rename {
                src: Box::new(src.replace_variables(src_to_dest)),
                src_to_dest: column_mappings
                    .into_iter()
                    .map(|(src, dest)| {
                        (
                            *src_to_dest.get(&src).unwrap_or(&src),
                            *src_to_dest.get(&dest).unwrap_or(&dest),
                        )
                    })
                    .collect(),
            },
        }
    }

    pub fn pretty_print(&self) {
        println!("{}", self.pretty_string());
    }

    pub fn pretty_string(&self) -> String {
        let mut out = String::new();
        self.print_inner(0, &mut out);
        out
    }

    fn print_inner(&self, indent: usize, out: &mut String) {
        match self {
            RelExpr::Scan {
                table_name,
                column_names,
            } => {
                out.push_str(&format!("{}-> scan({:?}, ", " ".repeat(indent), table_name,));
                let mut split = "";
                out.push_str("[");
                for col in column_names {
                    out.push_str(split);
                    out.push_str(&format!("@{}", col));
                    split = ", ";
                }
                out.push_str("])\n");
            }
            RelExpr::Select { src, predicates } => {
                out.push_str(&format!("{}-> select(", " ".repeat(indent)));
                let mut split = "";
                for pred in predicates {
                    out.push_str(split);
                    pred.print_inner(0, out);
                    split = " && ";
                }
                out.push_str(")\n");
                src.print_inner(indent + 2, out);
            }
            RelExpr::Join {
                join_type,
                left,
                right,
                predicates,
            } => {
                out.push_str(&format!("{}-> {}_join(", " ".repeat(indent), join_type));
                let mut split = "";
                for pred in predicates {
                    out.push_str(split);
                    pred.print_inner(0, out);
                    split = " && ";
                }
                out.push_str(")\n");
                left.print_inner(indent + 2, out);
                right.print_inner(indent + 2, out);
            }
            RelExpr::Project { src, cols } => {
                out.push_str(&format!("{}-> project(", " ".repeat(indent)));
                let mut split = "";
                for col in cols {
                    out.push_str(split);
                    out.push_str(&format!("@{}", col));
                    split = ", ";
                }
                out.push_str(")\n");
                src.print_inner(indent + 2, out);
            }
            RelExpr::OrderBy { src, cols } => {
                out.push_str(&format!("{}-> order_by({:?})\n", " ".repeat(indent), cols));
                src.print_inner(indent + 2, out);
            }
            RelExpr::Aggregate {
                src,
                group_by,
                aggrs,
            } => {
                out.push_str(&format!("{}-> aggregate(", " ".repeat(indent)));
                out.push_str(&format!("group_by: [",));
                let mut split = "";
                for col in group_by {
                    out.push_str(split);
                    out.push_str(&format!("@{}", col));
                    split = ", ";
                }
                out.push_str("], ");
                out.push_str(&format!("aggrs: ["));
                let mut split = "";
                for (id, (input_id, op)) in aggrs {
                    out.push_str(split);
                    out.push_str(&format!("@{} <- {:?}(@{})", id, op, input_id));
                    split = ", ";
                }
                out.push_str("]");
                out.push_str(&format!(")\n"));
                src.print_inner(indent + 2, out);
            }
            RelExpr::Map { input, exprs } => {
                out.push_str(&format!("{}-> map(\n", " ".repeat(indent)));
                for (id, expr) in exprs {
                    out.push_str(&format!("{}    @{} <- ", " ".repeat(indent), id));
                    expr.print_inner(indent, out);
                    out.push_str(",\n");
                }
                out.push_str(&format!("{})\n", " ".repeat(indent + 2)));
                input.print_inner(indent + 2, out);
            }
            RelExpr::FlatMap { input, func } => {
                out.push_str(&format!("{}-> flatmap\n", " ".repeat(indent)));
                input.print_inner(indent + 2, out);
                out.push_str(&format!("{}  λ.{:?}\n", " ".repeat(indent), func.free()));
                func.print_inner(indent + 2, out);
            }
            RelExpr::Rename {
                src,
                src_to_dest: colsk,
            } => {
                // Rename will be printed as @dest <- @src
                out.push_str(&format!("{}-> rename(", " ".repeat(indent)));
                let mut split = "";
                for (src, dest) in colsk {
                    out.push_str(split);
                    out.push_str(&format!("@{} <- @{}", dest, src));
                    split = ", ";
                }
                out.push_str(")\n");
                src.print_inner(indent + 2, out);
            }
        }
    }
}
