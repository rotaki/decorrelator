// Query Decorrelator
// References:
// * https://buttondown.email/jaffray/archive/the-little-planner-chapter-4-a-pushdown-party/
// * https://buttondown.email/jaffray/archive/a-very-basic-decorrelator/

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    col_id_generator::ColIdGeneratorRef,
    field::Field,
    rules::{Rule, RulesRef},
};

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
        expr: Box<Expr>,
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

    // Rewrite the free variables in the expression tree
    fn rewrite(self, src_to_dest: &HashMap<usize, usize>) -> Expr {
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
                left: Box::new(left.rewrite(src_to_dest)),
                right: Box::new(right.rewrite(src_to_dest)),
            },
            Expr::Case {
                expr,
                whens,
                else_expr,
            } => Expr::Case {
                expr: Box::new(expr.rewrite(src_to_dest)),
                whens: whens
                    .into_iter()
                    .map(|(when, then)| (when.rewrite(src_to_dest), then.rewrite(src_to_dest)))
                    .collect(),
                else_expr: Box::new(else_expr.rewrite(src_to_dest)),
            },
            Expr::Subquery { expr } => Expr::Subquery {
                expr: Box::new(expr.rewrite_free_variables(src_to_dest)),
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
                expr.print_inner(indent, out);
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
                out.push_str("λ.(\n");
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
            Expr::Binary { op, left, right } => {
                let mut set = left.free();
                set.extend(right.free());
                set
            }
            Expr::Case {
                expr,
                whens,
                else_expr,
            } => {
                let mut set = expr.free();
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
    pub fn scan(table_name: String, column_names: Vec<usize>) -> RelExpr {
        RelExpr::Scan {
            table_name,
            column_names,
        }
    }

    pub fn select(
        self,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        predicates: Vec<Expr>,
    ) -> RelExpr {
        if predicates.is_empty() {
            return self;
        }
        let mut predicates = predicates
            .into_iter()
            .flat_map(|expr| expr.split_conjunction())
            .collect();

        if enabled_rules.enabled(&Rule::SelectionPushdown) {
            match self {
                RelExpr::Select {
                    src,
                    predicates: mut preds,
                } => {
                    preds.append(&mut predicates);
                    src.select(enabled_rules, col_id_gen, preds)
                }
                RelExpr::Join {
                    join_type,
                    left,
                    right,
                    predicates: mut preds,
                } => {
                    preds.append(&mut predicates);
                    left.join(enabled_rules, col_id_gen, join_type, *right, preds)
                }
                RelExpr::Aggregate {
                    src,
                    group_by,
                    aggrs,
                } => {
                    // If the predicate is bound by the group by columns, we can push it to the source
                    let group_by_cols: HashSet<_> = group_by.iter().cloned().collect();
                    let (push_down, keep): (Vec<_>, Vec<_>) = predicates
                        .into_iter()
                        .partition(|pred| pred.free().is_subset(&group_by_cols));
                    src.select(enabled_rules, col_id_gen, push_down)
                        .aggregate(group_by, aggrs)
                        .select(enabled_rules, col_id_gen, keep)
                }
                RelExpr::Map { input, exprs } => {
                    // If the predicate does not intersect with the atts of exprs, we can push it to the source
                    let atts = exprs.iter().map(|(id, _)| *id).collect::<HashSet<_>>();
                    let (push_down, keep): (Vec<_>, Vec<_>) = predicates
                        .into_iter()
                        .partition(|pred| pred.free().is_disjoint(&atts));
                    input
                        .select(enabled_rules, col_id_gen, push_down)
                        .map(enabled_rules, col_id_gen, exprs)
                        .select(enabled_rules, col_id_gen, keep)
                }
                _ => RelExpr::Select {
                    src: Box::new(self),
                    predicates,
                },
            }
        } else {
            RelExpr::Select {
                src: Box::new(self),
                predicates,
            }
        }
    }

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

    pub fn project(
        self,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        cols: HashSet<usize>,
    ) -> RelExpr {
        if self.att() == cols {
            return self;
        }

        if self.att().is_subset(&cols) {
            panic!("Can't project columns that are not in the source")
        }

        if enabled_rules.enabled(&Rule::ProjectionPushdown) {
            match self {
                RelExpr::Project {
                    src,
                    cols: existing_cols,
                } => {
                    if cols.is_subset(&existing_cols) {
                        src.project(enabled_rules, col_id_gen, cols)
                    } else {
                        panic!("Can't project columns that are not in the source")
                    }
                }
                RelExpr::Map {
                    input,
                    exprs: mut existing_exprs,
                } => {
                    // Remove the mappings that are not used in the projection.
                    existing_exprs.retain(|(id, _)| cols.contains(id));
                    RelExpr::Project {
                        src: Box::new(RelExpr::Map {
                            input,
                            exprs: existing_exprs,
                        }),
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

    pub fn aggregate(self, group_by: Vec<usize>, aggrs: Vec<(usize, (usize, AggOp))>) -> RelExpr {
        RelExpr::Aggregate {
            src: Box::new(self),
            group_by,
            aggrs,
        }
    }

    pub fn map(
        self,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        exprs: impl IntoIterator<Item = (usize, Expr)>,
    ) -> RelExpr {
        let mut exprs: Vec<(usize, Expr)> = exprs.into_iter().collect();

        if exprs.is_empty() {
            return self;
        }

        if enabled_rules.enabled(&Rule::Hoist) {
            for i in 0..exprs.len() {
                // Only hoist expressions with subqueries
                if exprs[i].1.has_subquery() {
                    let (id, expr) = exprs.swap_remove(i);
                    return self.map(enabled_rules, col_id_gen, exprs).hoist(
                        enabled_rules,
                        col_id_gen,
                        id,
                        expr,
                    );
                }
            }
        }

        match self {
            RelExpr::Map {
                input,
                exprs: mut existing_exprs,
            } => {
                // If the free variables of the new exprs (exprs) does not intersect with the attrs
                // of the existing exprs (existing_exprs), then we can push the new exprs to existing_exprs.
                let atts = existing_exprs
                    .iter()
                    .map(|(id, _)| *id)
                    .collect::<HashSet<_>>();
                let (mut push_down, keep): (Vec<_>, Vec<_>) = exprs
                    .into_iter()
                    .partition(|(_, expr)| expr.free().is_disjoint(&atts));
                existing_exprs.append(&mut push_down);
                input.map(enabled_rules, col_id_gen, existing_exprs).map(
                    enabled_rules,
                    col_id_gen,
                    keep,
                )
            }
            _ => RelExpr::Map {
                input: Box::new(self),
                exprs,
            },
        }
    }

    pub fn flatmap(
        self,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        func: RelExpr,
    ) -> RelExpr {
        if enabled_rules.enabled(&Rule::Decorrelate) {
            // Not correlated!
            if func.free().is_empty() {
                return self.join(enabled_rules, col_id_gen, JoinType::CrossJoin, func, vec![]);
            }

            // Pull up Project
            if let RelExpr::Project { src, mut cols } = func {
                cols.extend(self.att());
                return self.flatmap(enabled_rules, col_id_gen, *src).project(
                    enabled_rules,
                    col_id_gen,
                    cols,
                );
            }

            // Pull up Maps
            if let RelExpr::Map { input, exprs } = func {
                return self.flatmap(enabled_rules, col_id_gen, *input).map(
                    enabled_rules,
                    col_id_gen,
                    exprs,
                );
            }

            // Pull up Selects
            if let RelExpr::Select { src, predicates } = func {
                return self.flatmap(enabled_rules, col_id_gen, *src).select(
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
                        .flatmap(enabled_rules, col_id_gen, *src)
                        .aggregate(group_by.into_iter().collect(), aggrs);
                } else {
                    // Deal with the COUNT BUG
                    let orig = self.clone();
                    let (mut plan, new_col_ids) = self.rename(&enabled_rules, &col_id_gen);
                    let new_att = plan.att();
                    let src = src.rewrite_free_variables(&new_col_ids);
                    plan = plan.flatmap(enabled_rules, col_id_gen, src);
                    plan = plan.aggregate(
                        group_by
                            .into_iter()
                            .chain(new_att.iter().cloned())
                            .collect(),
                        aggrs,
                    );
                    plan = orig.join(
                        enabled_rules,
                        col_id_gen,
                        JoinType::LeftOuter,
                        plan,
                        new_col_ids
                            .iter()
                            .map(|(src, dest)| Expr::col_ref(*src).eq(Expr::col_ref(*dest)))
                            .collect(),
                    );
                    let att = plan.att();
                    let project_att = att.difference(&new_att).cloned().collect();
                    return plan.project(enabled_rules, col_id_gen, project_att).map(
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

    // Make subquery into a FlatMap
    // FlatMap is sometimes called "Apply", "Dependent Join", or "Lateral Join"
    //
    // SQL Query:
    // Table x: a, b
    // Table y: c
    //
    // SELECT x.a, x.b, 4 + (SELECT x.a + y.c FROM y) FROM x
    //
    // Before:
    // ---------------------------------------
    // |  Map to @4                          |
    // |            ------------------------ |
    // |            |  Subquery: @3 + @1   | |
    // |    4 +     |  Scan @3             | |
    // |            ------------------------ |
    // ---------------------------------------
    //                 |
    // ---------------------------------------
    // |  Scan  @1, @2                       |
    // ---------------------------------------

    // After:
    // -------------------------------------------
    // |  Project @1, @2, @4                     |
    // -------------------------------------------
    //                  |
    // -------------------------------------------
    // |  Map to @4                              |
    // |     @lhs_id + @rhs_id                   |
    // -------------------------------------------
    //                  |
    // -------------------------------------------
    // |  FlatMap (@rhs_id <- @3 + @1)           |
    // -------------------------------------------
    //              /                   \
    // ---------------------------     -----------
    // |  Join (@lhs_id <- 4)    |     | @3 + @1 |
    // ---------------------------     -----------
    //          /         \
    // ----------------
    // |  Scan @1, @2 |     4
    // ----------------

    fn hoist(
        self,
        enabled_rules: &RulesRef,
        col_id_gen: &ColIdGeneratorRef,
        id: usize,
        expr: Expr,
    ) -> RelExpr {
        match expr {
            Expr::Subquery { expr } => {
                let att = expr.att();
                assert!(att.len() == 1);
                let input_col_id = att.iter().next().unwrap();
                // Give the column the name that's expected
                let rhs: RelExpr = expr.map(
                    enabled_rules,
                    col_id_gen,
                    vec![(id, Expr::col_ref(*input_col_id))],
                );
                self.flatmap(enabled_rules, col_id_gen, rhs)
            }
            Expr::Binary { op, left, right } => {
                // Hoist the left, hoist the right, then perform the binary operation
                let lhs_id = col_id_gen.next();
                let rhs_id = col_id_gen.next();
                let att = self.att();
                self.hoist(enabled_rules, col_id_gen, lhs_id, *left)
                    .hoist(enabled_rules, col_id_gen, rhs_id, *right)
                    .map(
                        enabled_rules,
                        col_id_gen,
                        [(
                            id,
                            Expr::Binary {
                                op,
                                left: Box::new(Expr::col_ref(lhs_id)),
                                right: Box::new(Expr::col_ref(rhs_id)),
                            },
                        )],
                    )
                    .project(
                        enabled_rules,
                        col_id_gen,
                        att.into_iter().chain([id].into_iter()).collect(),
                    )
            }
            Expr::Field { .. } | Expr::ColRef { .. } => {
                self.map(enabled_rules, col_id_gen, vec![(id, expr)])
            }
            Expr::Case { .. } => {
                panic!("Case expression is not supported in hoist")
            }
        }
    }

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

    fn rename_to(self, src_to_dest: HashMap<usize, usize>) -> RelExpr {
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
                // We assume that the cols are in the src attribute set.
                // Otherwise we can't project the columns.
                // Therefore, we don't need to check the free set of the cols.
                src.free()
            }
            RelExpr::OrderBy { src, .. } => src.free(),
            RelExpr::Aggregate { src, .. } => src.free(),
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
            RelExpr::Rename {
                src,
                src_to_dest: colsk,
            } => src.free(),
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
                src,
                src_to_dest: colsk,
                ..
            } => {
                let mut set = src.att();
                // rewrite the column names
                for (src, dest) in colsk {
                    set.remove(src);
                    set.insert(*dest);
                }
                set
            }
        }
    }

    // Rename the free variable to the given mapping
    fn rewrite_free_variables(self, src_to_dest: &HashMap<usize, usize>) -> RelExpr {
        // If there is no intersection between the free variables and the mapping, we can terminate early
        if self
            .free()
            .is_disjoint(&src_to_dest.keys().cloned().collect())
        {
            return self;
        }

        match self {
            RelExpr::Scan { .. } => self,
            RelExpr::Select { src, predicates } => RelExpr::Select {
                src: Box::new(src.rewrite_free_variables(src_to_dest)),
                predicates: predicates
                    .into_iter()
                    .map(|pred| pred.rewrite(src_to_dest))
                    .collect(),
            },
            RelExpr::Join {
                join_type,
                left,
                right,
                predicates,
            } => RelExpr::Join {
                join_type,
                left: Box::new(left.rewrite_free_variables(src_to_dest)),
                right: Box::new(right.rewrite_free_variables(src_to_dest)),
                predicates: predicates
                    .into_iter()
                    .map(|pred| pred.rewrite(src_to_dest))
                    .collect(),
            },
            RelExpr::Project { src, cols } => RelExpr::Project {
                src: Box::new(src.rewrite_free_variables(src_to_dest)),
                cols: cols
                    .into_iter()
                    .map(|col| *src_to_dest.get(&col).unwrap_or(&col))
                    .collect(),
            },
            RelExpr::OrderBy { src, cols } => RelExpr::OrderBy {
                src: Box::new(src.rewrite_free_variables(src_to_dest)),
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
                src: Box::new(src.rewrite_free_variables(src_to_dest)),
                group_by: group_by
                    .into_iter()
                    .map(|id| *src_to_dest.get(&id).unwrap_or(&id))
                    .collect(),
                aggrs: aggrs
                    .into_iter()
                    .map(|(id, (src_id, op))| {
                        (id, (*src_to_dest.get(&src_id).unwrap_or(&src_id), op))
                    })
                    .collect(),
            },
            RelExpr::Map { input, exprs } => RelExpr::Map {
                input: Box::new(input.rewrite_free_variables(src_to_dest)),
                exprs: exprs
                    .into_iter()
                    .map(|(id, expr)| (id, expr.rewrite(src_to_dest)))
                    .collect(),
            },
            RelExpr::FlatMap { input, func } => RelExpr::FlatMap {
                input: Box::new(input.rewrite_free_variables(src_to_dest)),
                func: Box::new(func.rewrite_free_variables(src_to_dest)),
            },
            RelExpr::Rename {
                src,
                src_to_dest: colsk,
            } => RelExpr::Rename {
                src: Box::new(src.rewrite_free_variables(src_to_dest)),
                src_to_dest: colsk
                    .into_iter()
                    .map(|(src, dest)| (*src_to_dest.get(&src).unwrap_or(&src), dest))
                    .collect(),
            },
        }
    }
}

impl RelExpr {
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
                out.push_str(&format!("{}-> aggregate(\n", " ".repeat(indent)));
                out.push_str(&format!("{}    group_by: [", " ".repeat(indent),));
                let mut split = "";
                for col in group_by {
                    out.push_str(split);
                    out.push_str(&format!("@{}", col));
                    split = ", ";
                }
                out.push_str("],\n");
                for (id, (input_id, op)) in aggrs {
                    out.push_str(&format!(
                        "{}    @{} <- {}(@{})\n",
                        " ".repeat(indent),
                        id,
                        op,
                        input_id
                    ));
                }
                out.push_str(&format!("{})\n", " ".repeat(indent + 2)));
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

#[cfg(test)]
mod tests {
    use crate::{col_id_generator::ColIdGenerator, rules::Rules};

    #[test]
    fn test_new_col_id() {
        use super::*;
        let col_id_gen = Rc::new(ColIdGenerator::new());
        let enabled_rules = Rc::new(Rules::new());
        let expr = RelExpr::Scan {
            table_name: "t".to_string(),
            column_names: vec![1, 2],
        };
        let (expr, new_ids) = expr.rename(&enabled_rules, &col_id_gen);
        expr.pretty_print();
    }

    #[test]
    fn test_new_col_ids() {
        use super::*;
        let col_id_gen = Rc::new(ColIdGenerator::new());
        let enabled_rules = Rc::new(Rules::new());
        let expr = RelExpr::Scan {
            table_name: "t".to_string(),
            column_names: vec![1, 2],
        };
        let (expr, new_ids) = expr.rename(&enabled_rules, &col_id_gen);
        let (expr, new_ids) = expr.rename(&enabled_rules, &col_id_gen);
        expr.pretty_print();
    }
}
