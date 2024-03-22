// Query Decorrelator
// References:
// * https://buttondown.email/jaffray/archive/the-little-planner-chapter-4-a-pushdown-party/
// * https://buttondown.email/jaffray/archive/a-very-basic-decorrelator/

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rule {
    Hoist,
    Decorrelate,
}

pub struct OptimizerCtx {
    next_id: Rc<RefCell<usize>>,
    enabled_rules: Rc<RefCell<HashSet<Rule>>>,
    id_to_col_name: Rc<RefCell<HashMap<usize, String>>>,
}

impl OptimizerCtx {
    pub fn new(id_to_col_name: Rc<RefCell<HashMap<usize, String>>>) -> Self {
        let start_id = 1000;
        OptimizerCtx {
            next_id: Rc::new(RefCell::new(start_id)),
            enabled_rules: Rc::new(RefCell::new(HashSet::new())),
            id_to_col_name,
        }
    }

    pub fn next(&self) -> usize {
        let id = *self.next_id.borrow();
        *self.next_id.borrow_mut() += 1;
        id
    }

    pub fn enable(&self, rule: Rule) {
        self.enabled_rules.borrow_mut().insert(rule);
    }

    pub fn enabled(&self, rule: Rule) -> bool {
        self.enabled_rules.borrow().contains(&rule)
    }

    pub fn register_column(&self, id: usize, name: String) {
        self.id_to_col_name.borrow_mut().insert(id, name);
    }

    pub fn column_name(&self, id: usize) -> Option<String> {
        self.id_to_col_name.borrow().get(&id).cloned()
    }

    pub fn get_column_id(&self, name: &str) -> Option<usize> {
        for (id, col_name) in self.id_to_col_name.borrow().iter() {
            if col_name == name {
                return Some(*id);
            }
        }
        None
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
pub enum Expr {
    ColRef {
        id: usize,
    },
    Int {
        val: i64,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
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
        Expr::Int { val }
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
            Expr::Int { val: _ } => false,
            Expr::Binary { left, right, .. } => left.has_subquery() || right.has_subquery(),
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
            Expr::Int { val } => {
                out.push_str(&format!("{}", val));
            }
            Expr::Binary { op, left, right } => {
                left.print_inner(indent, out);
                out.push_str(&format!("{}", op));
                right.print_inner(indent, out);
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
            Expr::Int { val: _ } => HashSet::new(),
            Expr::Binary { op, left, right } => {
                let mut set = left.free();
                set.extend(right.free());
                set
            }
            Expr::Subquery { expr } => expr.free(),
        }
    }

    pub fn bound_by(&self, rel: &RelExpr) -> bool {
        self.free().is_subset(&rel.att())
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
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
}

impl RelExpr {
    pub fn scan(table_name: String, column_names: Vec<usize>) -> RelExpr {
        RelExpr::Scan {
            table_name,
            column_names,
        }
    }

    pub fn select(self, predicates: Vec<Expr>) -> RelExpr {
        if predicates.is_empty() {
            return self;
        }
        let mut predicates = predicates
            .into_iter()
            .flat_map(|expr| expr.split_conjunction())
            .collect();
        match self {
            RelExpr::Select {
                src,
                predicates: mut preds,
            } => {
                preds.append(&mut predicates);
                src.select(preds)
            }
            RelExpr::Join {
                join_type,
                left,
                right,
                predicates: mut preds,
            } => {
                preds.append(&mut predicates);
                left.join(join_type, *right, preds)
            }
            _ => RelExpr::Select {
                src: Box::new(self),
                predicates,
            },
        }
    }

    pub fn join(self, join_type: JoinType, other: RelExpr, mut predicates: Vec<Expr>) -> RelExpr {
        predicates = predicates
            .into_iter()
            .flat_map(|expr| expr.split_conjunction())
            .collect();
        for i in 0..predicates.len() {
            if matches!(
                join_type,
                JoinType::Inner | JoinType::LeftOuter | JoinType::CrossJoin
            ) {
                if predicates[i].bound_by(&self) {
                    // We can push this predicate to the left side
                    let predicate = predicates.swap_remove(i);
                    return self
                        .select(vec![predicate])
                        .join(join_type, other, predicates);
                }
            }

            if matches!(
                join_type,
                JoinType::Inner | JoinType::RightOuter | JoinType::CrossJoin
            ) {
                if predicates[i].bound_by(&other) {
                    // We can push this predicate to the right side
                    let predicate = predicates.swap_remove(i);
                    return self.join(join_type, other.select(vec![predicate]), predicates);
                }
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
                return self.join(JoinType::Inner, other, predicates);
            }
        }

        RelExpr::Join {
            join_type,
            left: Box::new(self),
            right: Box::new(other),
            predicates,
        }
    }

    pub fn project(self, _opt_ctx: &OptimizerCtx, cols: HashSet<usize>) -> RelExpr {
        match self {
            RelExpr::Project {
                src,
                cols: existing_cols,
            } => {
                if cols.is_subset(&existing_cols) {
                    src.project(_opt_ctx, cols)
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
            _ => RelExpr::Project {
                src: Box::new(self),
                cols,
            },
        }
    }

    pub fn map(
        self,
        opt_ctx: &OptimizerCtx,
        exprs: impl IntoIterator<Item = (usize, Expr)>,
    ) -> RelExpr {
        let mut exprs: Vec<(usize, Expr)> = exprs.into_iter().collect();

        if exprs.is_empty() {
            return self;
        }

        if opt_ctx.enabled(Rule::Hoist) {
            for i in 0..exprs.len() {
                // Only hoist expressions with subqueries
                if exprs[i].1.has_subquery() {
                    let (id, expr) = exprs.swap_remove(i);
                    return self.map(opt_ctx, exprs).hoist(opt_ctx, id, expr);
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
                for i in 0..exprs.len() {
                    let free = exprs[i].1.free();
                    if free.is_disjoint(&atts) {
                        existing_exprs.push(exprs.swap_remove(i));
                        return input.map(opt_ctx, existing_exprs).map(opt_ctx, exprs);
                    }
                }
                // We can't push the new exprs to existing_exprs
                RelExpr::Map {
                    input: Box::new(RelExpr::Map {
                        input,
                        exprs: existing_exprs,
                    }),
                    exprs,
                }
            }
            _ => RelExpr::Map {
                input: Box::new(self),
                exprs,
            },
        }
    }

    pub fn flatmap(self, opt_ctx: &OptimizerCtx, func: RelExpr) -> RelExpr {
        if opt_ctx.enabled(Rule::Decorrelate) {
            // Not correlated!
            if func.free().is_empty() {
                return self.join(JoinType::CrossJoin, func, vec![]);
            }

            // Pull up Project
            if let RelExpr::Project { src, mut cols } = func {
                cols.extend(self.att());
                return self.flatmap(opt_ctx, *src).project(opt_ctx, cols);
            }

            // Pull up Maps
            if let RelExpr::Map { input, exprs } = func {
                return self.flatmap(opt_ctx, *input).map(opt_ctx, exprs);
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

    fn hoist(self, opt_ctx: &OptimizerCtx, id: usize, expr: Expr) -> RelExpr {
        match expr {
            Expr::Subquery { expr } => {
                let att = expr.att();
                assert!(att.len() == 1);
                let input_col_id = att.iter().next().unwrap();
                // Give the column the name that's expected
                let rhs: RelExpr = expr.map(opt_ctx, vec![(id, Expr::col_ref(*input_col_id))]);
                self.flatmap(opt_ctx, rhs)
            }
            Expr::Binary { op, left, right } => {
                // Hoist the left, hoist the right, then perform the binary operation
                let lhs_id = opt_ctx.next();
                let rhs_id = opt_ctx.next();
                let att = self.att();
                self.hoist(opt_ctx, lhs_id, *left)
                    .hoist(opt_ctx, rhs_id, *right)
                    .map(
                        opt_ctx,
                        [(
                            id,
                            Expr::Binary {
                                op,
                                left: Box::new(Expr::col_ref(lhs_id)),
                                right: Box::new(Expr::col_ref(rhs_id)),
                            },
                        )],
                    )
                    .project(opt_ctx, att.into_iter().chain([id].into_iter()).collect())
            }
            Expr::Int { .. } | Expr::ColRef { .. } => self.map(opt_ctx, vec![(id, expr)]),
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
                out.push_str(&format!("{}-> project({:?})\n", " ".repeat(indent), cols));
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
                out.push_str(&format!(
                    "{}    group_by: {:?},\n",
                    " ".repeat(indent),
                    group_by
                ));
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
        }
    }
}
fn main() {
    let id_to_col_name = Rc::new(RefCell::new(HashMap::new()));
    let opt_ctx = OptimizerCtx::new(id_to_col_name);
    opt_ctx.enable(Rule::Hoist);
    opt_ctx.enable(Rule::Decorrelate);

    let a = opt_ctx.next();
    let b = opt_ctx.next();
    let x = opt_ctx.next();
    let y = opt_ctx.next();

    let sum_col = opt_ctx.next();

    let v = RelExpr::scan("a".into(), vec![a, b]).map(
        &opt_ctx,
        vec![
            // (
            //     opt_ctx.next(),
            //     Expr::int(3).plus(Expr::Subquery {
            //         expr: Box::new(
            //             RelExpr::scan("x".into(), vec![x, y]).project([x].into_iter().collect()),
            //         ),
            //     }),
            // ),
            (
                opt_ctx.next(),
                Expr::int(4).add(Expr::Subquery {
                    expr: Box::new(
                        RelExpr::scan("x".into(), vec![x, y])
                            .project(&opt_ctx, [x].into_iter().collect())
                            .map(
                                &opt_ctx,
                                [(sum_col, Expr::col_ref(x).add(Expr::col_ref(a)))],
                            )
                            .project(&opt_ctx, [sum_col].into_iter().collect()),
                    ),
                }),
            ),
        ],
    );

    // let v = RelExpr::scan("a".into(), vec![a, b]).map(
    //     &opt_ctx,
    //     vec![(
    //         opt_ctx.next(),
    //         Expr::Subquery {
    //             expr: Box::new(
    //                 RelExpr::scan("x".into(), vec![x, y])
    //                     .project(&opt_ctx, [x].into_iter().collect()),
    //             ),
    //         },
    //     )],
    // );

    let mut out = String::new();
    v.print_inner(0, &mut out);

    println!("{}", out);
}
