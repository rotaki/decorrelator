#[derive(Debug, Clone, PartialEq)]
pub enum Field {
    Int(i64),
    String(String),
    Bool(bool),
    Null,
}

impl std::fmt::Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Field::Int(i) => write!(f, "{}", i),
            Field::String(s) => write!(f, "{}", s),
            Field::Bool(b) => write!(f, "{}", b),
            Field::Null => write!(f, "NULL"),
        }
    }
}
