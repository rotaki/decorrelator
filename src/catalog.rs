use std::{cell::RefCell, collections::HashMap, rc::Rc};

pub enum DataType {
    Int,
    String,
    Bool,
}

pub struct Column {
    pub name: String,
    pub data_type: DataType,
}

pub type ColumnRef = Rc<Column>;

pub struct Schema {
    pub name: String,
    pub columns: Vec<ColumnRef>,
}

pub type SchemaRef = Rc<Schema>;

pub struct Table {
    pub name: String,
    pub schema: SchemaRef,
}

pub type TableRef = Rc<Table>;

pub struct Catalog {
    col_ids: Rc<RefCell<HashMap<usize, String>>>,
    tables: RefCell<Vec<TableRef>>,
}

impl Catalog {
    pub fn new() -> Self {
        Catalog {
            col_ids: Rc::new(RefCell::new(HashMap::new())),
            tables: RefCell::new(vec![]),
        }
    }

    pub fn add_table(&self, table: Table) {
        let current_id = self.col_ids.borrow().len();
        for (i, col) in table.schema.columns.iter().enumerate() {
            let name = format!("{}.{}", table.name, col.name);
            self.col_ids.borrow_mut().insert(current_id + i, name);
        }
        self.tables.borrow_mut().push(Rc::new(table));
    }

    pub fn get_table(&self, table_name: &str) -> Option<TableRef> {
        self.tables
            .borrow()
            .iter()
            .find(|t| t.name == table_name)
            .cloned()
    }

    pub fn get_table_names(&self) -> Vec<String> {
        self.tables
            .borrow()
            .iter()
            .map(|t| t.name.clone())
            .collect()
    }

    pub fn get_table_schema(&self, table_name: &str) -> Option<SchemaRef> {
        self.get_table(table_name).map(|t| t.schema.clone())
    }

    pub fn is_valid_table(&self, table_name: &str) -> bool {
        self.get_table(table_name).is_some()
    }

    pub fn is_valid_column(&self, table_name: &str, column_name: &str) -> bool {
        self.get_table_schema(table_name)
            .map(|s| s.columns.iter().any(|c| c.name == column_name))
            .unwrap_or(false)
    }

    pub fn find_column(&self, column_name: &str) -> Option<(TableRef, ColumnRef)> {
        for table in self.tables.borrow().iter() {
            for column in table.schema.columns.iter() {
                if column.name == column_name {
                    return Some((table.clone(), column.clone()));
                }
            }
        }
        None
    }

    pub fn get_col_ids(&self) -> Rc<RefCell<HashMap<usize, String>>> {
        self.col_ids.clone()
    }

    pub fn get_col_ids_of_table(&self, table_name: &str) -> Vec<usize> {
        self.col_ids
            .borrow()
            .iter()
            .filter_map(|(id, name)| {
                if name.starts_with(table_name) {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_col_id(&self, name: &str) -> Option<usize> {
        self.col_ids
            .borrow()
            .iter()
            .find(|(_, n)| *n == name)
            .map(|(id, _)| *id)
    }
}

pub type CatalogRef = Rc<Catalog>;
