use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

pub enum DataType {
    Int,
    String,
    Bool,
}

pub struct Column {
    pub name: String,
    pub data_type: DataType,
}

impl Column {
    pub fn new(name: &str, data_type: DataType) -> ColumnRef {
        Rc::new(Column {
            name: name.to_string(),
            data_type,
        })
    }
}

pub type ColumnRef = Rc<Column>;

pub struct Schema {
    pub columns: Vec<ColumnRef>,
}

impl Schema {
    pub fn new(columns: Vec<ColumnRef>) -> SchemaRef {
        Rc::new(Schema { columns })
    }
}

pub type SchemaRef = Rc<Schema>;

pub struct Table {
    pub name: String,
    pub schema: SchemaRef,
}

impl Table {
    pub fn new(name: &str, schema: SchemaRef) -> TableRef {
        Rc::new(Table {
            name: name.to_string(),
            schema,
        })
    }
}

pub type TableRef = Rc<Table>;

pub struct Catalog {
    col_ids: Rc<RefCell<HashMap<usize, String>>>, // id -> table_name.column_name
    tables: RefCell<Vec<TableRef>>,
}

impl Catalog {
    pub fn new() -> Self {
        Catalog {
            col_ids: Rc::new(RefCell::new(HashMap::new())),
            tables: RefCell::new(vec![]),
        }
    }

    pub fn add_table(&self, table: TableRef) {
        let current_id = self.col_ids.borrow().len();
        for (i, col) in table.schema.columns.iter().enumerate() {
            let name = format!("{}.{}", table.name, col.name);
            self.col_ids.borrow_mut().insert(current_id + i, name);
        }
        self.tables.borrow_mut().push(table);
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

    pub fn find_column(&self, column_name: &str) -> Vec<(TableRef, ColumnRef)> {
        let mut result = vec![];
        for table in self.tables.borrow().iter() {
            for column in table.schema.columns.iter() {
                if column.name == column_name {
                    result.push((table.clone(), column.clone()));
                }
            }
        }
        result
    }

    pub fn get_col_ids(&self) -> Rc<RefCell<HashMap<usize, String>>> {
        self.col_ids.clone()
    }

    // Input: Table name
    pub fn get_col_ids_of_table(&self, table_name: &str) -> HashSet<usize> {
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

    // Input: Disambiguated column name (table_name.column_name)
    pub fn get_col_id(&self, name: &str) -> Option<usize> {
        self.col_ids
            .borrow()
            .iter()
            .find(|(_, n)| *n == name)
            .map(|(id, _)| *id)
    }

    pub fn get_cols(&self, table_name: &str) -> Vec<(String, usize)> {
        // find table_index in the vector of tables
        let table_index = self
            .tables
            .borrow()
            .iter()
            .position(|t| t.name == table_name)
            .unwrap();
        // return column name, table_index * 100 + column_index
        self.tables
            .borrow()
            .get(table_index)
            .unwrap()
            .schema
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.clone(), table_index * 100 + i))
            .collect()
    }
}

pub type CatalogRef = Rc<Catalog>;
