use std::{cell::RefCell, rc::Rc};

pub struct ColIdGenerator {
    current_id: RefCell<usize>,
}

impl ColIdGenerator {
    pub fn new() -> ColIdGenerator {
        ColIdGenerator {
            // Start from 10000
            current_id: RefCell::new(10000),
        }
    }

    pub fn next(&self) -> usize {
        let id = *self.current_id.borrow();
        *self.current_id.borrow_mut() += 1;
        id
    }
}

pub type ColIdGeneratorRef = Rc<ColIdGenerator>;
