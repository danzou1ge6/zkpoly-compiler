use crate::typ::Typ;
use std::any;
use zkpoly_common::heap;

zkpoly_common::define_usize_id!(ConstantId);

#[derive(Debug)]
pub struct Constant {
    name: String,
    typ: Typ,
    value: Box<dyn any::Any>,
}

impl Constant {
    pub fn new(name: String, typ: Typ, value: Box<dyn any::Any>) -> Self {
        Self { name, typ, value }
    }
}

pub type ConstantTable = heap::Heap<ConstantId, Constant>;
