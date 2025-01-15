use crate::devices::DeviceType;
use crate::typ::Typ;
use std::{any, sync::RwLock};
use zkpoly_common::heap;

zkpoly_common::define_usize_id!(VariableId);
zkpoly_common::define_usize_id!(ConstantId);

pub type VariableTable = heap::Heap<VariableId, RwLock<Option<Variable>>>;
pub type ConstantTable = heap::Heap<ConstantId, Constant>;

pub enum ArgId {
    Variable(VariableId),
    Constant(ConstantId),
}

#[derive(Debug)]
pub struct Constant {
    name: String,
    typ: Typ,
    device: DeviceType,
    value: Box<dyn any::Any>,
}

impl Constant {
    pub fn new(name: String, typ: Typ, value: Box<dyn any::Any>) -> Self {
        let device = DeviceType::CPU;
        Self {
            name,
            typ,
            device,
            value,
        }
    }
}

#[derive(Debug)]
pub struct Variable {
    pub typ: Typ,
    pub device: DeviceType,
    pub value: Box<dyn any::Any>,
}

impl Variable {
    pub fn new(typ: Typ, device: DeviceType, value: Box<dyn any::Any>) -> Self {
        Self { typ, device, value }
    }
}

unsafe impl Send for Variable {}
unsafe impl Sync for Variable {}
