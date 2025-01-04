use std::rc::Rc;

use crate::devices::DeviceType;
use crate::{mem::Allocator, typ::PolyType};
use group::ff::Field;

pub struct Polynomial<F: Field> {
    values: *mut F,
    typ: PolyType,
    log_n: u32,
    rotate: u64,
    device: DeviceType,
}

impl<F: Field> Polynomial<F> {
    pub fn new(typ: PolyType, log_n: u32, allocator: Box<dyn Allocator<F>>) -> Self {
        let ptr = allocator.allocate(1 << log_n);
        Self {
            values: ptr,
            typ,
            log_n,
            rotate: 0,
            device: allocator.device_type(),
        }
    }
}
