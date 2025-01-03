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
    allocator: Rc<dyn Allocator<F>>,
}

impl<F: Field> Polynomial<F> {
    pub fn new(typ: PolyType, log_n: u32, allocator: Rc<dyn Allocator<F>>) -> Self {
        let ptr = allocator.allocate(1 << log_n);
        Self {
            values: ptr,
            typ,
            log_n,
            rotate: 0,
            device: allocator.device_type(),
            allocator,
        }
    }
}

impl<F: Field> Drop for Polynomial<F> {
    fn drop(&mut self) {
        self.allocator.deallocate(self.values);
    }
}
