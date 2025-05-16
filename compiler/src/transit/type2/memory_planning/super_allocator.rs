use std::collections::BTreeMap;

use zkpoly_common::heap::{IdAllocator, UsizeId};

use crate::transit::{
    type2::object_analysis::ObjectId,
    type3::{Device, Size},
};

pub struct SuperAllocator<T, P> {
    mapping: BTreeMap<T, P>,
    p_allocator: IdAllocator<P>,
}

impl<T, P> SuperAllocator<T, P> {
    pub fn new() -> Self {
        Self {
            mapping: BTreeMap::new(),
            p_allocator: IdAllocator::new(),
        }
    }
}

pub struct Handle<'s, 'a, 'm, T, P> {
    allocator: &'a mut SuperAllocator<T, P>,
    machine: super::MachineHandle<'m, 's, T, P>,
}

impl<'s, 'a, 'm, T, P> super::AllocatorHandle<T, P> for Handle<'s, 'a, 'm, T, P>
where
    T: Ord + Clone + std::fmt::Debug,
    P: UsizeId,
{
    fn device(&self) -> Device {
        self.machine.device
    }

    fn access(&self, t: &T) -> Option<P> {
        self.allocator.mapping.get(&t).cloned()
    }

    fn allocate(&mut self, _size: Size, t: &T) -> P {
        let p = *self
            .allocator
            .mapping
            .entry(t.clone())
            .or_insert_with(|| self.allocator.p_allocator.alloc());

        self.machine.allocate(t.clone(), p);

        p
    }

    fn deallocate(&mut self, t: &T) {
        let p = self.allocator.mapping.remove(t).unwrap();

        self.machine.allocate(t.clone(), p);
    }

    fn read(&mut self, t: &T) {
        if !self.allocator.mapping.contains_key(t) {
            self.panic_no_token(t)
        }
    }

    fn read_write(&mut self, read_t: &T, write_t: &T) {
        let p = self
            .allocator
            .mapping
            .remove(read_t)
            .unwrap_or_else(|| self.panic_no_token(read_t));

        self.allocator.mapping.insert(write_t.clone(), p);
    }
}

impl<T, P> super::Allocator<T, P> for SuperAllocator<T, P>
where
    T: Ord + Clone + std::fmt::Debug,
    P: UsizeId,
{
    fn handle<'a: 'd, 'b: 'd, 'c: 'd, 'd>(
        &'a mut self,
        machine: super::MachineHandle<'b, '_, T, P>,
        _aux: &'c super::AuxiliaryInfo,
    ) -> Box<dyn super::AllocatorHandle<T, P> + 'd>
    {
        Box::new(Handle {
            allocator: self,
            machine,
        })
    }
}
