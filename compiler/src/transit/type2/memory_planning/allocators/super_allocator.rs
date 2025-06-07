use crate::transit::type2::memory_planning::prelude::*;
use planning::machine::*;

pub struct SuperAllocator<'f, P> {
    mapping: BTreeMap<ObjectId, P>,
    p_allocator: IdAllocator<P>,
    /// If `physical` is false, then this allocator will not instruct machine for allocation or deallocation.
    /// That is used in case when the underlying device is unplanned.
    physical: bool,
    _phantom: PhantomData<&'f u32>,
}

impl<'f, P> SuperAllocator<'f, P> {
    pub fn for_unplanned() -> Self {
        Self {
            mapping: BTreeMap::new(),
            p_allocator: IdAllocator::new(),
            physical: false,
            _phantom: PhantomData,
        }
    }

    pub fn new(p_allocator: IdAllocator<P>) -> Self {
        Self {
            mapping: BTreeMap::new(),
            p_allocator,
            physical: true,
            _phantom: PhantomData,
        }
    }
}

pub struct Handle<'s, 'a, 'm, P> {
    allocator: &'a mut SuperAllocator<'s, P>,
    machine: MachineHandle<'m, 's, ObjectId, P>,
}

impl<'s, 'a, 'm, P, Rt: RuntimeType> AllocatorHandle<'s, ObjectId, P, Rt> for Handle<'s, 'a, 'm, P>
where
    P: UsizeId + 'static,
{
    fn device(&self) -> Device {
        self.machine.device()
    }

    fn access(&mut self, t: &ObjectId) -> Option<P> {
        self.allocator.mapping.get(&t).cloned()
    }

    fn allocate(&mut self, _size: Size, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, P, Rt> {
        if self.allocator.mapping.contains_key(t) {
            panic!("{:?} already allocated on {:?}", t, self.machine.device());
        }

        let p = *self
            .allocator
            .mapping
            .entry(t.clone())
            .or_insert_with(|| self.allocator.p_allocator.alloc());

        if self.allocator.physical {
            self.machine.allocate(t.clone(), p);
        }

        Response::ok(p)
    }

    fn deallocate(&mut self, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        let p = self
            .allocator
            .mapping
            .remove(t)
            .unwrap_or_else(|| panic!("token {:?} not allocated", t));

        if self.allocator.physical {
            self.machine.deallocate(t.clone(), p);
        }

        Response::ok(())
    }

    fn claim(
        &mut self,
        t: &ObjectId,
        _size: Size,
        from: Device,
    ) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        if self.allocator.mapping.contains_key(t) {
            return Response::Complete(Ok(()));
        }

        // fixme
        println!("claim object {:?} from {:?}", t, from);

        let p = self.allocator.p_allocator.alloc();
        self.allocator.mapping.insert(t.clone(), p);

        Response::Continue(Continuation::simple_provide(
            self.machine.device(),
            p,
            from,
            t.clone(),
        ))
    }

    fn completeness(&mut self, object: ObjectId) -> Completeness {
        if self.allocator.mapping.contains_key(&object) {
            Completeness::plain_one()
        } else {
            Completeness::plain_zero()
        }
    }

    fn reuse(&mut self, new_object: ObjectId, old_object: ObjectId) {
        let p = self
            .allocator
            .mapping
            .remove(&old_object)
            .unwrap_or_else(|| panic!("token {:?} not allocated", old_object));

        // fixme
        println!("reuse object {:?} to {:?}", old_object, new_object);

        self.allocator.mapping.insert(new_object.clone(), p);
    }

    fn evoke(
        &mut self,
        _procedure: allocator::ProcedureId,
    ) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        panic!("this allocator does not produce continuation that evokes procedures")
    }
}

pub struct Realizer<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType> {
    allocator: &'a mut SuperAllocator<'s, P>,
    machine: realization::MachineHandle<'m, 's, P>,
    aux: &'au AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, 'f, P, Rt: RuntimeType> AllocatorRealizer<'s, ObjectId, P, Rt>
    for Realizer<'a, 'm, 's, 'au, 'i, P, Rt>
where
    P: UsizeId + 'static,
{
    fn allocate(&mut self, t: &ObjectId, pointer: &P) {
        let vn = self.aux.obj_info().typ(*t).with_normalized_p();
        let size = self.aux.obj_info().size(*t);
        let rv = ResidentalValue::new(Value::new(*t, self.machine.device(), vn), *pointer);
        self.machine.cpu_allocate(size, rv);
    }

    fn deallocate(&mut self, t: &ObjectId, pointer: &P) {
        let vn = self.aux.obj_info().typ(*t).with_normalized_p();
        let rv = ResidentalValue::new(Value::new(*t, self.machine.device(), vn), *pointer);
        self.machine.cpu_deallocate(&rv);
    }

    fn transfer(
        &mut self,
        t: &ObjectId,
        from_pointer: &P,
        to_device: Device,
        to_pointer: &P,
    ) -> RealizationResponse<'s, ObjectId, P, Result<(), Error<'s>>, Rt> {
        Response::Continue(Continuation::transfer_object(
            self.machine.device(),
            *from_pointer,
            to_device,
            *to_pointer,
            *t,
        ))
    }
}

impl<'s, P, Rt: RuntimeType> Allocator<'s, ObjectId, P, Rt> for SuperAllocator<'s, P>
where
    P: UsizeId + 'static,
{
    fn handle<'a: 'd, 'b: 'd, 'c: 'd, 'd, 'i: 'd>(
        &'a mut self,
        machine: MachineHandle<'b, 's, ObjectId, P>,
        _aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorHandle<'s, ObjectId, P, Rt> + 'd> {
        Box::new(Handle {
            allocator: self,
            machine,
        })
    }

    fn realizer<'a, 'b, 'c, 'd, 'i: 'd>(
        &'a mut self,
        machine: realization::MachineHandle<'b, 's, P>,
        aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorRealizer<'s, ObjectId, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
    {
        Box::new(Realizer {
            allocator: self,
            machine,
            aux,
        })
    }
}
