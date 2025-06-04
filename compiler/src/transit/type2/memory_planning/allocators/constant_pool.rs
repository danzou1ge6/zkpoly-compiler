use super::super::prelude::*;

/// Create an allocator that wraps around `inner`, adding a constant pool to it.
///
/// When allocating an object that is a constant object in this pool,
/// allocation and deallocation is intercepted and skipped,
/// while accesses to constant objects always return a valid pointer.
pub struct ConstantPool<A, T, P, Rt: RuntimeType> {
    inner: A,
    objects: BTreeMap<ObjectId, P>,
    _phantom: PhantomData<(T, P, Rt)>,
}

impl<A, T, P, Rt: RuntimeType> ConstantPool<A, T, P, Rt>
where
    A: Allocator<T, P, Rt>,
{
    /// `objects` are the constant objects in the pool, and the pointers must be
    /// distinct from those allocated and will be allocated in `inner`.
    pub fn new(inner: A, objects: impl Iterator<Item = (ObjectId, P)>) -> Self {
        Self {
            inner,
            objects: objects.collect(),
            _phantom: PhantomData,
        }
    }
}

pub struct Handle<'a, 'm, 's, 'i, 'au, A, T, P, Rt: RuntimeType> {
    allocator: &'a mut ConstantPool<A, T, P, Rt>,
    machine: planning::MachineHandle<'m, 's, T, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'i, 'au, A, P, Rt: RuntimeType> Handle<'a, 'm, 's, 'i, 'au, A, ObjectId, P, Rt>
where
    A: Allocator<ObjectId, P, Rt>,
{
    fn inner_handle<'se, 'd>(&'se mut self) -> Box<dyn AllocatorHandle<'s, ObjectId, P, Rt> + 'd>
    where
        'se: 'd,
        Self: 'd,
    {
        // unsafe here is safe because this method's signature enforces that `self` (i.e. the handle)
        // is not useable until the returned handle is dropped.
        let machine = unsafe { std::ptr::read(&mut self.machine) };
        self.allocator.inner.handle(machine, self.aux)
    }

    fn check_is_not_constant(&self, t: &ObjectId, msg: &'static str) {
        if self.allocator.objects.contains_key(t) {
            panic!("cannot {} object in constant pool", msg);
        }
    }
}

impl<'a, 'm, 's, 'i, 'au, A, P, Rt: RuntimeType> AllocatorHandle<'s, ObjectId, P, Rt>
    for Handle<'a, 'm, 's, 'i, 'au, A, ObjectId, P, Rt>
where
    A: Allocator<ObjectId, P, Rt>,
    P: Clone,
{
    fn access(&mut self, t: &ObjectId) -> Option<P> {
        if let Some(p) = self.allocator.objects.get(t) {
            Some(p.clone())
        } else {
            self.inner_handle().access(t)
        }
    }

    fn allocate<'f>(
        &mut self,
        size: Size,
        t: &ObjectId,
    ) -> Response<'f, planning::Machine<'s, ObjectId, P>, ObjectId, P, Result<P, Error<'s>>, Rt>
    where
        's: 'f,
    {
        if let Some(p) = self.allocator.objects.get(t) {
            Response::Complete(Ok(p.clone()))
        } else {
            self.inner_handle().allocate(size, t)
        }
    }

    fn claim<'f>(
        &mut self,
        t: &ObjectId,
        size: Size,
        from: Device,
    ) -> Response<'f, planning::Machine<'s, ObjectId, P>, ObjectId, P, Result<(), Error<'s>>, Rt>
    where
        's: 'f,
    {
        self.check_is_not_constant(t, "claim");
        self.inner_handle().claim(t, size, from)
    }

    fn completeness(&mut self, object: ObjectId) -> Completeness {
        if self.allocator.objects.contains_key(&object) {
            Completeness::plain_one()
        } else {
            self.inner_handle().completeness(object)
        }
    }

    fn deallocate<'f>(
        &mut self,
        t: &ObjectId,
    ) -> Response<'f, planning::Machine<'s, ObjectId, P>, ObjectId, P, (), Rt>
    where
        's: 'f,
    {
        if !self.allocator.objects.contains_key(t) {
            self.inner_handle().deallocate(t)
        } else {
            Response::Complete(())
        }
    }

    fn device(&self) -> Device {
        self.machine.device()
    }

    fn reuse(&mut self, new_object: ObjectId, old_object: ObjectId) {
        self.check_is_not_constant(&old_object, "reuse");
        self.inner_handle().reuse(new_object, old_object)
    }
}

pub struct Realizer<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType> {
    allocator: &'a mut ConstantPool<A, ObjectId, P, Rt>,
    machine: realization::MachineHandle<'m, 's, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType> Realizer<'a, 'm, 's, 'au, 'i, A, P, Rt>
where
    A: Allocator<ObjectId, P, Rt>,
    P: Clone,
{
    fn inner_realizer<'se, 'd>(
        &'se mut self,
    ) -> Box<dyn AllocatorRealizer<'s, ObjectId, P, Rt> + 'd>
    where
        'se: 'd,
        Self: 'd,
    {
        // unsafe here is safe because this method's signature enforces that `self` (i.e. the handle)
        // is not useable until the returned handle is dropped.
        let machine = unsafe { std::ptr::read(&mut self.machine) };
        self.allocator.inner.realizer(machine, self.aux)
    }
}

impl<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType> AllocatorRealizer<'s, ObjectId, P, Rt>
    for Realizer<'a, 'm, 's, 'au, 'i, A, P, Rt>
where
    A: Allocator<ObjectId, P, Rt>,
    P: UsizeId + 'static,
{
    fn allocate(&mut self, t: &ObjectId, pointer: &P) {
        if !self.allocator.objects.contains_key(t) {
            self.inner_realizer().allocate(t, pointer)
        }
    }

    fn deallocate(&mut self, t: &ObjectId, pointer: &P) {
        if !self.allocator.objects.contains_key(t) {
            self.inner_realizer().deallocate(t, pointer)
        }
    }

    fn transfer<'f>(
        &mut self,
        t: &ObjectId,
        from_pointer: &P,
        to_device: Device,
        to_pointer: &P,
    ) -> RealizationResponse<'f, 's, ObjectId, P, Result<(), Error<'s>>, Rt> {
        Response::Continue(Continuation::transfer_object(
            self.machine.device(),
            *from_pointer,
            to_device,
            *to_pointer,
            *t,
        ))
    }
}

impl<P, A, Rt: RuntimeType> Allocator<ObjectId, P, Rt> for ConstantPool<A, ObjectId, P, Rt>
where
    A: Allocator<ObjectId, P, Rt>,
    P: UsizeId + 'static,
{
    fn handle<'a, 'b, 'c, 'd, 's, 'i>(
        &'a mut self,
        machine: planning::MachineHandle<'b, 's, ObjectId, P>,
        aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorHandle<'s, ObjectId, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd,
    {
        Box::new(Handle {
            allocator: self,
            machine,
            aux,
        })
    }

    fn realizer<'a, 'b, 'c, 'd, 's, 'i>(
        &'a mut self,
        machine: realization::MachineHandle<'b, 's, P>,
        aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorRealizer<'s, ObjectId, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd,
    {
        Box::new(Realizer {
            allocator: self,
            machine,
            aux,
        })
    }
}
