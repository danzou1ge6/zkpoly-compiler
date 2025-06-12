use super::super::prelude::*;

/// Create an allocator that wraps around `inner`, adding a constant pool to it.
///
/// When allocating an object that is a constant object in this pool,
/// allocation and deallocation is intercepted and skipped,
/// while accesses to constant objects always return a valid pointer.
pub struct Wrapper<'f, A, T, P, Rt: RuntimeType, D: DeviceMarker> {
    inner: A,
    objects: BTreeMap<ObjectId, P>,
    _phantom: PhantomData<(&'f T, P, Rt, D)>,
}

impl<'f, A, T, P, Rt: RuntimeType, D: DeviceMarker> Wrapper<'f, A, T, P, Rt, D>
where
    A: Allocator<'f, T, P, Rt>,
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

    pub fn unwrap(self) -> A {
        self.inner
    }
}

pub struct Handle<'a, 'm, 's, 'i, 'au, A, T, P, Rt: RuntimeType, D: DeviceMarker> {
    allocator: &'a mut Wrapper<'s, A, T, P, Rt, D>,
    machine: planning::MachineHandle<'m, 's, T, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'i, 'au, 'f, A, P, Rt: RuntimeType, D: DeviceMarker>
    Handle<'a, 'm, 's, 'i, 'au, A, ObjectId, P, Rt, D>
where
    A: Allocator<'s, ObjectId, P, Rt>,
{
    fn inner_handle<'se, 'd>(&'se mut self) -> Box<dyn AllocatorHandle<'s, ObjectId, P, Rt> + 'd>
    where
        'se: 'd,
        Self: 'd,
        's: 'f,
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

impl<'a, 'm, 's, 'i, 'au, A, P, Rt: RuntimeType, D: DeviceMarker>
    AllocatorHandle<'s, ObjectId, P, Rt> for Handle<'a, 'm, 's, 'i, 'au, A, ObjectId, P, Rt, D>
where
    A: Allocator<'s, ObjectId, P, Rt>,
    P: Clone,
    's: 'a,
{
    fn access(&mut self, t: &ObjectId) -> Option<P> {
        if let Some(p) = self.allocator.objects.get(t) {
            Some(p.clone())
        } else {
            self.inner_handle().access(t)
        }
    }

    fn allocate(&mut self, size: Size, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, P, Rt> {
        if let Some(p) = self.allocator.objects.get(t) {
            Response::Complete(Ok(p.clone()))
        } else {
            self.inner_handle().allocate(size, t)
        }
    }

    fn claim(
        &mut self,
        t: &ObjectId,
        size: Size,
        from: Device,
    ) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
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

    fn deallocate(&mut self, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        if !self.allocator.objects.contains_key(t) {
            self.inner_handle().deallocate(t)
        } else {
            Response::ok(())
        }
    }

    fn device(&self) -> Device {
        self.machine.device()
    }

    fn reuse(&mut self, new_object: ObjectId, old_object: ObjectId) {
        self.check_is_not_constant(&old_object, "reuse");
        self.inner_handle().reuse(new_object, old_object)
    }

    fn typeid(&self) -> typeid::ConstTypeId {
        typeid::ConstTypeId::of::<Self>()
    }
}

pub struct Realizer<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType, D: DeviceMarker> {
    allocator: &'a mut Wrapper<'s, A, ObjectId, P, Rt, D>,
    machine: realization::MachineHandle<'m, 's, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType, D: DeviceMarker>
    Realizer<'a, 'm, 's, 'au, 'i, A, P, Rt, D>
where
    A: Allocator<'s, ObjectId, P, Rt>,
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

impl<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType, D: DeviceMarker>
    AllocatorRealizer<'s, ObjectId, P, Rt> for Realizer<'a, 'm, 's, 'au, 'i, A, P, Rt, D>
where
    A: Allocator<'s, ObjectId, P, Rt>,
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

impl<'s, P, A, Rt: RuntimeType, D: DeviceMarker> Allocator<'s, ObjectId, P, Rt>
    for Wrapper<'s, A, ObjectId, P, Rt, D>
where
    A: Allocator<'s, ObjectId, P, Rt>,
    P: UsizeId + 'static,
{
    fn handle<'a, 'b, 'c, 'd, 'i>(
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

    fn realizer<'a, 'b, 'c, 'd, 'i>(
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

    fn allcate_pointer(&mut self) -> P {
        self.inner.allcate_pointer()
    }

    fn inner<'t>(&'t mut self) -> Option<&'t mut dyn Allocator<'s, ObjectId, P, Rt>> {
        Some(&mut self.inner)
    }
}
