use super::{normalize_size, smithereens_pool};
use crate::transit::type2::memory_planning::prelude::*;
use planning::machine::*;

/// A wrapper that adds a pool for small objects to the underlying allocator.
/// The smithereen pool will not eject objects, and errors when space is insufficient.
pub struct Wrapper<'f, A, P, Rt: RuntimeType, D: DeviceMarker> {
    inner: A,
    sa: smithereens_pool::Pool,
    /// Only tracks smithereen objects
    objects_at: Bijection<ObjectId, P>,
    /// Only tracks pointers to smithereen objects
    mapping: BTreeMap<P, Addr>,
    offset: u64,
    _phantom: PhantomData<(&'f Rt, D)>,
}

impl<'f, A, P, Rt: RuntimeType, D: DeviceMarker> Wrapper<'f, A, P, Rt, D> {
    pub fn new(inner: A, capacity: u64, offset: u64) -> Self {
        Self {
            inner,
            sa: smithereens_pool::Pool::new(capacity),
            objects_at: Bijection::new(),
            mapping: BTreeMap::new(),
            offset,
            _phantom: PhantomData,
        }
    }
}

pub struct Handle<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType, D: DeviceMarker> {
    allocator: &'a mut Wrapper<'s, A, P, Rt, D>,
    machine: MachineHandle<'m, 's, ObjectId, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType, D: DeviceMarker>
    Handle<'a, 'm, 's, 'au, 'i, A, P, Rt, D>
where
    A: Allocator<'s, ObjectId, P, Rt>,
    P: 'static + UsizeId,
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
}

impl<'a, 'm, 's, 'au, 'i, A, P, Rt: RuntimeType, D: DeviceMarker>
    AllocatorHandle<'s, ObjectId, P, Rt> for Handle<'a, 'm, 's, 'au, 'i, A, P, Rt, D>
where
    A: Allocator<'s, ObjectId, P, Rt>,
    P: UsizeId + 'static,
{
    fn access(&mut self, t: &ObjectId) -> Option<P> {
        if let Some(p) = self.allocator.objects_at.get_forward(t) {
            Some(*p)
        } else {
            self.inner_handle().access(t)
        }
    }

    fn allocate(&mut self, size: Size, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, P, Rt> {
        let size = normalize_size(size);

        match size {
            Size::Integral(_) => self.inner_handle().allocate(size, t),
            Size::Smithereen(ss) => {
                if let Some(addr) = self.allocator.sa.allocate(ss) {
                    let p = self.allocator.inner.allcate_pointer();
                    self.allocator.mapping.insert(p, addr);
                    self.allocator.objects_at.insert(*t, p);
                    self.machine.allocate(*t, p);
                    Response::Complete(Ok(p))
                } else {
                    Response::Complete(Err(Error::InsufficientSmithereenSpace))
                }
            }
        }
    }

    fn claim(
        &mut self,
        t: &ObjectId,
        size: Size,
        from: Device,
    ) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        let size = normalize_size(size);

        let object = *t;
        let this_device = self.machine.device();

        match size {
            Size::Integral(_) => self.inner_handle().claim(t, size, from),
            Size::Smithereen(_) => {
                let resp = self.allocate(size, t);
                resp.bind_result(move |p| {
                    Continuation::simple_provide(this_device, p, from, object)
                })
            }
        }
    }

    fn completeness(&mut self, object: ObjectId) -> Completeness {
        if let Some(_) = self.allocator.objects_at.get_forward(&object) {
            Completeness::plain_one()
        } else {
            self.inner_handle().completeness(object)
        }
    }

    fn deallocate(&mut self, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        if let Some(p) = self.allocator.objects_at.get_forward(t) {
            self.allocator
                .sa
                .deallocate(*self.allocator.mapping.get(p).unwrap());
            self.machine.deallocate(*t, *p);
            self.allocator.objects_at.remove_forward(t);
            Response::Complete(Ok(()))
        } else {
            self.inner_handle().deallocate(t)
        }
    }

    fn device(&self) -> Device {
        self.machine.device()
    }

    fn reuse(&mut self, new_object: ObjectId, old_object: ObjectId) {
        if let Some(p) = self.allocator.objects_at.remove_forward(&old_object) {
            self.allocator.objects_at.insert(new_object, p);
        } else {
            self.inner_handle().reuse(new_object, old_object);
        }
    }

    fn typeid(&self) -> typeid::ConstTypeId {
        typeid::ConstTypeId::of::<Self>()
    }
}

pub struct Realizer<'a, 'm, 's, 'au, 'i, 'f, A, P, Rt: RuntimeType, D: DeviceMarker> {
    allocator: &'a mut Wrapper<'f, A, P, Rt, D>,
    machine: realization::MachineHandle<'m, 's, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, 'f, A, P, Rt: RuntimeType, D: DeviceMarker>
    Realizer<'a, 'm, 's, 'au, 'i, 'f, A, P, Rt, D>
where
    P: UsizeId + 'static,
    A: Allocator<'s, ObjectId, P, Rt>,
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

impl<'a, 'm, 's, 'au, 'i, 'f, A, P, Rt: RuntimeType, D: DeviceMarker>
    AllocatorRealizer<'s, ObjectId, P, Rt> for Realizer<'a, 'm, 's, 'au, 'i, 'f, A, P, Rt, D>
where
    P: UsizeId + 'static,
    A: Allocator<'s, ObjectId, P, Rt>,
{
    fn allocate(&mut self, t: &ObjectId, pointer: &P) {
        if let Some(addr) = self.allocator.mapping.get(pointer) {
            let size = self.aux.obj_info().size(*t);
            let addr =
                AllocMethod::Offset(addr.offset(self.allocator.offset).get().into(), size.into());
            let vn = self.aux.obj_info().typ(*t).with_normalized_p();
            self.machine.allocate(
                addr,
                ResidentalValue::new(Value::new(*t, self.machine.device(), vn), *pointer),
            );
        } else {
            self.inner_realizer().allocate(t, pointer);
        }
    }

    fn deallocate(&mut self, t: &ObjectId, pointer: &P) {
        if let Some(..) = self.allocator.mapping.get(pointer) {
            self.machine
                .deallocate_object(*t, pointer, self.aux.obj_info(), AllocVariant::Offset);
        } else {
            self.inner_realizer().deallocate(t, pointer);
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

impl<'s, A, P, Rt: RuntimeType> Allocator<'s, ObjectId, P, Rt> for Wrapper<'s, A, P, Rt, Cpu>
where
    A: Allocator<'s, ObjectId, P, Rt>,
    P: UsizeId + 'static,
{
    fn allcate_pointer(&mut self) -> P {
        self.inner.allcate_pointer()
    }

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

    fn inner<'t>(&'t mut self) -> Option<&'t mut dyn Allocator<'s, ObjectId, P, Rt>> {
        Some(&mut self.inner)
    }
}

impl<'s, A, P, Rt: RuntimeType> Allocator<'s, ObjectId, P, Rt> for Wrapper<'s, A, P, Rt, Gpu>
where
    A: Allocator<'s, ObjectId, P, Rt>,
    P: UsizeId + 'static,
{
    fn allcate_pointer(&mut self) -> P {
        self.inner.allcate_pointer()
    }

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

    fn inner<'t>(&'t mut self) -> Option<&'t mut dyn Allocator<'s, ObjectId, P, Rt>> {
        Some(&mut self.inner)
    }
}
