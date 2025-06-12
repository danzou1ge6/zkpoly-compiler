use super::slab_pool::AddrId;
use super::{normalize_size, slab_pool};
use crate::transit::type2::memory_planning::prelude::*;
use planning::machine::*;

pub type AddrMapping = Heap<AddrId, (Addr, IntegralSize)>;

pub struct OffsettedAddrMapping<'a> {
    mapping: &'a mut AddrMapping,
    offset: u64,
}

impl<'a> OffsettedAddrMapping<'a> {
    pub fn new(mapping: &'a mut AddrMapping, offset: u64) -> Self {
        Self { mapping, offset }
    }
}

impl<'a> slab_pool::AddrMappingHandler for OffsettedAddrMapping<'a> {
    fn add(&mut self, addr: Addr, size: IntegralSize) -> AddrId {
        self.mapping.push((addr.offset(self.offset), size))
    }

    fn update(&mut self, id: AddrId, addr: Addr, size: IntegralSize) {
        self.mapping[id] = (addr.offset(self.offset), size)
    }

    fn get(&self, id: AddrId) -> (Addr, IntegralSize) {
        let (addr, size) = self.mapping[id];
        (addr.unoffset(self.offset), size)
    }
}

pub struct SlabAllocator<'f, P, Rt: RuntimeType, D: DeviceMarker> {
    mapping: AddrMapping,
    /// Associate each object with its residence.
    /// Don't forget to update this mapping.
    objects_at: Bijection<ObjectId, AddrId>,
    offset: u64,
    ia: slab_pool::Allocator,
    _phantom: PhantomData<(&'f P, Rt, D)>,
}

impl<'f, P, Rt: RuntimeType, D: DeviceMarker> SlabAllocator<'f, P, Rt, D> {
    pub fn new(capacity: u64, offset: u64, lbss: Vec<IntegralSize>) -> Self {
        Self {
            mapping: AddrMapping::new(),
            objects_at: Bijection::new(),
            offset,
            ia: slab_pool::Allocator::new(capacity, lbss),
            _phantom: PhantomData,
        }
    }
}

/// All sizes entering the allocator must be normalized first.
pub struct Handle<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType, D: DeviceMarker> {
    allocator: &'a mut SlabAllocator<'s, P, Rt, D>,
    machine: MachineHandle<'m, 's, ObjectId, P>,
    aux: &'au mut AuxiliaryInfo<'i, Rt>,
}

fn p<P: UsizeId>(addr_id: AddrId) -> P {
    P::from(usize::from(addr_id))
}

fn p_inv<P: UsizeId>(addr_id: P) -> AddrId {
    AddrId::from(addr_id.into())
}

macro_rules! mapping_handler {
    ($self:expr, $offset:expr) => {
        &mut OffsettedAddrMapping::new(&mut $self.allocator.mapping, $offset)
    };
}

impl<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType, D: DeviceMarker> Handle<'a, 'm, 's, 'au, 'i, P, Rt, D>
where
    P: UsizeId + 'static,
{
    fn update_next_use(&mut self, addr_id: AddrId, next_use: Option<Index>) {
        self.allocator.ia.update_next_use(
            addr_id,
            next_use.unwrap_or(Index::inf()),
            mapping_handler!(self, self.allocator.offset),
        );
    }

    fn tick(&mut self) {
        let pc = self.aux.pc();
        self.allocator.ia.tick(pc);
    }
}

impl<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType, D: DeviceMarker> AllocatorHandle<'s, ObjectId, P, Rt>
    for Handle<'a, 'm, 's, 'au, 'i, P, Rt, D>
where
    P: UsizeId + 'static,
    SlabAllocator<'s, P, Rt, D>: Allocator<'s, ObjectId, P, Rt>,
{
    fn access(&mut self, t: &ObjectId) -> Option<P> {
        let updated_next_use = self.aux.mark_use(*t, self.device());

        let addr = self.allocator.objects_at.get_forward(t).copied();

        // fixme
        println!("GPU access {:?} got {:?}", t, addr);

        if let Some(addr) = addr {
            self.update_next_use(addr, updated_next_use);
        }

        addr.map(p)
    }

    fn allocate(&mut self, size: Size, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, P, Rt> {
        println!("GPU allocating {:?}", t);

        if self.allocator.objects_at.get_forward(t).is_some() {
            panic!("object {:?} already allocated", t);
        }

        self.tick();

        let size = normalize_size(size);
        match size {
            Size::Smithereen(..) => {
                panic!("this allocator is not responsible for smithereen objects")
            }
            Size::Integral(is) => {
                let next_use = self.aux.query_next_use(*t, self.device());

                let r = self.allocator.ia.allocate(
                    is,
                    next_use.unwrap_or(Index::inf()),
                    mapping_handler!(self, self.allocator.offset),
                );

                if let Some((transfers, addr)) = r {
                    for slab_pool::Transfer { from, to } in transfers {
                        let object = self
                            .allocator
                            .objects_at
                            .remove_backward(&from)
                            .expect("expected object on device to reallocate it");
                        self.machine.allocate(object, p(to));
                        self.machine
                            .transfer(self.device(), p(to), object, self.device(), p(from));
                        self.machine.deallocate(object, p(from));
                        self.allocator.objects_at.insert_checked(object, to);
                    }

                    self.allocator.objects_at.insert_checked(*t, addr);
                    self.machine.allocate(*t, p(addr));
                    Response::Complete(Ok(p(addr)))
                } else {
                    if let Some((addr, victims)) = self.allocator.ia.decide_and_realloc_victim(
                        is,
                        next_use.unwrap_or(Index::inf()),
                        mapping_handler!(self, self.allocator.offset),
                    ) {
                        let objects = victims
                            .iter()
                            .map(|victim| {
                                self.allocator
                                    .objects_at
                                    .get_backward(victim)
                                    .unwrap()
                                    .clone()
                            })
                            .collect::<Vec<_>>();
                        let sizes = victims
                            .iter()
                            .map(|victim| self.allocator.mapping[*victim].1)
                            .collect::<Vec<_>>();

                        let this_device = self.device();

                        let after_ejection = objects
                            .iter()
                            .copied()
                            .zip(victims.iter().copied())
                            .map(|(obj, victim_addr)| {
                                Continuation::on_allocator::<_, D>(
                                    this_device,
                                    move |handle: &mut Handle<'_, '_, '_, '_, '_, P, Rt, D>| {
                                        handle.allocator.objects_at.remove_forward(&obj);
                                        handle.machine.deallocate(obj, p(victim_addr));
                                        Ok(())
                                    },
                                )
                            })
                            .collect::<Vec<_>>();

                        let t = t.clone();

                        Response::Continue(
                            Continuation::collect_result(
                                objects
                                    .into_iter()
                                    .zip(victims.into_iter().map(p))
                                    .zip(sizes.into_iter())
                                    .zip(after_ejection.into_iter())
                                    .map(move |(((object, p), size), dc)| {
                                        Continuation::simple_eject(
                                            this_device,
                                            p,
                                            object,
                                            Size::Integral(size),
                                        )
                                        .bind_result(move |_| dc)
                                    }),
                            )
                            .bind_result({
                                let after_all_ejection = Continuation::on_allocator::<_, D>(
                                    this_device,
                                    move |handle: &mut Handle<'_, '_, '_, '_, '_, P, Rt, D>| {
                                        handle.allocator.objects_at.insert(t, addr);
                                        handle.machine.allocate(t, p(addr));
                                        Ok(())
                                    },
                                );
                                move |_| after_all_ejection
                            })
                            .bind_result(move |_| Continuation::return_(Ok(p(addr)))),
                        )
                    } else {
                        Response::Complete(Err(Error::VertexInputsOutputsNotAccommodated(None)))
                    }
                }
            }
        }
    }

    fn deallocate(&mut self, t: &ObjectId) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        self.tick();

        // fixme
        println!("GPU deallocating {:?}", t);

        let addr_id = self
            .allocator
            .objects_at
            .remove_forward(t)
            .unwrap_or_else(|| panic!("{:?} not found on {:?}", t, self.device()));
        self.allocator
            .ia
            .deallocate(addr_id, mapping_handler!(self, self.allocator.offset));

        self.machine.deallocate(*t, p(addr_id));

        Response::Complete(Ok(()))
    }

    fn completeness(&mut self, object: ObjectId) -> Completeness {
        self.allocator
            .objects_at
            .get_forward(&object)
            .map_or_else(Completeness::plain_zero, |_| Completeness::plain_one())
    }

    fn device(&self) -> Device {
        self.machine.device()
    }

    fn reuse(&mut self, new_object: ObjectId, old_object: ObjectId) {
        // fixme
        println!("GPU reuse {:?} from {:?}", new_object, old_object);

        let addr_id = self
            .allocator
            .objects_at
            .remove_forward(&old_object)
            .unwrap_or_else(|| panic!("{:?} not found on {:?}", old_object, self.device()));

        let next_use = self.aux.query_next_use(new_object, self.device());
        self.update_next_use(addr_id, next_use);

        self.allocator
            .objects_at
            .insert_checked(new_object, addr_id);
    }

    fn claim(
        &mut self,
        t: &ObjectId,
        size: Size,
        from: Device,
    ) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        // fixme
        println!("GPU claim {:?} from {:?}", t, from);

        if self.allocator.objects_at.get_forward(t).is_some() {
            return Response::Complete(Ok(()));
        }

        let this_device = self.device();
        let object = t.clone();

        let resp = self.allocate(size, t);
        resp.bind_result(move |p| Continuation::simple_provide(this_device, p, from, object))
    }

    fn typeid(&self) -> typeid::ConstTypeId {
        typeid::ConstTypeId::of::<Self>()
    }
}

pub struct Realizer<'a, 'm, 's, 'au, 'i, 'f, P, Rt: RuntimeType, D: DeviceMarker> {
    allocator: &'a mut SlabAllocator<'f, P, Rt, D>,
    machine: realization::MachineHandle<'m, 's, P>,
    aux: &'au AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, 'f, P: UsizeId + 'static, Rt: RuntimeType, D: DeviceMarker>
    AllocatorRealizer<'s, ObjectId, P, Rt> for Realizer<'a, 'm, 's, 'au, 'i, 'f, P, Rt, D>
{
    fn allocate(&mut self, t: &ObjectId, pointer: &P) {
        // fixme
        println!(
            "{:?} allocated at {:?} on {:?}",
            t,
            pointer,
            self.machine.device()
        );

        let (addr, size) = self.allocator.mapping[p_inv(*pointer)];
        let vn = self.aux.obj_info().typ(*t).with_normalized_p();
        let rv = ResidentalValue::new(Value::new(*t, self.machine.device(), vn), *pointer);
        self.machine.gpu_allocate(
            AllocMethod::Offset(addr.get(), size.into()),
            rv,
        );
    }

    fn deallocate(&mut self, t: &ObjectId, pointer: &P) {
        // fixme
        println!(
            "{:?} deallocated at {:?} on {:?}",
            t,
            pointer,
            self.machine.device()
        );

        let vn = self.aux.obj_info().typ(*t).with_normalized_p();
        let rv = ResidentalValue::new(Value::new(*t, self.machine.device(), vn), *pointer);
        self.machine.gpu_deallocate(&rv);
    }

    fn transfer(
        &mut self,
        t: &ObjectId,
        from_pointer: &P,
        to_device: Device,
        to_pointer: &P,
    ) -> realization::RealizationResponse<'s, ObjectId, P, Result<(), Error<'s>>, Rt> {
        Response::Continue(Continuation::transfer_object(
            self.machine.device(),
            *from_pointer,
            to_device,
            *to_pointer,
            *t,
        ))
    }
}

impl<'s, P: UsizeId + 'static, Rt: RuntimeType> Allocator<'s, ObjectId, P, Rt>
    for SlabAllocator<'s, P, Rt, Gpu>
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
        p(self.mapping.push((Addr(0), IntegralSize(0))))
    }

    fn inner<'t>(&'t mut self) -> Option<&'t mut dyn Allocator<'s, ObjectId, P, Rt>> {
        None
    }
}

impl<'s, P: UsizeId + 'static, Rt: RuntimeType> Allocator<'s, ObjectId, P, Rt>
    for SlabAllocator<'s, P, Rt, Cpu>
{
    fn handle<'a, 'b, 'c, 'd, 'i>(
        &'a mut self,
        _machine: planning::MachineHandle<'b, 's, ObjectId, P>,
        _aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorHandle<'s, ObjectId, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd,
    {
        unimplemented!()
    }

    fn realizer<'a, 'b, 'c, 'd, 'i>(
        &'a mut self,
        _machine: realization::MachineHandle<'b, 's, P>,
        _aux: &'c mut AuxiliaryInfo<'i, Rt>,
    ) -> Box<dyn AllocatorRealizer<'s, ObjectId, P, Rt> + 'd>
    where
        'a: 'd,
        'b: 'd,
        'c: 'd,
        'i: 'd,
    {
        unimplemented!()
    }

    fn allcate_pointer(&mut self) -> P {
        unimplemented!()
    }

    fn inner<'t>(&'t mut self) -> Option<&'t mut dyn Allocator<'s, ObjectId, P, Rt>> {
        None
    }
}
