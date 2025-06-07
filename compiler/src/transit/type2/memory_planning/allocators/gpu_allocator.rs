use super::{regretting_integral, smithereens_allocator, OffsettedAddrMapping};
use crate::transit::type2::memory_planning::prelude::*;
use planning::machine::*;

pub type Procedure<'f, P, Rt: RuntimeType> = Box<
    dyn for<'a, 's, 'm, 'au, 'i> FnOnce(
            &mut Handle<'a, 'm, 's, 'au, 'i, P, Rt>,
        ) -> PlanningResponse<
            'f,
            's,
            ObjectId,
            P,
            Result<(), Error<'s>>,
            Rt,
        > + 'f,
>;

pub struct GpuAllocator<'f, P, Rt: RuntimeType> {
    mapping: AddrMapping,
    /// Associate each object with its residence.
    /// Don't forget to update this mapping.
    objects_at: Bijection<ObjectId, AddrId>,
    smithereen_offset: u64,
    integral_offset: u64,
    smithereen_capacity: u64,
    integral_capacity: u64,
    smithereen_allocator: smithereens_allocator::Allocator,
    integral_allocator: regretting_integral::Allocator,
    procedures: BTreeMap<allocator::ProcedureId, Procedure<'f, P, Rt>>,
    procedure_id_allocator: IdAllocator<allocator::ProcedureId>,
    _phantom: PhantomData<P>,
}

const MIN_SMITHEREEN_SPACE: u64 = 2u64.pow(26);
const SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD: u64 = 2u64.pow(16);
const LOG_MIN_INTEGRAL_SIZE: u32 = 10;

fn normalize_size(size: Size) -> Size {
    match size {
        Size::Integral(is) => {
            if is.0 < LOG_MIN_INTEGRAL_SIZE {
                Size::Smithereen(SmithereenSize(2u64.pow(is.0)))
            } else {
                Size::Integral(is)
            }
        }
        Size::Smithereen(ss) => {
            if let Ok(is) = IntegralSize::try_from(ss) {
                if is.0 < LOG_MIN_INTEGRAL_SIZE {
                    Size::Smithereen(SmithereenSize(2u64.pow(is.0)))
                } else {
                    Size::Integral(is)
                }
            } else if ss.0 >= SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD {
                Size::Integral(IntegralSize::ceiling(ss))
            } else {
                Size::Smithereen(ss)
            }
        }
    }
}

pub fn collect_integral_sizes(sizes: impl Iterator<Item = Size>) -> Vec<IntegralSize> {
    let mut integral_sizes = BTreeSet::<IntegralSize>::new();
    for size in sizes {
        if let Size::Integral(is) = normalize_size(size) {
            integral_sizes.insert(is);
        }
    }

    integral_sizes.into_iter().collect()
}

impl<'f, P, Rt: RuntimeType> GpuAllocator<'f, P, Rt> {
    pub fn new(total_capacity: u64, smithereen_capacity: u64, lbss: Vec<IntegralSize>) -> Self {
        Self {
            mapping: AddrMapping::new(),
            objects_at: Bijection::new(),
            smithereen_offset: 0,
            integral_offset: smithereen_capacity,
            smithereen_capacity: smithereen_capacity,
            integral_capacity: total_capacity - smithereen_capacity,
            smithereen_allocator: smithereens_allocator::Allocator::new(smithereen_capacity),
            integral_allocator: regretting_integral::Allocator::new(
                total_capacity - smithereen_capacity,
                lbss,
            ),
            procedures: BTreeMap::new(),
            procedure_id_allocator: IdAllocator::new(),
            _phantom: PhantomData,
        }
    }
}

/// All sizes entering the allocator must be normalized first.
pub struct Handle<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType> {
    allocator: &'a mut GpuAllocator<'s, P, Rt>,
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

impl<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType> Handle<'a, 'm, 's, 'au, 'i, P, Rt>
where
    P: UsizeId + 'static,
{
    fn try_update_next_use(&mut self, addr_id: AddrId, next_use: Option<Index>) {
        if let Size::Integral(..) = self.allocator.mapping[addr_id].1 {
            self.allocator.integral_allocator.update_next_use(
                addr_id,
                next_use.unwrap_or(Index::inf()),
                mapping_handler!(self, self.allocator.integral_offset),
            );
        }
    }

    fn tick(&mut self) {
        let pc = self.aux.pc();
        self.allocator.integral_allocator.tick(pc);
    }

    fn add_procedure(
        &mut self,
        procedure: Procedure<'s, P, Rt>,
    ) -> Continuation<'s, planning::Machine<'s, ObjectId, P>, ObjectId, P, Result<(), Error<'s>>, Rt>
    {
        let pid = self.allocator.procedure_id_allocator.alloc();
        self.allocator.procedures.insert(pid, procedure);
        Continuation::procedure(self.machine.device(), pid)
    }
}

impl<'a, 'm, 's, 'au, 'i, P, Rt: RuntimeType> AllocatorHandle<'s, ObjectId, P, Rt>
    for Handle<'a, 'm, 's, 'au, 'i, P, Rt>
where
    P: UsizeId + 'static,
{
    fn access(&mut self, t: &ObjectId) -> Option<P> {
        let updated_next_use = self.aux.mark_use(*t, self.device());

        let addr = self.allocator.objects_at.get_forward(t).copied();

        // fixme
        println!("GPU access {:?} got {:?}", t, addr);

        if let Some(addr) = addr {
            self.try_update_next_use(addr, updated_next_use);
        }

        addr.map(p)
    }

    fn allocate(
        &mut self,
        size: Size,
        t: &ObjectId,
    ) -> allocator::AResp<'s, ObjectId, P, P, Rt> {
        println!("GPU allocating {:?}", t);

        if self.allocator.objects_at.get_forward(t).is_some() {
            panic!("object {:?} already allocated", t);
        }

        self.tick();

        let size = normalize_size(size);
        let resp = match size {
            Size::Smithereen(ss) => {
                if let Some(addr_id) = self.allocator.smithereen_allocator.allocate(
                    ss,
                    &mut OffsettedAddrMapping::new(
                        &mut self.allocator.mapping,
                        self.allocator.smithereen_offset,
                    ),
                ) {
                    self.allocator.objects_at.insert_checked(*t, addr_id);
                    Response::Complete(Ok(p(addr_id)))
                } else {
                    Response::Complete(Err(Error::InsufficientSmithereenSpace))
                }
            }
            Size::Integral(is) => {
                let next_use = self.aux.query_next_use(*t, self.device());

                let r = self.allocator.integral_allocator.allocate(
                    is,
                    next_use.unwrap_or(Index::inf()),
                    mapping_handler!(self, self.allocator.integral_offset),
                );

                if let Some((transfers, addr)) = r {
                    for regretting_integral::Transfer { from, to } in transfers {
                        let object = self
                            .allocator
                            .objects_at
                            .remove_backward(&from)
                            .expect("expected object on device to reallocate it");
                        self.machine
                            .transfer(self.device(), p(to), object, self.device(), p(from));
                        self.allocator.objects_at.insert_checked(object, to);
                    }

                    self.allocator.objects_at.insert_checked(*t, addr);
                    Response::Complete(Ok(p(addr)))
                } else {
                    if let Some((addr, victims)) =
                        self.allocator.integral_allocator.decide_and_realloc_victim(
                            is,
                            next_use.unwrap_or(Index::inf()),
                            mapping_handler!(self, self.allocator.integral_offset),
                        )
                    {
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
                            .map(|obj| {
                                self.add_procedure(Box::new(move |handle| {
                                    handle.allocator.objects_at.remove_forward(&obj);
                                    Response::Complete(Ok(()))
                                }))
                            })
                            .collect::<Vec<_>>();

                        let t = t.clone();

                        let after_all_ejection = self.add_procedure(Box::new(move |handle| {
                            handle.allocator.objects_at.insert(t, addr);
                            Response::Complete(Ok(()))
                        }));

                        Response::Continue(
                            Continuation::collect_result(
                                objects
                                    .into_iter()
                                    .zip(victims.into_iter().map(p))
                                    .zip(sizes.into_iter())
                                    .zip(after_ejection.into_iter())
                                    .map(move |(((object, p), size), dc)| {
                                        Continuation::simple_eject(this_device, p, object, size)
                                            .bind_result(move |_| dc)
                                    }),
                            )
                            .bind_result(move |_| after_all_ejection)
                            .bind_result(move |_| Continuation::return_(Ok(p(addr)))),
                        )
                    } else {
                        Response::Complete(Err(Error::VertexInputsOutputsNotAccommodated(None)))
                    }
                }
            }
        };

        let this_device = self.device();
        let t = *t;
        resp.bind_result(move |p| {
            Continuation::new(
                move |_allocators: &mut AllocatorCollection<ObjectId, P, Rt>,
                      machine: &mut Machine<ObjectId, P>,
                      _aux: &mut AuxiliaryInfo<Rt>| {
                    machine.handle(this_device).allocate(t, p);
                    Ok(p)
                },
            )
        })
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
        match self.allocator.mapping[addr_id].1 {
            Size::Integral(_) => self.allocator.integral_allocator.deallocate(
                addr_id,
                mapping_handler!(self, self.allocator.integral_offset),
            ),
            Size::Smithereen(_) => self.allocator.smithereen_allocator.deallocate(
                addr_id,
                mapping_handler!(self, self.allocator.smithereen_offset),
            ),
        };

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
        self.try_update_next_use(addr_id, next_use);

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

    fn evoke(
        &mut self,
        procedure: allocator::ProcedureId,
    ) -> allocator::AResp<'s, ObjectId, P, (), Rt> {
        let p = self
            .allocator
            .procedures
            .remove(&procedure)
            .expect("expected procedure to be present");
        p(self)
    }
}

pub struct Realizer<'a, 'm, 's, 'au, 'i, 'f, P, Rt: RuntimeType> {
    allocator: &'a mut GpuAllocator<'f, P, Rt>,
    machine: realization::MachineHandle<'m, 's, P>,
    aux: &'au AuxiliaryInfo<'i, Rt>,
}

impl<'a, 'm, 's, 'au, 'i, 'f, P: UsizeId + 'static, Rt: RuntimeType>
    AllocatorRealizer<'s, ObjectId, P, Rt> for Realizer<'a, 'm, 's, 'au, 'i, 'f, P, Rt>
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
        self.machine.gpu_allocate(addr.get().into(), size, rv);
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
    for GpuAllocator<'s, P, Rt>
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
            'i: 'd {
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
}
