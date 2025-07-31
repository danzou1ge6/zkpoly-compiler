use crate::transit::type2::memory_planning::prelude::*;

#[derive(Debug, Clone)]
pub struct Machine<'s, T, P> {
    pub(super) ops: OperationSeq<'s, T, P>,
}

fn values_for_transfer<Rt: RuntimeType>(
    to_device: Device,
    to_object: ObjectId,
    from_device: Device,
    from_object: ObjectId,
    slice: Option<Slice>,
    obj_info: &object_info::Info<Rt>,
) -> (Value, Value) {
    let (vn1, vn2) = if let Some(slice) = slice {
        let (o2_deg, _) = obj_info.typ(from_object).unwrap_poly();

        assert!(slice.len() <= o2_deg as u64);
        if !(slice.begin() <= o2_deg as u64) {
            panic!("assertion failed for sliced transfer from {:?} to {:?}: from_object has degree {}, got slice {:?}", from_device, to_object, o2_deg, slice);
        }
        assert!(obj_info.typ(to_object).unwrap_poly().0 == slice.len() as usize);

        (
            ValueNode::plain_scalar_array(slice.len() as usize),
            ValueNode::ScalarArray {
                len: o2_deg,
                meta: slice,
            },
        )
    } else {
        assert!(obj_info.typ(from_object) == obj_info.typ(to_object));

        (
            obj_info.typ(to_object).with_normalized_p(),
            obj_info.typ(from_object).with_normalized_p(),
        )
    };

    (
        Value::new(to_object, to_device, vn1),
        Value::new(from_object, from_device, vn2),
    )
}

impl<'s, T, P> Machine<'s, T, P> {
    pub fn new() -> Self {
        Machine {
            ops: OperationSeq::empty(),
        }
    }

    pub fn emit(&mut self, op: Operation<'s, T, P>) {
        self.ops.emit(op);
    }

    pub fn handle<'m>(&'m mut self, device: Device) -> MachineHandle<'m, 's, T, P> {
        MachineHandle { m: self, device }
    }

    pub fn transfer(
        &mut self,
        to_device: Device,
        to_pointer: P,
        t: T,
        from_device: Device,
        from_pointer: P,
    ) {
        self.ops.emit(Operation::Transfer(
            to_device,
            to_pointer,
            t,
            from_device,
            from_pointer,
        ));
    }

    pub fn eject(&mut self, to_device: Device, t: T, from_device: Device, from_pointer: P) {
        self.ops
            .emit(Operation::Eject(to_device, t, from_device, from_pointer))
    }

    pub fn reclaim(&mut self, to_device: Device, to_pointer: P, t: T, from_device: Device) {
        self.ops
            .emit(Operation::Reclaim(to_device, to_pointer, t, from_device))
    }

    pub fn transfer_object(&mut self, rv_to: ResidentalValue<P>, rv_from: ResidentalValue<P>) {
        self.ops.emit(Operation::TransferObject(rv_to, rv_from));
    }

    /// Transfer an object from one (device, pointer) to another.
    /// `from_object` and `to_object` can be the same, but there can't be two objects with same ID on one device.
    pub fn transfer_object_sliced<Rt: RuntimeType>(
        &mut self,
        to_device: Device,
        to_pointer: P,
        to_object: ObjectId,
        from_device: Device,
        from_pointer: P,
        from_object: ObjectId,
        slice: Option<Slice>,
        obj_info: &object_info::Info<Rt>,
    ) {
        let (v_to, v_from) = values_for_transfer(
            to_device,
            to_object,
            from_device,
            from_object,
            slice,
            obj_info,
        );

        self.transfer_object(
            ResidentalValue::new(v_to, to_pointer),
            ResidentalValue::new(v_from, from_pointer),
        )
    }

    pub fn reclaim_object(&mut self, rv_to: ResidentalValue<P>, v_from: Value) {
        self.ops.emit(Operation::ReclaimObject(rv_to, v_from))
    }

    pub fn reclaim_object_sliced<Rt: RuntimeType>(
        &mut self,
        to_device: Device,
        to_pointer: P,
        to_object: ObjectId,
        from_device: Device,
        from_object: ObjectId,
        slice: Option<Slice>,
        obj_info: &object_info::Info<Rt>,
    ) {
        let (v_to, v_from) = values_for_transfer(
            to_device,
            to_object,
            from_device,
            from_object,
            slice,
            obj_info,
        );

        self.reclaim_object(ResidentalValue::new(v_to, to_pointer), v_from)
    }

    pub fn eject_object_sliced<Rt: RuntimeType>(
        &mut self,
        to_device: Device,
        to_object: ObjectId,
        from_device: Device,
        from_pointer: P,
        from_object: ObjectId,
        slice: Option<Slice>,
        obj_info: &object_info::Info<Rt>,
    ) {
        let (v_to, v_from) = values_for_transfer(
            to_device,
            to_object,
            from_device,
            from_object,
            slice,
            obj_info,
        );

        self.ops.emit(Operation::EjectObject(
            v_to,
            ResidentalValue::new(v_from, from_pointer),
        ))
    }
}

pub struct MachineHandle<'m, 's, T, P> {
    m: &'m mut Machine<'s, T, P>,
    device: Device,
}

impl<'m, 's, T, P> MachineHandle<'m, 's, T, P> {
    /// Emit a [`Operation::Allocate`], for similar purpose as `deallocate`.
    pub fn allocate(&mut self, t: T, pointer: P)
    where
        T: std::fmt::Debug,
        P: std::fmt::Debug,
    {
        self.m.emit(Operation::Allocate(t, self.device, pointer))
    }

    /// Emit a [`Operation::Deallocate`], which is used to notify realizer of the allocator to emit
    /// currespounding memory deallocation instructions, if needed.
    pub fn deallocate(&mut self, t: T, pointer: P) {
        self.m.emit(Operation::Deallocate(t, self.device, pointer))
    }

    pub fn transfer(
        &mut self,
        to_device: Device,
        to_pointer: P,
        t: T,
        from_device: Device,
        from_pointer: P,
    ) {
        self.m
            .transfer(to_device, to_pointer, t, from_device, from_pointer)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl<'s, P, Rt: RuntimeType>
    Continuation<'s, Machine<'s, ObjectId, P>, ObjectId, P, Result<(), super::super::Error<'s>>, Rt>
where
    P: 'static,
{
    /// Creates a simple continuation that let some device receive ejection of whole token `t` from `from_device`.
    /// `from_device` must be planning.
    ///
    /// It does not modify state of `from_device`
    pub fn simple_eject(
        from_device: Device,
        from_pointer: P,
        object: ObjectId,
        size: Size,
    ) -> Self {
        let f = move |allocators: &mut AllocatorCollection<'_, 's, ObjectId, P, Rt>,
                      machine: &mut Machine<'s, ObjectId, P>,
                      aux: &mut AuxiliaryInfo<Rt>| {
            assert!(aux.is_planning(from_device));

            // Dead object needs no ejection
            if aux.dead(object) {
                return Ok(());
            }

            // If the object is alive elsewhere, no ejection is needed
            let devices = aux
                .planning_devices()
                .chain(aux.unplanned_devices())
                .filter(|d| *d != from_device)
                .collect::<Vec<_>>();
            if let Some(alive_on) = allocators
                .object_available_on(devices.into_iter(), object, machine, aux)
                .pop()
            {
                return Ok(());
            }

            // Otherwise, we eject the object to direct parent device
            let to_device = from_device.parent().unwrap_or_else(|| {
                panic!("ejecting from a device {:?} with no parent", from_device)
            });

            if !aux.is_unplanned(to_device) {
                panic!(
                    "can only eject to an unplanned device, but {:?} is not",
                    to_device
                );
            }

            let c =
                Continuation::simple_receive(from_device, from_pointer, to_device, size, object);
            allocators.apply_continuation(c, machine, aux)
        };

        Continuation::new(f)
    }

    /// Creates a simple continuation that let some device provide a copy of `object` for `to_device`, potentially sliced.
    ///
    /// `from_device` must be planning or unplanned,
    /// `to_device` must be planning or unplanned.
    pub fn provide_object_sliced(
        from_device: Device,
        from_object: ObjectId,
        to_device: Device,
        to_object: ObjectId,
        to_pointer: P,
        slice: Option<Slice>,
    ) -> Self {
        let f = move |allocators: &mut AllocatorCollection<'_, 's, ObjectId, P, Rt>,
                      machine: &mut Machine<'s, ObjectId, P>,
                      aux: &mut AuxiliaryInfo<Rt>| {
            if slice.is_some() && from_device == Device::Disk {
                panic!("slicing from disk is not supported");
            }

            if aux.is_planning(to_device) {
                if aux.is_planning(from_device) {
                    let from_pointer = allocators
                        .handle(from_device, machine, aux)
                        .access(&from_object)
                        .expect("object not found");
                    machine.transfer_object_sliced(
                        to_device,
                        to_pointer,
                        to_object,
                        from_device,
                        from_pointer,
                        from_object,
                        slice,
                        aux.obj_info(),
                    );
                } else if aux.is_unplanned(from_device) {
                    machine.reclaim_object_sliced(
                        to_device,
                        to_pointer,
                        to_object,
                        from_device,
                        from_object,
                        slice,
                        aux.obj_info(),
                    );
                } else {
                    panic!(
                        "from_device {:?} is neither planning nor unplanned",
                        from_device
                    )
                }
            } else if aux.is_unplanned(to_device) {
                if aux.is_planning(from_device) {
                    let from_pointer = allocators
                        .handle(from_device, machine, aux)
                        .access(&from_object)
                        .expect("object not found");
                    machine.eject_object_sliced(
                        to_device,
                        to_object,
                        from_device,
                        from_pointer,
                        from_object,
                        slice,
                        aux.obj_info(),
                    );
                } else if aux.is_unplanned(from_device) {
                    // do nothing
                } else {
                    panic!(
                        "from_device {:?} is neither planning nor unplanned",
                        from_device
                    )
                }
            } else {
                panic!(
                    "to_device {:?} is neither planning nor unplanned",
                    to_device
                )
            }

            Ok(())
        };

        Continuation::new(f)
    }
}

impl<'s, T, P, R, Rt: RuntimeType>
    Continuation<'s, Machine<'s, T, P>, T, P, Result<R, super::super::Error<'s>>, Rt>
where
    T: Clone + 'static,
    P: 'static,
{
    pub fn on_allocator<H, D: DeviceMarker>(
        device: Device,
        f: impl FnOnce(&mut H) -> Result<R, super::super::Error<'s>> + 'static,
    ) -> Self
    where
        H: AllocatorHandle<'s, T, P, Rt>,
    {
        let f = move |allocators: &mut AllocatorCollection<'_, 's, T, P, Rt>,
                      machine: &mut Machine<'s, T, P>,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let mut allocator = allocators.get(device);

            while allocator.handle(machine.handle(device), aux).typeid()
                != typeid::ConstTypeId::of::<H>()
            {
                if let Some(inner) = allocator.inner() {
                    allocator = inner;
                } else {
                    panic!("cannot cast to {:?}", typeid::ConstTypeId::of::<H>());
                }
            }

            let handle = allocator.handle(machine.handle(device), aux);
            // safety: we have checked that the handle is of correct type,
            let handle = unsafe {
                let raw = Box::into_raw(handle);
                &mut *(raw as *mut H)
            };

            f(handle)
        };

        Continuation::new(f)
    }
}

impl<'s, T, P, Rt: RuntimeType>
    Continuation<'s, Machine<'s, T, P>, T, P, Result<(), super::super::Error<'s>>, Rt>
where
    T: Clone + 'static,
    P: 'static,
{
    /// Creates a simple continuation that instructs the target `to_device` to receive rejected
    /// token `t` with size `size`.
    /// The receiving device simply allocates space for the whole token, if the token is not present,
    /// then emit a transfer or eject opertion.
    ///
    /// `from_device` must be planning or planned, that is , its pointer is meaningful.
    /// `to_device` must be planning or unplanned.
    pub fn simple_receive(
        from_device: Device,
        from_pointer: P,
        to_device: Device,
        size: Size,
        t: T,
    ) -> Self
    where
        T: std::fmt::Debug,
    {
        let f = move |allocators: &mut AllocatorCollection<'_, 's, T, P, Rt>,
                      machine: &mut Machine<'s, T, P>,
                      aux: &mut AuxiliaryInfo<Rt>| {
            assert!(aux.is_planning(from_device) || aux.is_planned(from_device));

            if allocators
                .handle(to_device, machine, aux)
                .access(&t)
                .is_none()
            {
                let resp = allocators
                    .handle(to_device, machine, aux)
                    .allocate(size, &t);
                allocators.run(resp, machine, aux)?;
                let to_pointer = allocators
                    .handle(to_device, machine, aux)
                    .access(&t)
                    .expect("i just allocated this object");

                if aux.is_planning(to_device) {
                    machine.transfer(to_device, to_pointer, t, from_device, from_pointer);
                } else if aux.is_unplanned(to_device) {
                    machine.eject(to_device, t, from_device, from_pointer);
                } else {
                    panic!("to_device {:?} must be planning or unplanned", to_device);
                }
            }
            Ok(())
        };

        Continuation::new(f)
    }

    /// Creates a simple continuation that instructs the target `from_device` to provide a token `t`.
    /// The token must be present in the target device.
    /// Transfer always happens.
    ///
    /// `to_device` can be planned, planning or unplanned. In the last case, `to_pointer` is ignored.
    /// `from_device` must be planning or unplanned.
    pub fn simple_provide(to_device: Device, to_pointer: P, from_device: Device, t: T) -> Self
    where
        T: std::fmt::Debug,
    {
        let f = move |allocators: &mut AllocatorCollection<'_, 's, T, P, Rt>,
                      machine: &mut Machine<'s, T, P>,
                      aux: &mut AuxiliaryInfo<Rt>| {
            if aux.is_planning(from_device) {
                let from_pointer = allocators
                    .handle(from_device, machine, aux)
                    .access(&t)
                    .expect("token not found");

                // - planning -> planning/planned, use transfer
                // - planning -> unplanned, use eject
                if aux.is_planning(to_device) || aux.is_planned(to_device) {
                    machine.transfer(to_device, to_pointer, t, from_device, from_pointer);
                } else {
                    assert!(aux.is_unplanned(to_device));
                    machine.eject(to_device, t, from_device, from_pointer)
                }
            } else {
                // - unplanned -> unplanned, do nothing
                // - unplanned -> planning/planned, use reclaim
                assert!(aux.is_unplanned(from_device));
                if aux.is_planning(to_device) | aux.is_planned(to_device) {
                    machine.reclaim(to_device, to_pointer, t, from_device);
                }
            }
            Ok(())
        };

        Continuation::new(f)
    }

    pub fn simple_send(
        to_device: Device,
        from_device: Device,
        from_pointer: P,
        t: T,
        size: Size,
    ) -> Self
    where
        T: std::fmt::Debug,
    {
        let f = move |allocators: &mut AllocatorCollection<'_, 's, T, P, Rt>,
                      machine: &mut Machine<'s, T, P>,
                      aux: &mut AuxiliaryInfo<Rt>| {
            if allocators
                .handle(to_device, machine, aux)
                .access(&t)
                .is_some()
            {
                Ok(())
            } else {
                let resp = allocators
                    .handle(to_device, machine, aux)
                    .allocate(size, &t);
                resp.commit(allocators, machine, aux)?;
                let to_pointer = allocators
                    .handle(to_device, machine, aux)
                    .access(&t)
                    .expect("i just allocated this object");

                {
                    if aux.is_planning(from_device) {
                        if aux.is_planning(to_device) {
                            machine.transfer(to_device, to_pointer, t, from_device, from_pointer);
                        } else if aux.is_unplanned(to_device) {
                            machine.eject(to_device, t, from_device, from_pointer);
                        }
                    } else if aux.is_unplanned(from_device) {
                        if aux.is_planning(to_device) {
                            machine.reclaim(to_device, to_pointer, t, from_device);
                        } else if aux.is_unplanned(to_device) {
                            // do nothing
                        }
                    }

                    Ok(())
                }
            }
        };

        Continuation::new(f)
    }
}

pub type PlanningResponse<'f, 's, T, P, R, Rt: RuntimeType> =
    Response<'f, Machine<'s, T, P>, T, P, R, Rt>;
