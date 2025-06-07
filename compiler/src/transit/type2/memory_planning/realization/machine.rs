use crate::transit::type2::memory_planning::prelude::*;
use type3::{Instruction, InstructionNode};

/// A register's status progresses from [`Undefined`] to [`Defined`] to [`Freed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterStatus {
    /// The machine is prepared for such a register, but its data is not defined by some instruction
    Undefined,
    /// The register's pointing-to data is defined
    Defined,
    /// The register itself is freed (not its pointer)
    Freed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterValue<P> {
    /// A register itself pointing to some object
    Single(ResidentalValue<P>, RegisterStatus),
    /// A register containing multiple single registers, each having different values.
    /// We don't care about these children registers' values, and it must be defined.
    Tuple,
}

impl<P> RegisterValue<P> {
    pub fn unwrap_single(&self) -> (&ResidentalValue<P>, &RegisterStatus) {
        match self {
            RegisterValue::Single(rv, status) => (rv, status),
            RegisterValue::Tuple => panic!("called unwrap_single on RegisterValue::Tuple"),
        }
    }

    pub fn unwrap_single_mut(&mut self) -> (&mut ResidentalValue<P>, &mut RegisterStatus) {
        match self {
            RegisterValue::Single(rv, status) => (rv, status),
            RegisterValue::Tuple => panic!("called unwrap_single on RegisterValue::Tuple"),
        }
    }

    pub fn is_single_and(&self, f: impl Fn(&ResidentalValue<P>, RegisterStatus) -> bool) -> bool {
        match self {
            RegisterValue::Single(rv, status) => f(rv, *status),
            RegisterValue::Tuple => false,
        }
    }

    pub fn typ(&self) -> type3::typ::Typ {
        match self {
            RegisterValue::Single(rv, _) => rv.node().clone(),
            RegisterValue::Tuple => type3::typ::Typ::Tuple,
        }
    }

    pub fn device(&self) -> Device {
        match self {
            RegisterValue::Single(rv, _) => rv.device(),
            RegisterValue::Tuple => Device::Stack,
        }
    }

    pub fn memory_block(&self) -> Option<MemoryBlock<P>>
    where
        P: Clone,
    {
        match self {
            RegisterValue::Single(rv, _) => Some(MemoryBlock(
                rv.device(),
                rv.pointer().clone(),
                rv.object_id(),
            )),
            RegisterValue::Tuple => None,
        }
    }
}

/// Keep track of what registers point to
pub struct RegBooking<P> {
    /// [`RegisterId`]'s are allocated from this mapping.
    values: Heap<RegisterId, RegisterValue<P>>,
    /// Mapping from each (object, device) pair to register.
    p2reg: BTreeMap<(P, Device), BTreeSet<RegisterId>>,
}

impl<P> RegBooking<P> {
    pub fn empty() -> Self {
        Self {
            values: Heap::new(),
            p2reg: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryBlock<P>(Device, P, ObjectId);

impl<P> MemoryBlock<P> {
    pub fn with_object_id(self, object: ObjectId) -> Self {
        Self(self.0, self.1, object)
    }
}

impl<P> RegBooking<P>
where
    P: UsizeId,
{
    pub fn new_reg(&mut self, rv: ResidentalValue<P>) -> RegisterId {
        let reg = self
            .values
            .push(RegisterValue::Single(rv.clone(), RegisterStatus::Undefined));
        self.p2reg
            .entry((*rv.pointer(), rv.device()))
            .or_default()
            .insert(reg);
        reg
    }

    pub fn new_tuple_reg(&mut self) -> RegisterId {
        self.values.push(RegisterValue::Tuple)
    }

    /// Look for a register that has the designated [`ResidentalValue`] and has the desired status.
    pub fn lookup_reg(
        &self,
        rv: &ResidentalValue<P>,
        status: RegisterStatus,
    ) -> Option<RegisterId> {
        self.p2reg
            .get(&(*rv.pointer(), rv.device()))
            .map(|r| {
                r.iter().copied().find(|reg| {
                    self.values[*reg].is_single_and(|reg_rv, reg_s| reg_rv == rv && reg_s == status)
                })
            })
            .flatten()
    }

    /// Look for a register that points to `p` on `device`, and is defined or defined as desired.
    pub fn regs_pointing_to<'a>(
        &'a self,
        object: ObjectId,
        p: P,
        device: Device,
        status: RegisterStatus,
    ) -> impl Iterator<Item = RegisterId> + 'a {
        self.p2reg
            .get(&(p, device))
            .map(|r| r.iter().copied())
            .into_iter()
            .flatten()
            .filter(move |r| {
                self.values[*r].is_single_and(|reg_rv, reg_s| {
                    reg_rv.object_id() == object
                        && reg_rv.pointer() == &p
                        && reg_rv.device() == device
                        && reg_s == status
                })
            })
    }

    /// Get the value of a register, assuming it is single.
    pub fn reg_value(&self, reg: RegisterId) -> (&ResidentalValue<P>, &RegisterStatus) {
        self.values[reg].unwrap_single()
    }

    /// Mark a register as deifned, ignoring tuple registers.
    pub fn define(&mut self, reg: RegisterId) {
        let (_, status) = match &mut self.values[reg] {
            RegisterValue::Single(rv, status) => (rv, status),
            RegisterValue::Tuple => return,
        };

        if *status != RegisterStatus::Undefined {
            panic!(
                "register {:?} has status {:?}, can't be defined again",
                reg, *status
            );
        }

        *status = RegisterStatus::Defined;
    }

    /// Mark a register as freed, ignoring tuple registers.
    pub fn free_reg(&mut self, reg: RegisterId) {
        match &mut self.values[reg] {
            RegisterValue::Single(rv, status) => {
                use RegisterStatus::*;
                match *status {
                    Defined => {}
                    Undefined => {
                        if !rv.node().is_gpu_buffer() {
                            panic!(
                                "can only free undefined register {:?} when its value a buffer",
                                reg
                            );
                        }
                    }
                    Freed => panic!("register {:?} already freed, can't be freed", reg),
                }
                *status = RegisterStatus::Freed;
            }
            RegisterValue::Tuple => {}
        }
    }

    pub fn export_reg_types(&self) -> RoHeap<RegisterId, type3::typ::Typ> {
        self.values
            .map_by_ref(&mut |_, reg_v| reg_v.typ())
            .freeze()
            .0
    }

    pub fn export_reg_devices(&self) -> BTreeMap<RegisterId, Device> {
        self.values
            .iter_with_id()
            .map(|(r, reg_v)| (r, reg_v.device()))
            .collect()
    }

    pub fn export_memory_blocks(&self) -> BTreeMap<RegisterId, MemoryBlock<P>> {
        self.values
            .iter_with_id()
            .filter_map(|(r, reg_v)| Some((r, reg_v.memory_block()?)))
            .collect()
    }

    pub fn reg_id_allocator(self) -> IdAllocator<RegisterId> {
        self.values.freeze().1
    }
}

pub struct Machine<'s, P> {
    pub(super) instructions: Vec<type3::Instruction<'s>>,
    pub(super) reg_booking: RegBooking<P>,
}

impl<'s, P> Machine<'s, P> {
    pub fn empty() -> Self {
        Self {
            instructions: Vec::new(),
            reg_booking: RegBooking::empty(),
        }
    }

    pub fn handle<'m>(&'m mut self, device: Device) -> MachineHandle<'m, 's, P> {
        MachineHandle {
            machine: self,
            device,
        }
    }
}

impl<'s, P> Machine<'s, P>
where
    P: UsizeId,
{
    /// Emit an instruction.
    /// All defined registers in the instruction must be defined for the first time.
    /// Allocations are not considered definitions.
    pub fn emit(&mut self, inst: type3::Instruction<'s>) {
        if !inst.node.is_allloc() {
            inst.defs().for_each(|r| self.reg_booking.define(r));
        }
        if let type3::InstructionNode::Type2 { temp, .. } = &inst.node {
            temp.iter().for_each(|r| self.reg_booking.define(*r))
        }
        self.instructions.push(inst);
    }

    /// Look for a register that has the designated [`ResidentalValue`] and is defined.
    ///
    /// If no such register is found, this function attempt to make one by slicing an existing one.
    /// If no register pointing to the [`ResidentalValue`] is found, this function panics.
    pub fn defined_reg_for(&mut self, rv: &ResidentalValue<P>) -> RegisterId {
        if let Some(reg) = self.reg_booking.lookup_reg(rv, RegisterStatus::Defined) {
            return reg;
        }

        let reg = self
            .reg_booking
            .regs_pointing_to(
                rv.object_id(),
                *rv.pointer(),
                rv.device(),
                RegisterStatus::Defined,
            )
            .next()
            .unwrap_or_else(|| {
                panic!(
                    "cannot find a defined register even just pointing to {:?}",
                    &rv
                )
            });

        let (reg_rv, _) = self.reg_booking.reg_value(reg);
        // Since a register pointing to (p, device) is defined, but it does not have the type we desire,
        // it must of different slice
        let (deg, slice) = rv.node().unwrap_poly();
        let (reg_deg, _) = reg_rv.node().unwrap_poly();
        assert!(deg == reg_deg);

        let new_reg = self.reg_booking.new_reg(rv.clone());

        self.emit(Instruction::new_no_src(InstructionNode::SetPolyMeta {
            id: new_reg,
            from: reg,
            offset: slice.begin() as usize,
            len: slice.len() as usize,
        }));

        new_reg
    }

    /// Lookup for an undefined register having exactly the `rv`.
    /// Unlike `defined_reg_for`, this function does not create new register through slicing.
    ///
    /// However, this function does create new register by calling realizer.
    pub fn undefined_reg_for(&mut self, rv: &ResidentalValue<P>) -> RegisterId
    where
        P: UsizeId,
    {
        self.reg_booking
            .lookup_reg(&rv, RegisterStatus::Undefined)
            .unwrap_or_else(|| panic!("no undefined register found for {:?}", rv))
    }

    /// Add a new undefined register to the machine.
    /// The space that the register points to is not necessarily allocated.
    pub fn new_reg(&mut self, rv: ResidentalValue<P>) -> RegisterId {
        self.reg_booking.new_reg(rv)
    }

    /// Create a new tuple register from `regs`, which must all be single registers.
    pub fn new_tuple_reg(&mut self, regs: Vec<RegisterId>) -> RegisterId {
        // check single
        for reg in regs.iter() {
            let _ = self.reg_booking.reg_value(*reg);
        }

        let reg = self.reg_booking.new_tuple_reg();
        self.emit(Instruction::new_no_src(InstructionNode::Tuple {
            id: reg,
            oprands: regs,
        }));
        reg
    }

    /// Free a register
    pub fn free_reg(&mut self, reg: RegisterId) {
        self.reg_booking.free_reg(reg);
        self.emit(Instruction::new_no_src(InstructionNode::StackFree {
            id: reg,
        }));
    }

    pub fn transfer_object(&mut self, rv_to: ResidentalValue<P>, rv_from: ResidentalValue<P>) {
        let reg_to = self.undefined_reg_for(&rv_to);
        let reg_from = self.defined_reg_for(&rv_from);

        self.emit(Instruction::new_no_src(InstructionNode::Transfer {
            id: reg_to,
            from: reg_from,
        }))
    }

    /// Free registers pointing to `pointer`
    fn free_regs_to(&mut self, pointer: P, object: ObjectId, device: Device, except: RegisterId) {
        let regs = self
            .reg_booking
            .regs_pointing_to(object, pointer, device, RegisterStatus::Defined)
            .filter(|r| *r != except)
            .collect::<Vec<_>>();
        regs.into_iter().for_each(|r| self.free_reg(r));
    }
}

pub struct MachineHandle<'m, 's, P> {
    machine: &'m mut Machine<'s, P>,
    device: Device,
}

impl<'m, 's, P> MachineHandle<'m, 's, P>
where
    P: UsizeId,
{
    /// Emit a GpuMalloc instruction at `va`, for object, type and pointer specified in `rv`.
    pub fn gpu_allocate(
        &mut self,
        va: type3::VirtualAddr,
        size: Size,
        rv: ResidentalValue<P>,
    ) -> RegisterId {
        assert!(rv.device() == self.device);

        let reg = self.machine.new_reg(rv);
        self.machine
            .emit(Instruction::new_no_src(InstructionNode::GpuMalloc {
                id: reg,
                addr: va,
                size: size.into(),
            }));
        reg
    }

    pub fn gpu_deallocate(&mut self, rv: &ResidentalValue<P>) {
        assert!(rv.device() == self.device);

        let reg = self.machine.defined_reg_for(rv);
        self.machine
            .emit(Instruction::new_no_src(InstructionNode::GpuFree {
                id: reg,
            }));
        self.machine
            .free_regs_to(*rv.pointer(), rv.object_id(), rv.device(), reg);
    }

    pub fn cpu_allocate(&mut self, size: Size, rv: ResidentalValue<P>) -> RegisterId {
        assert!(rv.device() == self.device);

        let reg = self.machine.new_reg(rv);
        self.machine
            .emit(Instruction::new_no_src(InstructionNode::CpuMalloc {
                id: reg,
                size,
            }));
        reg
    }

    pub fn cpu_deallocate(&mut self, rv: &ResidentalValue<P>) {
        assert!(rv.device() == self.device);

        let reg = self.machine.defined_reg_for(rv);
        self.machine
            .emit(Instruction::new_no_src(InstructionNode::CpuFree {
                id: reg,
            }));
        self.machine
            .free_regs_to(*rv.pointer(), rv.object_id(), rv.device(), reg);
    }

    /// Inform the machine that `pointer` on `device` to `object` is no longer available
    pub fn device(&self) -> Device {
        self.device
    }
}

pub type RealizationResponse<'s, T, P, R, Rt: RuntimeType> =
    Response<'s, Machine<'s, P>, T, P, R, Rt>;

impl<'s, T, P, Rt: RuntimeType>
    continuations::Continuation<'s, Machine<'s, P>, T, P, Result<(), Error<'s>>, Rt>
{
    pub fn transfer_object(
        from_device: Device,
        from_pointer: P,
        to_device: Device,
        to_pointer: P,
        object: ObjectId,
    ) -> Self
    where
        P: UsizeId + 'static,
    {
        let f = move |_allocators: &mut AllocatorCollection<T, P, Rt>,
                      machine: &mut Machine<'s, P>,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let vn = aux.obj_info().typ(object).with_normalized_p();

            let rv_to = ResidentalValue::new(Value::new(object, to_device, vn.clone()), to_pointer);
            let rv_from = ResidentalValue::new(Value::new(object, from_device, vn), from_pointer);

            machine.transfer_object(rv_to, rv_from);
            Ok(())
        };

        continuations::Continuation::new(f)
    }
}
