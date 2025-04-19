use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use zkpoly_common::{
    digraph::internal::SubDigraph,
    heap::{Heap, IdAllocator},
    injection::Injection,
    load_dynamic::Libs,
};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::type3::{typ::PolyMeta, Chunk, DeviceSpecific};

use super::{
    super::type3::{
        typ::Typ, Addr, AddrId, AddrMapping, Device as DeterminedDevice, Instruction,
        InstructionNode, IntegralSize, RegisterId, Size, SmithereenSize,
    },
    object_analysis::AtModifier,
    user_function,
};

use super::object_analysis::{
    self, ObjectId, ObjectUse, ObjectsDef, ObjectsDieAfter, ObjectsDieAfterReversed,
    ObjectsGpuNextUse, Value, ValueNode,
};
use super::{Cg, Device, VertexId};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Instant(pub(super) usize);

mod integral_allocator;
mod smithereens_allocator;

trait AddrMappingHandler {
    fn add(&mut self, addr: Addr, size: Size) -> AddrId;
    fn update(&mut self, id: AddrId, addr: Addr, size: Size);
    fn get(&self, id: AddrId) -> (Addr, Size);
}

struct GpuAddrMappingHandler<'m>(&'m mut AddrMapping, u64);

impl<'m> GpuAddrMappingHandler<'m> {
    pub fn new(addr_mapping: &'m mut AddrMapping, offset: u64) -> Self {
        Self(addr_mapping, offset)
    }
}

impl<'m> AddrMappingHandler for GpuAddrMappingHandler<'m> {
    fn add(&mut self, addr: Addr, size: Size) -> AddrId {
        self.0.push((addr.offset(self.1), size))
    }

    fn update(&mut self, id: AddrId, addr: Addr, size: Size) {
        self.0[id] = (addr.offset(self.1), size);
    }

    fn get(&self, id: AddrId) -> (Addr, Size) {
        let (addr, size) = self.0[id];
        (addr.unoffset(self.1), size)
    }
}

fn collect_integral_sizes<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
    g: &SubDigraph<'_, VertexId, super::Vertex<'s, Rt>>,
    libs: &mut Libs,
) -> Vec<IntegralSize> {
    let mut integral_sizes = BTreeSet::<IntegralSize>::new();
    for vid in g.vertices() {
        let v = g.vertex(vid);
        for &size in v.typ().size().iter() {
            if let Size::Integral(is) = normalize_size(Size::Smithereen(SmithereenSize(size))) {
                integral_sizes.insert(is);
            }
        }
        if let Some((temp_sizes, _)) = cg.temporary_space_needed(vid, libs) {
            for temp_size in temp_sizes {
                if let Size::Integral(is) =
                    normalize_size(Size::Smithereen(SmithereenSize(temp_size)))
                {
                    integral_sizes.insert(is);
                }
            }
        }
    }

    integral_sizes.into_iter().collect()
}
fn divide_integral_smithereens(total: u64, max_integral: IntegralSize) -> (u64, u64) {
    let max_integral = 2u64.pow(max_integral.0);
    let mut integral_space = total / max_integral * max_integral;

    let mut smithereen_space = total - integral_space;
    while smithereen_space < MIN_SMITHEREEN_SPACE {
        smithereen_space += max_integral;
        integral_space -= max_integral;
    }

    (integral_space, smithereen_space)
}

pub type RegisterPlaces = DeviceSpecific<BTreeSet<RegisterId>>;

impl RegisterPlaces {
    pub fn empty() -> Self {
        Self::default()
    }
}

impl RegisterPlaces {
    pub fn get_any(&self) -> Option<(Vec<RegisterId>, DeterminedDevice)> {
        DeterminedDevice::iter()
            .filter_map(|d| {
                if !self.get_device(d).is_empty() {
                    Some((self.get_device(d).iter().copied(), d))
                } else {
                    None
                }
            })
            .next()
            .map(|(it, d)| (it.collect::<Vec<_>>(), d))
    }
}

#[derive(Debug)]
struct Context {
    gpu_reg2addr: Injection<RegisterId, AddrId>,
    gpu_addr_mapping: AddrMapping,
    object_residence: BTreeMap<ObjectId, RegisterPlaces>,
    /// Tuple registers will not be included
    reg_device: BTreeMap<RegisterId, DeterminedDevice>,
    reg2obj: BTreeMap<RegisterId, ObjectId>,
    inplace_obj: BTreeSet<ObjectId>,
}

#[derive(Debug)]
struct ImmutableContext<'a> {
    obj_def: &'a ObjectsDef,
    obj_dies_after: &'a ObjectsDieAfter,
    obj_dies_after_reversed: &'a ObjectsDieAfterReversed,
    vertex_inputs: &'a ObjectUse,
}

impl<'a> ImmutableContext<'a> {
    pub fn obj_size(&self, obj_id: ObjectId) -> Size {
        self.obj_def.sizes[&obj_id]
    }
}

impl Context {
    pub fn add_residence_for_object(
        &mut self,
        obj_id: ObjectId,
        reg_id: RegisterId,
        device: DeterminedDevice,
    ) {
        self.object_residence
            .entry(obj_id)
            .or_insert_with(|| RegisterPlaces::empty())
            .get_device_mut(device)
            .insert(reg_id);
        self.reg_device.insert(reg_id, device);
        self.reg2obj.insert(reg_id, obj_id);
    }

    pub fn remove_residence_in_reg_for_object(
        &mut self,
        obj_id: ObjectId,
        device: DeterminedDevice,
        reg_id: RegisterId,
    ) {
        if !self
            .object_residence
            .get_mut(&obj_id)
            .unwrap()
            .get_device_mut(device)
            .remove(&reg_id)
        {
            panic!(
                "{:?} is not in residence of {:?} on {:?}",
                reg_id, obj_id, device
            )
        }
    }

    pub fn remove_residence_for_object(&mut self, obj_id: ObjectId, device: DeterminedDevice) {
        self.object_residence
            .get_mut(&obj_id)
            .unwrap()
            .get_device_mut(device)
            .clear();
    }

    pub fn pop_residence_of_object(
        &mut self,
        obj_id: ObjectId,
        device: DeterminedDevice,
    ) -> Option<RegisterId> {
        self.object_residence
            .get_mut(&obj_id)
            .unwrap()
            .get_device_mut(device)
            .pop_first()
    }

    pub fn get_residence_of_object(
        &self,
        obj_id: ObjectId,
        device: DeterminedDevice,
    ) -> Option<RegisterId> {
        self.object_residence
            .get(&obj_id)
            .and_then(|places| places.get_device(device).first().copied())
    }

    pub fn is_obj_alive_on(&self, obj_id: ObjectId, device: DeterminedDevice) -> bool {
        self.object_residence
            .get(&obj_id)
            .is_some_and(|places| !places.get_device(device).is_empty())
    }

    pub fn attempt_copy_gpu_addr_id_from(&mut self, from: RegisterId, to: RegisterId) {
        if let Some(addr_id) = self.gpu_reg2addr.get_forward(&from) {
            self.gpu_reg2addr.insert(to, *addr_id);
        }
    }
}

struct Code<'s>(Vec<Instruction<'s>>, Heap<RegisterId, Typ>);

impl<'s> Code<'s> {
    pub fn new() -> Self {
        Self(Vec::new(), Heap::new())
    }

    pub fn alloc_register_id(&mut self, typ: Typ) -> RegisterId {
        self.1.push(typ)
    }

    pub fn emit(&mut self, inst: Instruction<'s>) {
        self.0.push(inst);
    }

    pub fn typ_of(&self, id: RegisterId) -> &Typ {
        &self.1[id]
    }
}

struct GpuAllocator {
    ialloc: integral_allocator::regretting::Allocator,
    salloc: smithereens_allocator::Allocator,
}

fn move_victims(victims: &[AddrId], code: &mut Code, ctx: &mut Context, imctx: &ImmutableContext) {
    victims.iter().for_each(|&victim| {
        let first_reg = *ctx
            .gpu_reg2addr
            .get_backward(&victim)
            .unwrap()
            .next()
            .unwrap();
        let obj_id = ctx.reg2obj[&first_reg];

        // If obj_id is already on CPU, or if obj_id is sliced from some sliced_obj on GPU, we don't need to transfer anything.
        // The latter case is because that sliced_obj dies after obj_id in object analysis.
        if ctx.is_obj_alive_on(obj_id, DeterminedDevice::Cpu) {
            return;
        }
        if imctx
            .vertex_inputs
            .cloned_slice_from(obj_id)
            .is_some_and(|(sliced_obj, _)| ctx.is_obj_alive_on(sliced_obj, DeterminedDevice::Cpu))
        {
            return;
        }

        let normalized_typ = code.typ_of(first_reg).normalized();
        let src_reg = ensure_same_type(
            DeterminedDevice::Gpu,
            obj_id,
            normalized_typ.clone(),
            ctx.gpu_reg2addr
                .get_backward(&victim)
                .unwrap()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter(),
            code,
            ctx,
        );

        let new_reg = code.alloc_register_id(normalized_typ);
        let size = imctx.obj_size(obj_id);
        allocate_cpu(obj_id, new_reg, size, code, ctx);
        code.emit(Instruction::new_no_src(InstructionNode::Transfer {
            id: new_reg,
            from: src_reg,
        }));
    });
}

fn get_gpu_reg_of_normalized_typ(
    addr_id: AddrId,
    ctx: &mut Context,
    code: &mut Code,
) -> RegisterId {
    let first_reg = *ctx
        .gpu_reg2addr
        .get_backward(&addr_id)
        .unwrap()
        .next()
        .unwrap();
    let obj_id = ctx.reg2obj[&first_reg];

    let normalized_typ = code.typ_of(first_reg).normalized();
    let reg = ensure_same_type(
        DeterminedDevice::Gpu,
        obj_id,
        normalized_typ.clone(),
        ctx.gpu_reg2addr
            .get_backward(&addr_id)
            .unwrap()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter(),
        code,
        ctx,
    );

    reg
}

fn allocate_gpu_integral(
    size: IntegralSize,
    next_use: Instant,
    gpu_ialloc: &mut integral_allocator::regretting::Allocator,
    ioffset: u64,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
) -> AddrId {
    let addr = gpu_ialloc.allocate(
        size,
        next_use,
        &mut GpuAddrMappingHandler::new(&mut ctx.gpu_addr_mapping, ioffset),
    );

    if let Some((transfers, addr)) = addr {
        for t in transfers {
            let reg_from = get_gpu_reg_of_normalized_typ(t.from, ctx, code);
            let reg_to = get_gpu_reg_of_normalized_typ(t.to, ctx, code);

            code.emit(Instruction::new_no_src(InstructionNode::Transfer {
                id: reg_to,
                from: reg_from,
            }));
        }
        return addr;
    }

    let (addr, victims) = gpu_ialloc.decide_and_realloc_victim(
        size,
        next_use,
        &mut GpuAddrMappingHandler::new(&mut ctx.gpu_addr_mapping, ioffset),
    );
    move_victims(&victims, code, ctx, imctx);

    addr
}

#[derive(Debug)]
pub struct InsufficientSmithereenSpace;

fn allocate_gpu_smithereen(
    size: SmithereenSize,
    gpu_salloc: &mut smithereens_allocator::Allocator,
    soffset: u64,
    mapping: &mut AddrMapping,
) -> Result<AddrId, InsufficientSmithereenSpace> {
    if let Some(addr) = gpu_salloc.allocate(size, &mut GpuAddrMappingHandler::new(mapping, soffset))
    {
        return Ok(addr);
    }

    Err(InsufficientSmithereenSpace)
}

fn whether_ceil_smithereen_to_integral(size: SmithereenSize) -> bool {
    size.0 >= SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD
}

impl GpuAllocator {
    fn ioffset(&self) -> u64 {
        self.salloc.capacity()
    }

    fn soffset(&self) -> u64 {
        0
    }

    pub fn allocate(
        &mut self,
        size: Size,
        obj_id: ObjectId,
        reg_id: RegisterId,
        next_use: Instant,
        code: &mut Code,
        ctx: &mut Context,
        imctx: &ImmutableContext,
    ) -> Result<(), InsufficientSmithereenSpace> {
        let ioffset = self.ioffset();
        let soffset = self.soffset();

        let addr = match normalize_size(size) {
            Size::Integral(size) => {
                allocate_gpu_integral(size, next_use, &mut self.ialloc, ioffset, code, ctx, imctx)
            }
            Size::Smithereen(size) => {
                allocate_gpu_smithereen(size, &mut self.salloc, soffset, &mut ctx.gpu_addr_mapping)?
            }
        };

        code.emit(Instruction::new_no_src(InstructionNode::GpuMalloc {
            id: reg_id,
            addr,
        }));

        ctx.gpu_reg2addr.insert(reg_id, addr);

        ctx.add_residence_for_object(obj_id, reg_id, DeterminedDevice::Gpu);

        Ok(())
    }

    pub fn deallocate(&mut self, obj_id: ObjectId, code: &mut Code, ctx: &mut Context) {
        let reg_id = ctx
            .pop_residence_of_object(obj_id, DeterminedDevice::Gpu)
            .unwrap();
        let addr = ctx.gpu_reg2addr.get_forward(&reg_id).cloned().unwrap();

        let (_, size) = ctx.gpu_addr_mapping[addr];
        match size {
            Size::Integral(..) => self.ialloc.deallocate(
                addr,
                &mut GpuAddrMappingHandler::new(&mut ctx.gpu_addr_mapping, self.ioffset()),
            ),
            Size::Smithereen(..) => self.salloc.deallocate(
                addr,
                &mut GpuAddrMappingHandler::new(&mut ctx.gpu_addr_mapping, self.soffset()),
            ),
        }

        code.emit(Instruction::new_no_src(InstructionNode::GpuFree {
            id: reg_id,
        }));

        while let Some(reg_id) = ctx.pop_residence_of_object(obj_id, DeterminedDevice::Gpu) {
            code.emit(Instruction::new_no_src(InstructionNode::StackFree {
                id: reg_id,
            }))
        }

        ctx.remove_residence_for_object(obj_id, DeterminedDevice::Gpu);
    }

    pub fn tick(&mut self, now: Instant) {
        self.ialloc.tick(now);
    }

    pub fn update_next_use(
        &mut self,
        obj_id: ObjectId,
        next_use: Instant,
        ctx: &mut Context,
    ) -> bool {
        if let Some(reg_id) = ctx.get_residence_of_object(obj_id, DeterminedDevice::Gpu) {
            let addr = ctx.gpu_reg2addr.get_forward(&reg_id).cloned().unwrap();

            let (_, size) = ctx.gpu_addr_mapping[addr];
            match size {
                Size::Integral(..) => {
                    self.ialloc.update_next_use(
                        addr,
                        next_use,
                        &mut GpuAddrMappingHandler::new(&mut ctx.gpu_addr_mapping, self.ioffset()),
                    );
                    true
                }
                Size::Smithereen(..) => false,
            }
        } else {
            // The object might has been ejected from GPU memory, so the update fails silently
            false
        }
    }
}

fn allocate_cpu(obj_id: ObjectId, id: RegisterId, size: Size, code: &mut Code, ctx: &mut Context) {
    code.emit(Instruction::new_no_src(InstructionNode::CpuMalloc {
        id,
        size,
    }));

    ctx.add_residence_for_object(obj_id, id, DeterminedDevice::Cpu);
}

fn allocate(
    device: DeterminedDevice,
    reg_id: RegisterId,
    obj_id: ObjectId,
    next_use: Instant,
    size: Size,
    gpu_allocator: &mut GpuAllocator,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
) -> Result<(), InsufficientSmithereenSpace> {
    match device {
        DeterminedDevice::Gpu => {
            gpu_allocator.allocate(size, obj_id, reg_id, next_use, code, ctx, imctx)?
        }
        DeterminedDevice::Cpu => allocate_cpu(obj_id, reg_id, size, code, ctx),
        DeterminedDevice::Stack => ctx.add_residence_for_object(obj_id, reg_id, device),
    };
    Ok(())
}

fn deallocate_cpu(obj_id: ObjectId, code: &mut Code, ctx: &mut Context) {
    let reg_id = ctx
        .pop_residence_of_object(obj_id, DeterminedDevice::Cpu)
        .unwrap_or_else(|| panic!("cpu {:?} residence not found", obj_id));

    code.emit(Instruction::new_no_src(InstructionNode::CpuFree {
        id: reg_id,
    }));

    while let Some(reg_id) = ctx.pop_residence_of_object(obj_id, DeterminedDevice::Cpu) {
        code.emit(Instruction::new_no_src(InstructionNode::StackFree {
            id: reg_id,
        }));
    }

    ctx.remove_residence_for_object(obj_id, DeterminedDevice::Cpu);
}

fn deallocate_stack(obj_id: ObjectId, code: &mut Code, ctx: &mut Context) {
    while let Some(reg_id) = ctx.pop_residence_of_object(obj_id, DeterminedDevice::Stack) {
        code.emit(Instruction::new_no_src(InstructionNode::StackFree {
            id: reg_id,
        }));
    }

    ctx.remove_residence_for_object(obj_id, DeterminedDevice::Stack);
}

fn deallocate(
    device: DeterminedDevice,
    obj_id: ObjectId,
    gpu_allocator: &mut GpuAllocator,
    code: &mut Code,
    ctx: &mut Context,
) {
    match device {
        DeterminedDevice::Gpu => gpu_allocator.deallocate(obj_id, code, ctx),
        DeterminedDevice::Cpu => deallocate_cpu(obj_id, code, ctx),
        DeterminedDevice::Stack => deallocate_stack(obj_id, code, ctx),
    }
}

fn ensure_same_type(
    device: DeterminedDevice,
    obj_id: ObjectId,
    typ: Typ,
    mut candidate_regs: impl Iterator<Item = RegisterId> + Clone,
    code: &mut Code,
    ctx: &mut Context,
) -> RegisterId {
    if let Some(reg_id) = candidate_regs.clone().find(|&r| code.typ_of(r) == &typ) {
        reg_id
    } else {
        let (deg, meta) = typ.unwrap_poly();
        let meta = meta.clone();
        let reg_id = candidate_regs.next().unwrap().clone();
        let new_reg = code.alloc_register_id(typ);
        let (slice_offset, slice_len) = meta.offset_and_len(deg as u64);
        code.emit(Instruction::new_no_src(InstructionNode::SetPolyMeta {
            id: new_reg,
            from: reg_id,
            offset: slice_offset as usize,
            len: slice_len as usize,
        }));
        ctx.attempt_copy_gpu_addr_id_from(reg_id, new_reg);
        ctx.add_residence_for_object(obj_id, new_reg, device);
        new_reg
    }
}

// Try to find a register pointing to `obj_id` on `device`, but not necessarily that device.
// Type of the returned register is ensured to be `typ`.
fn attempt_on_device(
    device: DeterminedDevice,
    obj_id: ObjectId,
    typ: Typ,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
) -> RegisterId {
    let candidate_registers = ctx.object_residence[&obj_id]
        .get_device(device)
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    if !candidate_registers.is_empty() {
        ensure_same_type(
            device,
            obj_id,
            typ,
            candidate_registers.into_iter(),
            code,
            ctx,
        )
    } else {
        if let Some((from_reg_candidates, _)) = ctx.object_residence[&obj_id].get_any() {
            ensure_same_type(
                device,
                obj_id,
                typ,
                from_reg_candidates.iter().copied(),
                code,
                ctx,
            )
        } else {
            let (sliced_obj, meta) = imctx.vertex_inputs.cloned_slice_from(obj_id).unwrap();
            let (len, _) = typ.unwrap_poly();
            let sliced_typ = Typ::ScalarArray { len, meta };
            attempt_on_device(device, sliced_obj, sliced_typ, code, ctx, imctx)
        }
    }
}

// Return a register that is ensured to point to `obj_id` on `device`, with type `typ`.
fn ensure_on_device(
    device: DeterminedDevice,
    obj_id: ObjectId,
    typ: Typ,
    next_use: Instant,
    gpu_allocator: &mut GpuAllocator,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
) -> Result<RegisterId, InsufficientSmithereenSpace> {
    // - If the same device has the desired object
    //     - If some register `r` has type same as desired, use it directly
    //     - Otherwise, the only possiblility is different polynomial metadata, so we copy the pointer
    //       and reset metadata
    // - Otherwise, allocate new space and try find register pointing to desired object on another device
    //     - If some regiter `r` is found on any device, transfer whole piece and then ensure correct type
    //       then transfer `r` to a new register
    //     - Otherwise, the object must be sliced from some other object `src`.
    //       In this case, we first attempt to find a register pointing to `src` and ensure its slice is desired,
    //       Then we transfer `src` to a new register.
    let candidate_registers = ctx
        .object_residence
        .get(&obj_id)
        .map(|x| x.get_device(device).iter().cloned())
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if !candidate_registers.is_empty() {
        Ok(ensure_same_type(
            device,
            obj_id,
            typ,
            candidate_registers.into_iter(),
            code,
            ctx,
        ))
    } else {
        let new_reg = code.alloc_register_id(typ.clone());
        let size = imctx.obj_size(obj_id);

        if let Some((from_reg_candidates, from_device)) = ctx
            .object_residence
            .get(&obj_id)
            .map(|x| x.get_any())
            .flatten()
        {
            allocate(
                device,
                new_reg,
                obj_id,
                next_use,
                size,
                gpu_allocator,
                code,
                ctx,
                imctx,
            )?;

            let from_reg = ensure_same_type(
                from_device,
                obj_id,
                typ.normalized(),
                from_reg_candidates.iter().copied(),
                code,
                ctx,
            );

            code.emit(Instruction::new_no_src(InstructionNode::Transfer {
                id: new_reg,
                from: from_reg,
            }));
            Ok(new_reg)
        } else {
            allocate(
                device,
                new_reg,
                obj_id,
                next_use,
                size,
                gpu_allocator,
                code,
                ctx,
                imctx,
            )?;

            let (sliced_obj, meta) = imctx
                .vertex_inputs
                .cloned_slices_reversed
                .get(&obj_id)
                .unwrap_or_else(|| {
                    dbg!(&ctx);
                    panic!(
                        "{:?} is not sliced, but it is not found in any register",
                        obj_id
                    )
                })
                .clone();
            let (len, _) = typ.unwrap_poly();
            let sliced_typ = Typ::ScalarArray { len, meta };
            let from_reg = attempt_on_device(device, sliced_obj, sliced_typ, code, ctx, imctx);

            code.emit(Instruction::new_no_src(InstructionNode::Transfer {
                id: new_reg,
                from: from_reg,
            }));
            Ok(new_reg)
        }
    }
}

fn typ_for_copied(typ: &Typ, original_size: Size) -> (Typ, Size) {
    match typ {
        Typ::ScalarArray { len, meta } => {
            let t = Typ::ScalarArray {
                len: meta.len(*len as u64) as usize,
                meta: PolyMeta::plain(),
            };
            let size = original_size * meta.len(*len as u64) / (*len as u64);
            (t, size)
        }
        _ => (typ.clone(), original_size),
    }
}

fn ensure_copied(
    device: DeterminedDevice,
    vid: VertexId,
    typ: Typ,
    original_obj_id: ObjectId,
    now: Instant,
    gpu_allocator: &mut GpuAllocator,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
) -> Result<RegisterId, InsufficientSmithereenSpace> {
    let no_need_copy = if device == DeterminedDevice::Gpu {
        let (v, modifier) =
            &imctx.obj_dies_after.get_device(DeterminedDevice::Gpu)[&original_obj_id];
        *v == vid && *modifier == AtModifier::After
    } else {
        imctx
            .obj_dies_after
            .get_device(device)
            .get(&original_obj_id)
            .is_some_and(|(v, modifier)| *v == vid && *modifier == AtModifier::After)
    };

    let input_reg = if !no_need_copy {
        // that is, need copy
        let transferred_reg = ensure_on_device(
            device,
            original_obj_id,
            typ.clone(),
            now,
            gpu_allocator,
            code,
            ctx,
            imctx,
        )?;
        let size = imctx.obj_size(original_obj_id);
        let (copied_typ, copied_size) = typ_for_copied(&typ, size);
        let copied_reg = code.alloc_register_id(copied_typ);
        allocate(
            device,
            copied_reg,
            original_obj_id, // this will be corrected later when assigning register for output
            now,
            copied_size,
            gpu_allocator,
            code,
            ctx,
            &imctx,
        )?;
        code.emit(Instruction::new_no_src(InstructionNode::Transfer {
            id: copied_reg,
            from: transferred_reg,
        }));
        copied_reg
    } else {
        let on_device_reg = ensure_on_device(
            device,
            original_obj_id,
            typ.clone(),
            now,
            gpu_allocator,
            code,
            ctx,
            imctx,
        )?;
        on_device_reg
    };
    Ok(input_reg)
}

pub fn lower_typ<Rt: RuntimeType>(t2typ: &super::Typ<Rt>, value: &Value) -> Typ {
    use super::typ::template::Typ::*;
    match t2typ {
        Poly((_, deg0)) => match value.node() {
            ValueNode::Poly { rotation, deg } => {
                assert!(deg0 == deg);
                Typ::ScalarArray {
                    len: *deg as usize,
                    meta: PolyMeta::Rotated(*rotation),
                }
            }
            ValueNode::SlicedPoly { slice, .. } => {
                // deg0 == deg should not be enforced, as deg is degree of the sliced polynomial, not the slice
                Typ::ScalarArray {
                    len: *deg0 as usize,
                    meta: PolyMeta::Sliced(slice.clone()),
                }
            }
            _ => panic!("expected poly here"),
        },
        Scalar => Typ::Scalar,
        Point => Typ::Point,
        PointBase { log_n } => Typ::PointBase {
            len: 2usize.pow(*log_n),
        },
        Transcript => Typ::Transcript,
        Tuple(..) => panic!("tuple unexpected"),
        Array(..) => panic!("array unexpected"),
        Any(tid, size) => Typ::Any(tid.clone().into(), *size as usize),
        _Phantom(_) => unreachable!(),
    }
}

fn deallocate_dead_objects<'a>(
    dead_objects: impl Iterator<Item = (ObjectId, &'a object_analysis::DeviceCollection)>,
    gpu_allocator: &mut GpuAllocator,
    code: &mut Code,
    ctx: &mut Context,
) {
    for (dead_obj, device_collection) in dead_objects {
        if ctx.inplace_obj.contains(&dead_obj) {
            continue;
        }

        if device_collection.gpu() {
            gpu_allocator.deallocate(dead_obj, code, ctx);
        }

        if device_collection.cpu() {
            deallocate_cpu(dead_obj, code, ctx);
        }
        if device_collection.stack() {
            deallocate_stack(dead_obj, code, ctx);
        }
    }
}

pub fn plan<'s, Rt: RuntimeType>(
    capacity: u64,
    cg: &Cg<'s, Rt>,
    g: &SubDigraph<'_, VertexId, super::Vertex<'s, Rt>>,
    seq: &[VertexId],
    obj_dies_after: &ObjectsDieAfter,
    obj_dies_after_reversed: &ObjectsDieAfterReversed,
    obj_def: &ObjectsDef,
    obj_gpu_next_use: &ObjectsGpuNextUse,
    vertex_inputs: &ObjectUse,
    devices: &BTreeMap<VertexId, Device>,
    uf_table: &user_function::Table<Rt>,
    obj_id_allocator: &mut IdAllocator<ObjectId>,
    mut libs: Libs,
) -> Result<Chunk<'s, Rt>, InsufficientSmithereenSpace> {
    let integral_sizes = collect_integral_sizes(cg, &g, &mut libs);
    let (ispace, sspace) =
        divide_integral_smithereens(capacity, integral_sizes.last().unwrap().clone());

    let mut gpu_allocator = GpuAllocator {
        ialloc: integral_allocator::regretting::Allocator::new(ispace, integral_sizes),
        salloc: smithereens_allocator::Allocator::new(sspace),
    };

    let mut ctx = Context {
        gpu_reg2addr: Injection::new(),
        gpu_addr_mapping: AddrMapping::new(),
        object_residence: BTreeMap::new(),
        reg_device: BTreeMap::new(),
        reg2obj: BTreeMap::new(),
        inplace_obj: BTreeSet::new(),
    };
    let mut register_types = BTreeMap::new();
    let imctx = ImmutableContext {
        obj_def,
        obj_dies_after,
        obj_dies_after_reversed,
        vertex_inputs,
    };

    let mut code = Code::new();

    for ((i, &vid), updated_next_uses) in
        seq.iter().enumerate().zip(obj_gpu_next_use.iter_updates())
    {
        if g.vertex(vid).is_virtual() {
            continue;
        }

        let now = Instant(i);
        gpu_allocator.tick(now);
        let exe_device = devices[&vid];

        // Prepare inputs
        // - Move input objects to desired device
        // - Make copies, if input is mutated and later used on the same device
        //   - If the input's space is used inplace by an output, the copy's object ID is that of the output
        //   - Otherwise, a new temporary object ID is allocated
        // - Build tuples, if needed
        let mutable_uses: Vec<VertexId> = g.vertex(vid).mutable_uses().collect();
        let outputs_inplace: Vec<Option<VertexId>> = g
            .vertex(vid)
            .outputs_inplace(uf_table, exe_device)
            .collect();
        let mut tuple_registers = Vec::new();
        let input_registers: Vec<RegisterId> = cg
            .g
            .vertex(vid)
            .uses()
            .zip(imctx.vertex_inputs.input_of(vid))
            .map(|(input_vid, input_vv)| {
                if mutable_uses.contains(&input_vid) && outputs_inplace.contains(&Some(input_vid)) {
                    // This input's space is used inplace by an output
                    let input_value = match input_vv {
                        object_analysis::VertexValue::Single(input_value) => input_value.clone(),
                        object_analysis::VertexValue::Tuple(..) => {
                            panic!("an inplace input must not be a tuple")
                        }
                    };
                    ctx.inplace_obj.insert(input_value.object_id());
                    ensure_copied(
                        input_value.device(),
                        vid,
                        lower_typ(g.vertex(input_vid).typ(), &input_value),
                        input_value.object_id(),
                        now,
                        &mut gpu_allocator,
                        &mut code,
                        &mut ctx,
                        &imctx,
                    )
                } else if mutable_uses.contains(&input_vid) {
                    let input_value = match input_vv {
                        object_analysis::VertexValue::Single(input_value) => input_value.clone(),
                        object_analysis::VertexValue::Tuple(..) => {
                            panic!("an mutable input must not be a tuple")
                        }
                    };
                    // This input is mutated
                    let copied_reg = ensure_copied(
                        input_value.device(),
                        vid,
                        lower_typ(g.vertex(input_vid).typ(), &input_value),
                        input_value.object_id(),
                        now,
                        &mut gpu_allocator,
                        &mut code,
                        &mut ctx,
                        &imctx,
                    )?;
                    ctx.remove_residence_in_reg_for_object(
                        input_value.object_id(),
                        input_value.device(),
                        copied_reg,
                    );
                    ctx.add_residence_for_object(
                        obj_id_allocator.alloc(),
                        copied_reg,
                        input_value.device(),
                    );
                    Ok(copied_reg)
                } else if !mutable_uses.contains(&input_vid)
                    && !outputs_inplace.contains(&Some(input_vid))
                {
                    // The input is not mutated
                    match input_vv {
                        object_analysis::VertexValue::Single(input_value) => ensure_on_device(
                            input_value.device(),
                            input_value.object_id(),
                            lower_typ(g.vertex(input_vid).typ(), &input_value),
                            now,
                            &mut gpu_allocator,
                            &mut code,
                            &mut ctx,
                            &imctx,
                        ),
                        object_analysis::VertexValue::Tuple(input_values) => {
                            let elements = input_values
                                .iter()
                                .zip(g.vertex(input_vid).typ().iter())
                                .map(|(input_value, input_typ)| {
                                    ensure_on_device(
                                        input_value.device(),
                                        input_value.object_id(),
                                        lower_typ(input_typ, input_value),
                                        now,
                                        &mut gpu_allocator,
                                        &mut code,
                                        &mut ctx,
                                        &imctx,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let tuple_reg = code.alloc_register_id(Typ::Tuple);
                            ctx.reg_device.insert(tuple_reg, DeterminedDevice::Stack);
                            code.emit(Instruction::new_no_src(InstructionNode::Tuple {
                                id: tuple_reg,
                                oprands: elements,
                            }));
                            tuple_registers.push(tuple_reg);
                            Ok(tuple_reg)
                        }
                    }
                } else {
                    panic!("an inplace output must use an mutable input")
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Some objects die before the vertex is executed
        deallocate_dead_objects(
            obj_dies_after_reversed.before(vid),
            &mut gpu_allocator,
            &mut code,
            &mut ctx,
        );

        // Allocate outputs
        // - If the vertex is Entry, only new register is needed
        // - Otherwise
        //   - If the output use some input's space, kills the input register in ctx and let the output register inherit its space
        //   - Otherwise, allocate new space for the output
        let v = g.vertex(vid);
        let output_registers: Vec<(RegisterId, Option<RegisterId>)> = if v
            .node()
            .no_allocate_output()
        {
            imctx.obj_def.values[&vid]
                .iter()
                .zip(v.outputs_inplace(uf_table, exe_device))
                .zip(v.typ().iter())
                .map(|((output_value, _), output_typ)| {
                    let output_obj = output_value.object_id();
                    let output_reg = code.alloc_register_id(lower_typ(output_typ, output_value));
                    ctx.add_residence_for_object(output_obj, output_reg, output_value.device());
                    Ok((output_reg, None))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            imctx.obj_def.values[&vid]
                .iter()
                .zip(v.outputs_inplace(uf_table, exe_device))
                .zip(v.typ().iter())
                .map(|((output_value, inplace), output_typ)| {
                    let output_obj = output_value.object_id();
                    let output_reg = code.alloc_register_id(lower_typ(output_typ, output_value));
                    if let Some(input_vid) = inplace {
                        let index = v.uses().position(|x| x == input_vid).unwrap();
                        let inplace_reg = input_registers[index];
                        ctx.add_residence_for_object(output_obj, output_reg, output_value.device());
                        ctx.remove_residence_in_reg_for_object(
                            ctx.reg2obj[&inplace_reg],
                            output_value.device(),
                            inplace_reg,
                        );
                        ctx.attempt_copy_gpu_addr_id_from(inplace_reg, output_reg);
                        Ok((output_reg, Some(inplace_reg)))
                    } else {
                        let size = imctx.obj_size(output_obj);
                        allocate(
                            output_value.device(),
                            output_reg,
                            output_obj,
                            obj_gpu_next_use
                                .first_use_of(output_obj)
                                .unwrap_or(Instant(usize::MAX)),
                            size,
                            &mut gpu_allocator,
                            &mut code,
                            &mut ctx,
                            &imctx,
                        )?;
                        Ok((output_reg, None))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let temp_register_obj = if let Some((temp_space_sizes, temp_space_device)) =
            cg.temporary_space_needed(vid, &mut libs)
        {
            let rs = temp_space_sizes
                .into_iter()
                .map(|temp_space_size| {
                    let temp_register =
                        code.alloc_register_id(Typ::GpuBuffer(temp_space_size as usize));
                    let temp_obj = obj_id_allocator.alloc();
                    allocate(
                        temp_space_device,
                        temp_register,
                        temp_obj,
                        now,
                        Size::Smithereen(SmithereenSize(temp_space_size)),
                        &mut gpu_allocator,
                        &mut code,
                        &mut ctx,
                        &imctx,
                    )?;
                    Ok((temp_register, temp_obj, temp_space_device))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Some(rs)
        } else {
            None
        };

        // Emit vertex

        let uses: Vec<VertexId> = v.uses().collect();
        let v3 = v.node().relabeled(|input_vid| {
            uses.iter()
                .zip(input_registers.iter())
                .find(|(vid, _)| **vid == input_vid)
                .map(|(_, &reg)| reg)
                .unwrap()
        });

        output_registers
            .iter()
            .zip(v.typ().iter())
            .for_each(|((output_reg, _), typ)| {
                register_types.insert(*output_reg, typ.clone());
            });

        code.emit(Instruction::new(
            InstructionNode::Type2 {
                ids: output_registers,
                temp: temp_register_obj
                    .as_ref()
                    .map_or_else(Vec::new, |ros| ros.iter().map(|(reg, ..)| *reg).collect()),
                vertex: v3,
                vid,
            },
            v.src().clone(),
        ));

        // Deallocate temporary space
        for (_, temp_obj, temp_device) in temp_register_obj.unwrap_or_default().into_iter() {
            deallocate(
                temp_device,
                temp_obj,
                &mut gpu_allocator,
                &mut code,
                &mut ctx,
            );
        }

        // We can't deallocate those objects that dies exactly after return vertex, as they are returned.
        // Though this doesn't matter as the runtime should exit immediately at seeing a return instruction.
        if v.node().is_return() {
            break;
        }

        // Deallocate used tuple register
        for tuple_reg in tuple_registers.into_iter() {
            code.emit(Instruction::new_no_src(InstructionNode::StackFree {
                id: tuple_reg,
            }));
        }

        // Deallocate dead register
        deallocate_dead_objects(
            obj_dies_after_reversed.after(vid),
            &mut gpu_allocator,
            &mut code,
            &mut ctx,
        );

        // Update next_uses on GPU integral allocator
        for (obj, next_use) in updated_next_uses {
            let _updated = gpu_allocator.update_next_use(obj, next_use, &mut ctx);
        }
    }

    let (register_types, reg_id_allocator) = code.1.freeze();

    Ok(Chunk {
        instructions: code.0,
        register_types,
        register_devices: ctx.reg_device,
        gpu_addr_mapping: ctx.gpu_addr_mapping,
        reg_id_allocator,
        libs,
        _phantom: PhantomData,
    })
}
