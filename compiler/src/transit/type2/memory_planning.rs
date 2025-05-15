static DEBUG: bool = false;

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use zkpoly_common::{
    bijection::Bijection,
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
    object_analysis::{AtModifier, VertexValue},
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
mod page_allocator;

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

impl std::ops::AddAssign<RegisterPlaces> for RegisterPlaces {
    fn add_assign(&mut self, rhs: RegisterPlaces) {
        for d in DeterminedDevice::iter() {
            self.get_device_mut(d).extend(rhs.get_device(d).iter());
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryBlock {
    Gpu(ObjectId, AddrId),
    Cpu(ObjectId),
}

impl MemoryBlock {
    pub fn object_id(&self) -> ObjectId {
        match self {
            MemoryBlock::Gpu(obj_id, _) => *obj_id,
            MemoryBlock::Cpu(obj_id) => *obj_id,
        }
    }

    pub fn with_object_id(&self, obj_id: ObjectId) -> Self {
        match self {
            MemoryBlock::Gpu(_, addr_id) => MemoryBlock::Gpu(obj_id, *addr_id),
            MemoryBlock::Cpu(_) => MemoryBlock::Cpu(obj_id),
        }
    }
}

#[derive(Debug)]
struct Context {
    gpu_obj2addr: Bijection<ObjectId, AddrId>,
    gpu_addr_mapping: AddrMapping,
    gpu_reg2addr: BTreeMap<RegisterId, AddrId>,
    object_residence: BTreeMap<ObjectId, RegisterPlaces>,
    /// Tuple registers will not be included
    reg_device: BTreeMap<RegisterId, DeterminedDevice>,
    reg2mb: BTreeMap<RegisterId, MemoryBlock>,
    inplace_obj: BTreeMap<ObjectId, DeviceSpecific<bool>>,
}

#[derive(Debug)]
struct ImmutableContext<'a> {
    obj_def: &'a ObjectsDef,
    obj_dies_after: &'a ObjectsDieAfter,
    obj_dies_after_reversed: &'a ObjectsDieAfterReversed,
    obj_gpu_next_use: &'a ObjectsGpuNextUse,
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
        if DEBUG {
            println!(
                "[MP.ctx] Add residence of {:?} in {:?} on {:?}",
                obj_id, reg_id, device
            );
        }

        self.object_residence
            .entry(obj_id)
            .or_insert_with(|| RegisterPlaces::empty())
            .get_device_mut(device)
            .insert(reg_id);
        self.reg_device.insert(reg_id, device);

        self.set_reg2obj(reg_id, device, obj_id);
    }

    pub fn set_reg2obj(&mut self, reg_id: RegisterId, device: DeterminedDevice, obj_id: ObjectId) {
        let mb = match device {
            DeterminedDevice::Cpu => MemoryBlock::Cpu(obj_id),
            DeterminedDevice::Gpu => {
                MemoryBlock::Gpu(obj_id, *self.gpu_reg2addr.get(&reg_id).unwrap())
            }
            DeterminedDevice::Stack => panic!("objects cannot reside on stack"),
        };

        self.reg2mb.insert(reg_id, mb);
    }

    pub fn remove_residence_in_reg_for_object(
        &mut self,
        obj_id: ObjectId,
        device: DeterminedDevice,
        reg_id: RegisterId,
    ) {
        if DEBUG {
            println!(
                "[MP.ctx] Remove residence of {:?} in {:?} on {:?}",
                obj_id, reg_id, device
            );
        }

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
        if DEBUG {
            println!(
                "[MP.ctx] Remove all residence of {:?} on {:?}",
                obj_id, device
            );
        }

        self.object_residence
            .get_mut(&obj_id)
            .unwrap()
            .get_device_mut(device)
            .clear();

        if device == DeterminedDevice::Gpu {
            let addr = self.gpu_obj2addr.remove_forward(&obj_id).unwrap();

            if DEBUG {
                println!(
                    "[MP.ctx] {:?} used to store in {:?}({:?}), but not any longer",
                    obj_id, addr, self.gpu_addr_mapping[addr]
                );
            }
        }
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

    pub fn set_gpu_addr_id_for_obj(&mut self, obj_id: ObjectId, addr_id: AddrId) {
        self.gpu_obj2addr.insert(obj_id, addr_id);

        if DEBUG {
            println!(
                "[MP.ctx] {:?} now stores in {:?}({:?})",
                obj_id, addr_id, self.gpu_addr_mapping[addr_id]
            );
        }
    }

    pub fn set_gpu_addr_id_for_reg(&mut self, reg_id: RegisterId, addr_id: AddrId) {
        self.gpu_reg2addr.insert(reg_id, addr_id);

        if DEBUG {
            println!(
                "[MP.ctx] {:?} now points to {:?}({:?})",
                reg_id, addr_id, self.gpu_addr_mapping[addr_id]
            );
        }
    }

    pub fn mark_space_inplace(&mut self, obj_id: ObjectId, device: DeterminedDevice) {
        if DEBUG {
            println!("[MP.ctx] Mark {:?} inplace on {:?}", obj_id, device);
        }
        *self
            .inplace_obj
            .entry(obj_id)
            .or_default()
            .get_device_mut(device) = true;
    }

    pub fn try_copy_reg_addr_id(&mut self, from: RegisterId, to: RegisterId) {
        if let Some(&addr_id) = self.gpu_reg2addr.get(&from) {
            self.gpu_reg2addr.insert(to, addr_id);

            if DEBUG {
                println!(
                    "[MP.ctx] Copy address recrod from {:?}, {:?} now also points to {:?}({:?})",
                    from, to, addr_id, self.gpu_addr_mapping[addr_id]
                );
            }
        }
    }

    pub fn get_residences_of_object(
        &self,
        obj_id: ObjectId,
        device: DeterminedDevice,
    ) -> Vec<RegisterId> {
        self.object_residence
            .get(&obj_id)
            .unwrap()
            .get_device(device)
            .iter()
            .copied()
            .collect()
    }

    pub fn is_obj_alive_on(&self, obj_id: ObjectId, device: DeterminedDevice) -> bool {
        self.object_residence
            .get(&obj_id)
            .is_some_and(|places| !places.get_device(device).is_empty())
    }

    pub fn normalized_typ_for_obj(&self, obj_id: ObjectId, code: &Code) -> Typ {
        let reg = self.object_residence[&obj_id]
            .get_any()
            .unwrap()
            .0
            .pop()
            .unwrap();
        let typ = code.typ_of(reg).normalized();
        typ
    }

    pub fn reg2obj(&self, reg_id: RegisterId) -> ObjectId {
        self.reg2mb[&reg_id].object_id()
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
        let obj_id = *ctx.gpu_obj2addr.get_backward(&victim).unwrap();

        if DEBUG {
            println!("[MP.move_victims] Eject victim {:?} at {:?}({:?} to CPU", obj_id, victim, ctx.gpu_addr_mapping[victim]);
        }


        // If obj_id is already on CPU, or if obj_id is sliced from some sliced_obj on CPU, we don't need to transfer anything.
        // The latter case is because that sliced_obj dies after obj_id in object analysis.
        if ctx.is_obj_alive_on(obj_id, DeterminedDevice::Cpu) {
            if DEBUG {
                println!("[MP.move_victims] Victim {:?} is already on CPU", obj_id);
            }
            ctx.remove_residence_for_object(obj_id, DeterminedDevice::Gpu);
            return;
        }
        if let Some((sliced_obj, _)) = imctx.vertex_inputs.cloned_slice_from(obj_id) {
            if ctx.is_obj_alive_on(sliced_obj, DeterminedDevice::Cpu) {
                if DEBUG {
                    println!(
                        "[MP.move_victims] Victim {:?} is sliced from {:?} which is already on CPU",
                        obj_id, sliced_obj
                    );
                }
                ctx.remove_residence_for_object(obj_id, DeterminedDevice::Gpu);
                return;
            }
        }

        let normalized_typ = ctx.normalized_typ_for_obj(obj_id, code);
        let src_reg = ensure_same_type(
            obj_id,
            normalized_typ.clone(),
            ctx.get_residences_of_object(obj_id, DeterminedDevice::Gpu).into_iter(),
            code,
            ctx,
        );

        let new_reg = code.alloc_register_id(normalized_typ.clone());
        let size = imctx.obj_size(obj_id);
        allocate_cpu(obj_id, new_reg, size, code, ctx, true);
        code.emit(Instruction::new_no_src(InstructionNode::Transfer {
            id: new_reg,
            from: src_reg,
        }));

        ctx.remove_residence_for_object(obj_id, DeterminedDevice::Gpu);
        ctx.add_residence_for_object(obj_id, new_reg, DeterminedDevice::Cpu);

        if DEBUG {
            println!(
                "[MP.move_victims] Victim {:?} is transferred to CPU from register {:?} in new register {:?} with typ {:?}",
                obj_id, src_reg, new_reg, &normalized_typ
            );
        }
    });
}

fn get_gpu_reg_of_normalized_typ(
    addr_id: AddrId,
    ctx: &mut Context,
    code: &mut Code,
) -> (RegisterId, Typ, ObjectId) {
    let obj_id = *ctx.gpu_obj2addr.get_backward(&addr_id).unwrap();

    let normalized_typ = ctx.normalized_typ_for_obj(obj_id, code);
    let reg = ensure_same_type(
        obj_id,
        normalized_typ.clone(),
        ctx.get_residences_of_object(obj_id, DeterminedDevice::Gpu)
            .into_iter(),
        code,
        ctx,
    );

    (reg, normalized_typ, obj_id)
}

fn unoffset_addr_id(addr_id: AddrId, offset: u64, ctx: &Context) -> (Size, u64) {
    let (addr, size) = ctx.gpu_addr_mapping[addr_id];
    match size {
        Size::Integral(_) => (size, addr.unoffset(offset).0),
        Size::Smithereen(_) => (size, addr.0),
    }
}

fn allocate_gpu_integral(
    size: IntegralSize,
    next_use: Instant,
    gpu_ialloc: &mut integral_allocator::regretting::Allocator,
    ioffset: u64,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
) -> Result<AddrId, Error> {
    let addr = gpu_ialloc.allocate(
        size,
        next_use,
        &mut GpuAddrMappingHandler::new(&mut ctx.gpu_addr_mapping, ioffset),
    );

    if let Some((transfers, addr)) = addr {
        if DEBUG {
            println!(
                "[MP.gpu_ialloc] Ialloc allocated GPU space at {:?}({:?}), reallocating on GPU {:?}",
                addr, unoffset_addr_id(addr, ioffset, ctx), &transfers
            );
        }

        for t in transfers {
            let (reg_from, normalized_typ, obj_id) =
                get_gpu_reg_of_normalized_typ(t.from, ctx, code);

            let reg_to = code.alloc_register_id(normalized_typ);
            code.emit(Instruction::new_no_src(InstructionNode::GpuMalloc {
                id: reg_to,
                addr: t.to,
            }));
            ctx.set_gpu_addr_id_for_reg(reg_to, t.to);

            code.emit(Instruction::new_no_src(InstructionNode::Transfer {
                id: reg_to,
                from: reg_from,
            }));

            code.emit(Instruction::new_no_src(InstructionNode::StackFree {
                id: reg_from,
            }));

            ctx.remove_residence_in_reg_for_object(obj_id, DeterminedDevice::Gpu, reg_from);
            ctx.add_residence_for_object(obj_id, reg_to, DeterminedDevice::Gpu);
            ctx.set_gpu_addr_id_for_obj(obj_id, t.to);
        }

        return Ok(addr);
    }

    let (addr, victims) = gpu_ialloc
        .decide_and_realloc_victim(
            size,
            next_use,
            &mut GpuAddrMappingHandler::new(&mut ctx.gpu_addr_mapping, ioffset),
        )
        .ok_or(Error::VertexInputsAndOutputsNotAccommodated(0.into()))?;

    if DEBUG {
        println!(
            "[MP.gpu_ialloc] Ialloc allocated GPU space at {:?}({:?}), ejecting victims {:?}",
            addr,
            unoffset_addr_id(addr, ioffset, ctx),
            &victims
        );
    }

    move_victims(&victims, code, ctx, imctx);

    Ok(addr)
}

#[derive(Debug)]
pub enum Error {
    InsufficientSmithereenSpace,
    VertexInputsAndOutputsNotAccommodated(VertexId),
}

impl Error {
    pub fn try_with_vid(self, vid: VertexId) -> Self {
        match self {
            Error::VertexInputsAndOutputsNotAccommodated(_) => {
                Error::VertexInputsAndOutputsNotAccommodated(vid)
            }
            _ => self,
        }
    }
}

fn allocate_gpu_smithereen(
    size: SmithereenSize,
    gpu_salloc: &mut smithereens_allocator::Allocator,
    soffset: u64,
    mapping: &mut AddrMapping,
) -> Result<AddrId, Error> {
    if let Some(addr) = gpu_salloc.allocate(size, &mut GpuAddrMappingHandler::new(mapping, soffset))
    {
        return Ok(addr);
    }

    Err(Error::InsufficientSmithereenSpace)
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
    ) -> Result<(), Error> {
        self.allocate_(size, Some(obj_id), reg_id, next_use, code, ctx, imctx)
    }

    pub fn allocate_no_ctx_update(
        &mut self,
        size: Size,
        reg_id: RegisterId,
        next_use: Instant,
        code: &mut Code,
        ctx: &mut Context,
        imctx: &ImmutableContext,
    ) -> Result<(), Error> {
        self.allocate_(size, None, reg_id, next_use, code, ctx, imctx)
    }

    fn allocate_(
        &mut self,
        size: Size,
        obj_id: Option<ObjectId>,
        reg_id: RegisterId,
        next_use: Instant,
        code: &mut Code,
        ctx: &mut Context,
        imctx: &ImmutableContext,
    ) -> Result<(), Error> {
        let ioffset = self.ioffset();
        let soffset = self.soffset();

        let addr = match normalize_size(size) {
            Size::Integral(size) => {
                allocate_gpu_integral(size, next_use, &mut self.ialloc, ioffset, code, ctx, imctx)?
            }
            Size::Smithereen(size) => {
                allocate_gpu_smithereen(size, &mut self.salloc, soffset, &mut ctx.gpu_addr_mapping)?
            }
        };

        code.emit(Instruction::new_no_src(InstructionNode::GpuMalloc {
            id: reg_id,
            addr,
        }));

        ctx.set_gpu_addr_id_for_reg(reg_id, addr);
        if let Some(obj_id) = obj_id {
            ctx.set_gpu_addr_id_for_obj(obj_id, addr);
            ctx.add_residence_for_object(obj_id, reg_id, DeterminedDevice::Gpu);
        } else {
            ctx.reg_device.insert(reg_id, DeterminedDevice::Gpu);
        }

        if DEBUG {
            println!(
                "[MP.gpu_alloc] Allocated GPU space for {:?} at {:?}({:?}) in {:?}",
                obj_id,
                addr,
                unoffset_addr_id(addr, ioffset, ctx),
                reg_id
            );
        }

        Ok(())
    }

    pub fn deallocate(&mut self, obj_id: ObjectId, code: &mut Code, ctx: &mut Context) {
        if let Some(reg_id) = ctx
            .pop_residence_of_object(obj_id, DeterminedDevice::Gpu)
        {
            let addr: AddrId = ctx.gpu_obj2addr.get_forward(&obj_id).cloned().unwrap_or_else(|| panic!("no GPU addr recorded for {:?}", obj_id));

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

            if DEBUG {
                println!(
                    "[MP.gpu_alloc] Deallocated GPU space for {:?} at {:?}({:?}) in {:?}",
                    obj_id,
                    addr,
                    unoffset_addr_id(addr, self.ioffset(), ctx),
                    reg_id
                );
            }
        } else {
            if DEBUG {
                println!(
                    "[MP.gpu_alloc] Deallocating GPU space for {:?}, but the object is not on GPU, skipping",
                    obj_id
                );
            }
        }
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
        if let Some(addr) = ctx.gpu_obj2addr.get_forward(&obj_id).cloned() {
            if DEBUG {
                println!(
                    "[MP.gpu_alloc] Updating next use of {:?} at {:?}({:?}) to {:?}",
                    obj_id,
                    addr,
                    unoffset_addr_id(addr, self.ioffset(), ctx),
                    next_use
                );
            }

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

fn allocate_cpu(
    obj_id: ObjectId,
    id: RegisterId,
    size: Size,
    code: &mut Code,
    ctx: &mut Context,
    update_obj_residence: bool,
) {
    code.emit(Instruction::new_no_src(InstructionNode::CpuMalloc {
        id,
        size,
    }));
    if update_obj_residence {
        ctx.add_residence_for_object(obj_id, id, DeterminedDevice::Cpu);
    }
}

fn allocate_(
    device: DeterminedDevice,
    reg_id: RegisterId,
    obj_id: ObjectId,
    next_use: Instant,
    size: Size,
    gpu_allocator: &mut GpuAllocator,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
    update_obj_residence: bool,
) -> Result<(), Error> {
    match device {
        DeterminedDevice::Gpu if update_obj_residence => {
            gpu_allocator.allocate(size, obj_id, reg_id, next_use, code, ctx, imctx)?
        }
        DeterminedDevice::Gpu => {
            gpu_allocator.allocate_no_ctx_update(size, reg_id, next_use, code, ctx, imctx)?
        }
        DeterminedDevice::Cpu => {
            allocate_cpu(obj_id, reg_id, size, code, ctx, update_obj_residence)
        }
        DeterminedDevice::Stack => ctx.add_residence_for_object(obj_id, reg_id, device),
    };
    Ok(())
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
) -> Result<(), Error> {
    allocate_(
        device,
        reg_id,
        obj_id,
        next_use,
        size,
        gpu_allocator,
        code,
        ctx,
        imctx,
        true,
    )
}

fn deallocate_cpu(obj_id: ObjectId, code: &mut Code, ctx: &mut Context) {
    if let Some(reg_id) = ctx.pop_residence_of_object(obj_id, DeterminedDevice::Cpu) {
        code.emit(Instruction::new_no_src(InstructionNode::CpuFree {
            id: reg_id,
        }));

        while let Some(reg_id) = ctx.pop_residence_of_object(obj_id, DeterminedDevice::Cpu) {
            code.emit(Instruction::new_no_src(InstructionNode::StackFree {
                id: reg_id,
            }));
        }

        ctx.remove_residence_for_object(obj_id, DeterminedDevice::Cpu);

        if DEBUG {
            println!(
                "[MP.deallocate] Deallocated Cpu space for {:?} using {:?}",
                obj_id, reg_id
            );
        }
    } else {
        if DEBUG {
            println!("[MP.deallocate] {:?} not alive on Cpu, skipping", obj_id);
        }
    }
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
    obj_id: ObjectId,
    typ: Typ,
    mut candidate_regs: impl Iterator<Item = RegisterId> + Clone,
    code: &mut Code,
    ctx: &mut Context,
) -> RegisterId {
    if let Some(reg_id) = candidate_regs.clone().find(|&r| code.typ_of(r) == &typ) {
        if DEBUG {
            println!(
                "[MP.ensure_same_type] Found same type register {:?} for {:?} with typ {:?}",
                reg_id, obj_id, &typ
            );
        }
        reg_id
    } else {
        let (deg, meta) = typ.unwrap_poly();
        let meta = meta.clone();
        let reg_id = candidate_regs.next().unwrap().clone();
        let device = ctx.reg_device[&reg_id];

        let new_reg = code.alloc_register_id(typ.clone());
        let (slice_offset, slice_len) = meta.offset_and_len(deg as u64);
        code.emit(Instruction::new_no_src(InstructionNode::SetPolyMeta {
            id: new_reg,
            from: reg_id,
            offset: slice_offset as usize,
            len: slice_len as usize,
        }));
        ctx.try_copy_reg_addr_id(reg_id, new_reg);
        ctx.add_residence_for_object(obj_id, new_reg, device);

        if DEBUG {
            println!(
                "[MP.ensure_same_type] Built new register {:?} for {:?} on {:?} for typ {:?}",
                new_reg, obj_id, device, &typ
            );
        }

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
        if DEBUG {
            println!(
                "[MP.attemp_on_device] Found candidate registers {:?} for {:?} on {:?}",
                &candidate_registers, obj_id, device
            );
        }

        ensure_same_type(obj_id, typ, candidate_registers.into_iter(), code, ctx)
    } else {
        if let Some((from_reg_candidates, _)) = ctx.object_residence[&obj_id].get_any() {
            if DEBUG {
                println!(
                    "[MP.attemp_on_device] Found candidate registers {:?} for {:?} on some device",
                    &from_reg_candidates, obj_id
                );
            }
            ensure_same_type(obj_id, typ, from_reg_candidates.iter().copied(), code, ctx)
        } else {
            let (sliced_obj, meta) = imctx.vertex_inputs.cloned_slice_from(obj_id).unwrap();
            let (len, _) = typ.unwrap_poly();
            let sliced_typ = Typ::ScalarArray { len, meta };
            attempt_on_device(device, sliced_obj, sliced_typ, code, ctx, imctx)
        }
    }
}

fn register_for_object(
    device: DeterminedDevice,
    obj_id: ObjectId,
    typ: Typ,
    code: &mut Code,
    ctx: &mut Context,
) -> Option<RegisterId> {
    let candidate_registers = ctx.object_residence[&obj_id]
        .get_device(device)
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    if !candidate_registers.is_empty() {
        if DEBUG {
            println!(
                "[MP.register_for_object] Found candidate registers {:?} with {:?} for {:?} on {:?}",
                &candidate_registers, typ.clone(), obj_id, device
            );
        }

        Some(ensure_same_type(
            obj_id,
            typ,
            candidate_registers.into_iter(),
            code,
            ctx,
        ))
    } else {
        if DEBUG {
            println!(
                "[MP.register_for_object] {:?} with {:?} not found on {:?}",
                obj_id, typ, device
            );
        }
        None
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
) -> Result<RegisterId, Error> {
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
        if DEBUG {
            println!(
                "[MP.ensure_on_device] Found candidate registers {:?} for {:?} on {:?}",
                &candidate_registers, obj_id, device
            );
        }

        Ok(ensure_same_type(
            obj_id,
            typ,
            candidate_registers.into_iter(),
            code,
            ctx,
        ))
    } else {
        let size = imctx.obj_size(obj_id);

        if let Some((from_reg_candidates, from_device)) = ctx
            .object_residence
            .get(&obj_id)
            .map(|x| x.get_any())
            .flatten()
        {
            let copied_typ = ctx.normalized_typ_for_obj(obj_id, &code);
            let new_reg = code.alloc_register_id(copied_typ.clone());
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
                obj_id,
                copied_typ.clone(),
                from_reg_candidates.iter().copied(),
                code,
                ctx,
            );

            code.emit(Instruction::new_no_src(InstructionNode::Transfer {
                id: new_reg,
                from: from_reg,
            }));

            if copied_typ != typ {
                if DEBUG {
                    println!("[MP.ensure_on_device] Transferred {:?} from {:?} in {:?} to {:?} in {:?}, correcting typ", obj_id, from_reg, from_device, new_reg, device);
                }
                let corrected_reg = emit_correct_type(new_reg, device, obj_id, typ, code, ctx);

                // Throw away the old register
                ctx.remove_residence_in_reg_for_object(obj_id, device, new_reg);
                ctx.add_residence_for_object(obj_id, corrected_reg, device);
                code.emit(Instruction::new_no_src(InstructionNode::StackFree {
                    id: new_reg,
                }));

                Ok(corrected_reg)
            } else {
                if DEBUG {
                    println!(
                        "[MP.ensure_on_device] Transferred {:?} from {:?} in {:?} to {:?} in {:?}",
                        obj_id, from_reg, from_device, new_reg, device
                    );
                }
                Ok(new_reg)
            }
        } else {
            let new_reg = code.alloc_register_id(typ.clone());
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

            let sliced_poly_typ = ctx.normalized_typ_for_obj(sliced_obj, code);
            let (len, _) = sliced_poly_typ.unwrap_poly();
            let slice_typ = Typ::ScalarArray { len, meta };

            let from_reg =
                attempt_on_device(device, sliced_obj, slice_typ.clone(), code, ctx, imctx);

            code.emit(Instruction::new_no_src(InstructionNode::Transfer {
                id: new_reg,
                from: from_reg,
            }));

            if DEBUG {
                println!("[MP.ensure_on_device] Sliced object {:?}, stored in {:?}, with typ {:?} from {:?} in {:?} with typ {:?}", obj_id, new_reg, typ, sliced_obj, from_reg, slice_typ);
            }
            Ok(new_reg)
        }
    }
}

fn emit_correct_type(
    reg_id: RegisterId,
    copied_reg_device: DeterminedDevice,
    original_obj_id: ObjectId,
    typ: Typ,
    code: &mut Code,
    ctx: &mut Context,
) -> RegisterId {
    let corrected_reg = code.alloc_register_id(typ.clone());
    let (len, meta) = typ.unwrap_poly();
    let (offset, slice_len) = meta.offset_and_len(len as u64);
    code.emit(Instruction::new_no_src(InstructionNode::SetPolyMeta {
        id: corrected_reg,
        from: reg_id,
        offset: offset as usize,
        len: slice_len as usize,
    }));

    ctx.try_copy_reg_addr_id(reg_id, corrected_reg);
    if DEBUG {
        println!(
            "[MP.emit_correct_type] Corrected typ of {:?} storing {:?} to {:?} in {:?}",
            reg_id, original_obj_id, corrected_reg, corrected_reg
        );
    }

    corrected_reg
}

fn ensure_copied(
    device: DeterminedDevice,
    vid: VertexId,
    typ: Typ,
    original_obj_id: ObjectId,
    new_obj_id: ObjectId,
    now: Instant,
    gpu_allocator: &mut GpuAllocator,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
) -> Result<RegisterId, Error> {
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
        let copied_typ = ctx.normalized_typ_for_obj(original_obj_id, &code);
        let transferred_reg = ensure_on_device(
            device,
            original_obj_id,
            copied_typ.clone(),
            now,
            gpu_allocator,
            code,
            ctx,
            imctx,
        )?;
        let size = imctx.obj_size(original_obj_id);
        let copied_reg = code.alloc_register_id(copied_typ.clone());
        allocate_(
            device,
            copied_reg,
            new_obj_id,
            now,
            size,
            gpu_allocator,
            code,
            ctx,
            &imctx,
            true,
        )?;

        code.emit(Instruction::new_no_src(InstructionNode::Transfer {
            id: copied_reg,
            from: transferred_reg,
        }));

        if copied_typ != typ {
            if DEBUG {
                println!("[MP.ensure_copied] Copied {:?} from {:?} to {:?} with typ {:?}, correcting typ", original_obj_id, transferred_reg, copied_reg, copied_typ);
            }
            let corrected_reg =
                emit_correct_type(copied_reg, device, original_obj_id, typ, code, ctx);

            ctx.add_residence_for_object(new_obj_id, corrected_reg, device);
            ctx.remove_residence_in_reg_for_object(new_obj_id, device, copied_reg);

            // Throw away the old register
            code.emit(Instruction::new_no_src(InstructionNode::StackFree {
                id: copied_reg,
            }));

            corrected_reg
        } else {
            if DEBUG {
                println!(
                    "[MP.ensure_copied] Copied {:?} from {:?} to {:?} with typ {:?}",
                    original_obj_id, transferred_reg, copied_reg, copied_typ
                );
            }

            copied_reg
        }
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

        if DEBUG {
            println!(
                "[MP.ensure_copied] No need to copy {:?}, using {:?}",
                original_obj_id, on_device_reg
            );
        }

        while let Some(r) = ctx.pop_residence_of_object(original_obj_id, device) {
            if r != on_device_reg {
                code.emit(Instruction::new_no_src(InstructionNode::StackFree {
                    id: r,
                }))
            }
        }
        ctx.add_residence_for_object(new_obj_id, on_device_reg, device);

        ctx.mark_space_inplace(new_obj_id, device);
        ctx.mark_space_inplace(original_obj_id, device);

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
            ValueNode::SlicedPoly { slice, deg } => {
                // deg0 == deg should not be enforced, as deg is degree of the sliced polynomial, not the slice
                Typ::ScalarArray {
                    len: *deg as usize,
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
        let inplace_collection = ctx.inplace_obj.entry(dead_obj).or_default().clone();
        let dead_devices = device_collection.clone() - inplace_collection.clone();

        if DEBUG {
            println!(
                "[MP.deallocate_dead_objects] Deallocating {:?} on {:?}, original device collection is {:?}, masked by inplace object collection {:?}",
                dead_obj, &dead_devices, &device_collection, &inplace_collection
            );
        }

        if dead_devices.gpu() {
            gpu_allocator.deallocate(dead_obj, code, ctx);
        }

        if dead_devices.cpu() {
            deallocate_cpu(dead_obj, code, ctx);
        }
        if dead_devices.stack() {
            deallocate_stack(dead_obj, code, ctx);
        }
    }
}

fn check_values_ready<'a>(values: impl Iterator<Item = &'a Value>, ctx: &Context) -> bool {
    for v in values {
        let live = ctx.is_obj_alive_on(v.object_id(), v.device());

        if DEBUG {
            println!(
                "[MP.check_values_ready] checking {:?} live on {:?} = {}",
                v.object_id(),
                v.device(),
                live
            );
        }
        if !live {
            return false;
        }
    }
    true
}

fn plan_vertex<'s, Rt: RuntimeType>(
    vid: VertexId,
    exe_device: Device,
    now: Instant,
    updated_next_uses: impl Iterator<Item = (ObjectId, Instant)>,
    cg: &Cg<'s, Rt>,
    g: &SubDigraph<'_, VertexId, super::Vertex<'s, Rt>>,
    ctx: &mut Context,
    register_types: &mut BTreeMap<RegisterId, super::Typ<Rt>>,
    imctx: &ImmutableContext,
    code: &mut Code<'s>,
    gpu_allocator: &mut GpuAllocator,
    uf_table: &user_function::Table<Rt>,
    obj_id_allocator: &mut IdAllocator<ObjectId>,
    libs: &mut Libs,
) -> Result<bool, Error> {
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
    let inplace_inputs_objects: BTreeSet<ObjectId> = outputs_inplace.iter()
        .filter_map(|x| *x)
        .map(|vid| imctx.obj_def.values[&vid].unwrap_single().object_id())
        .collect();

    let mut tuple_registers = Vec::new();
    let mut mutable_object2temp: BTreeMap<ObjectId, ObjectId> = BTreeMap::new();

    let v = g.vertex(vid);

    let input_values: Vec<(VertexValue, Vec<Typ>)> = v
        .uses()
        .zip(imctx.vertex_inputs.input_of(vid))
        .map(|(input_vid, input_vv)| {
            if mutable_uses.contains(&input_vid) && 
                input_vv.try_unwrap_single().is_some_and(|v| inplace_inputs_objects.contains(&v.object_id())) {
                let input_value = match input_vv {
                    VertexValue::Single(input_value) => input_value.clone(),
                    VertexValue::Tuple(..) => {
                        panic!("an inplace input must not be a tuple")
                    }
                };

                let typ = lower_typ(g.vertex(input_vid).typ(), &input_value);

                let (tmp_obj_id, typ) = if let Some(tmp_obj_id) =
                    mutable_object2temp.get(&input_value.object_id()) {
 
                    if DEBUG {
                        println!(
                            "[MP.plan] {:?} is inplace input of {:?} from {:?}, using already assigned temporary {:?}",
                            input_value, vid, input_vid, tmp_obj_id
                        );
                    }

                    (*tmp_obj_id, typ.clone())
                } else {
                    let tmp_obj_id = obj_id_allocator.alloc();

                    if DEBUG {
                        println!(
                            "[MP.plan] {:?} is inplace input of {:?} from {:?}, assigning temporary {:?}",
                            input_value, vid, input_vid, tmp_obj_id
                        );
                    }


                    let _ = ensure_copied(
                        input_value.device(),
                        vid,
                        typ.clone(),
                        input_value.object_id(),
                        tmp_obj_id,
                        now,
                        gpu_allocator,
                        code,
                        ctx,
                        &imctx,
                    )?;

                    mutable_object2temp.insert(input_value.object_id(), tmp_obj_id);

                    (tmp_obj_id, typ)
                };

                Ok((
                    VertexValue::Single(input_value.with_object_id(tmp_obj_id)),
                    vec![typ],
                ))
            } else if mutable_uses.contains(&input_vid) {
                let input_value = match input_vv {
                    object_analysis::VertexValue::Single(input_value) => input_value.clone(),
                    object_analysis::VertexValue::Tuple(..) => {
                        panic!("an mutable input must not be a tuple")
                    }
                };

                let tmp_obj_id = obj_id_allocator.alloc();

                let typ = lower_typ(g.vertex(input_vid).typ(), &input_value);

                // This input is mutated
                let _ = ensure_copied(
                    input_value.device(),
                    vid,
                    typ.clone(),
                    input_value.object_id(),
                    tmp_obj_id,
                    now,
                    gpu_allocator,
                    code,
                    ctx,
                    &imctx,
                )?;

                Ok((
                    VertexValue::Single(input_value.with_object_id(tmp_obj_id)),
                    vec![typ],
                ))
            } else {
                // The input is not mutated
                match input_vv {
                    object_analysis::VertexValue::Single(input_value) => {
                        if DEBUG {
                            println!(
                                "[MP.plan] {:?} is an immutable input of {:?} from {:?}",
                                input_value, vid, input_vid
                            );
                        }
                        let typ = lower_typ(g.vertex(input_vid).typ(), &input_value);

                        let obj_id = if let Some(tmp_obj_id) = mutable_object2temp.get(&input_value.object_id()) {
                            if DEBUG {
                                println!(
                                    "[MP.plan] The input object is inplaced, using temporary {:?}", tmp_obj_id
                                )
                            }
                            tmp_obj_id.clone()
                        } else {
                            let _ = ensure_on_device(
                                input_value.device(),
                                input_value.object_id(),
                                typ.clone(),
                                now,
                                gpu_allocator,
                                code,
                                ctx,
                                &imctx,
                            )?;

                            input_value.object_id()
                        };

                       Ok((VertexValue::Single(input_value.with_object_id(obj_id)), vec![typ]))
                    }
                    object_analysis::VertexValue::Tuple(input_values) => {
                        if DEBUG {
                            println!(
                                "[MP.plan] {:?} are immutable inputs of {:?}",
                                input_values, vid
                            );
                        }

                        let typs = input_values
                            .iter()
                            .zip(g.vertex(input_vid).typ().iter())
                            .map(|(input_value, input_typ)| {
                                let typ = lower_typ(input_typ, input_value);
                                ensure_on_device(
                                    input_value.device(),
                                    input_value.object_id(),
                                    typ.clone(),
                                    now,
                                    gpu_allocator,
                                    code,
                                    ctx,
                                    &imctx,
                                )?;
                                Ok(typ)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok((VertexValue::Tuple(input_values.clone()), typs))
                    }
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Some objects die before the vertex is executed
    deallocate_dead_objects(
        imctx.obj_dies_after_reversed.before(vid),
        gpu_allocator,
        code,
        ctx,
    );

    // Allocate outputs
    // - If the vertex is Entry, only new register is needed
    // - Otherwise
    //   - If the output use some input's space, kills the input register in ctx and let the output register inherit its space
    //   - Otherwise, allocate new space for the output
    if !v.node().no_allocate_output() {
        imctx.obj_def.values[&vid]
            .iter()
            .zip(outputs_inplace.iter())
            .zip(v.typ().iter())
            .map(|((output_value, output_inplace), output_typ)| {
                let output_obj = output_value.object_id();
                let output_reg = code.alloc_register_id(lower_typ(output_typ, output_value));

                if output_inplace.is_none() {
                    if DEBUG {
                        println!(
                            "[MP.plan] {:?} in {:?} is output of {:?}",
                            output_value, output_reg, vid
                        )
                    }

                    let size = imctx.obj_size(output_obj);
                    allocate(
                        output_value.device(),
                        output_reg,
                        output_obj,
                        imctx
                            .obj_gpu_next_use
                            .first_use_of(output_obj)
                            .unwrap_or(Instant(usize::MAX)),
                        size,
                        gpu_allocator,
                        code,
                        ctx,
                        &imctx,
                    )?;
                }

                Ok(())
            })
            .collect::<Result<Vec<_>, _>>()?;
    }

    // Allocate temporary space
    // This is last step of allocation, so we can use the assigned register directly, without worrying that
    // the underlying object is moved elsewhere
    let temp_register_obj =
        if let Some((temp_space_sizes, temp_space_device)) = cg.temporary_space_needed(vid, libs) {
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
                        gpu_allocator,
                        code,
                        ctx,
                        &imctx,
                    )?;
                    Ok((temp_register, temp_obj, temp_space_device))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Some(rs)
        } else {
            None
        };

    // Get registers for input values
    let input_registers = input_values
        .iter()
        .map(|(input_vv, input_typs)| match input_vv {
            VertexValue::Single(input_value) => {
                let r = register_for_object(
                    input_value.device(),
                    input_value.object_id(),
                    input_typs[0].clone(),
                    code,
                    ctx,
                )
                .ok_or_else(|| Error::VertexInputsAndOutputsNotAccommodated(vid))?;
                Ok(r)
            }
            VertexValue::Tuple(input_values) => {
                let rs = input_values
                    .iter()
                    .zip(input_typs.iter())
                    .map(|(input_value, input_typ)| {
                        let r = register_for_object(
                            input_value.device(),
                            input_value.object_id(),
                            input_typ.clone(),
                            code,
                            ctx,
                        )
                        .ok_or_else(|| Error::VertexInputsAndOutputsNotAccommodated(vid))?;
                        Ok(r)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let tuple_reg = code.alloc_register_id(Typ::Tuple);
                ctx.reg_device.insert(tuple_reg, DeterminedDevice::Stack);
                code.emit(Instruction::new_no_src(InstructionNode::Tuple {
                    id: tuple_reg,
                    oprands: rs,
                }));
                tuple_registers.push(tuple_reg);
                Ok(tuple_reg)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Get registers for output values
    // - If the output need no allocation, just assign a register for it
    // - Otherwise, for each output of the vertex, first assign new register, then
    //   - If it is inplace, inherit space from curresponding input
    //   - Otherwise, get a register for the object
    let output_registers: Vec<(RegisterId, Option<RegisterId>)> = if v.node().no_allocate_output() {
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

                    if DEBUG {
                        println!(
                            "[MP.plan] {:?} in {:?} replacing {:?} at {:?} is output of {:?}",
                            output_value, output_reg, inplace_reg, inplace, vid
                        );
                    }

                    if output_value.device() == DeterminedDevice::Gpu {
                        if let Some(&addr_id) = ctx.gpu_reg2addr.get(&inplace_reg) {
                            ctx.set_gpu_addr_id_for_obj(output_obj, addr_id);
                            ctx.set_gpu_addr_id_for_reg(output_reg, addr_id);
                        } else {
                            return Err(Error::VertexInputsAndOutputsNotAccommodated(vid));
                        }
                    }

                    ctx.add_residence_for_object(output_obj, output_reg, output_value.device());

                    Ok((output_reg, Some(inplace_reg)))
                } else {
                    if DEBUG {
                        println!(
                            "[MP.plan] {:?} in {:?} is output of {:?}",
                            output_value, output_reg, vid
                        )
                    }

                    let output_reg = register_for_object(
                        output_value.device(),
                        output_obj,
                        lower_typ(output_typ, output_value),
                        code,
                        ctx,
                    )
                    .ok_or_else(|| Error::VertexInputsAndOutputsNotAccommodated(vid))?;
                    Ok((output_reg, None))
                }
            })
            .collect::<Result<Vec<_>, _>>()?
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
            ids: output_registers.clone(),
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
        deallocate(temp_device, temp_obj, gpu_allocator, code, ctx);
    }

    // We can't deallocate those objects that dies exactly after return vertex, as they are returned.
    // Though this doesn't matter as the runtime should exit immediately at seeing a return instruction.
    if v.node().is_return() {
        return Ok(true);
    }

    // Deallocate used tuple register
    for tuple_reg in tuple_registers.into_iter() {
        code.emit(Instruction::new_no_src(InstructionNode::StackFree {
            id: tuple_reg,
        }));
    }

    // Deallocate dead register
    deallocate_dead_objects(
        imctx.obj_dies_after_reversed.after(vid),
        gpu_allocator,
        code,
        ctx,
    );

    // Update next_uses on GPU integral allocator
    for (obj, next_use) in updated_next_uses {
        let _updated = gpu_allocator.update_next_use(obj, next_use, ctx);
    }

    Ok(false)
}

pub fn plan<'s, Rt: RuntimeType>(
    capacity: u64,
    smithereens_space: u64,
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
    mut obj_id_allocator: IdAllocator<ObjectId>,
    mut libs: Libs,
) -> Result<Chunk<'s, Rt>, Error> {
    let integral_sizes = collect_integral_sizes(cg, &g, &mut libs);

    if DEBUG {
        println!("[MP.plan] Integral sizes: {:?}", &integral_sizes);
    }

    let (ispace, sspace) = (capacity - smithereens_space, smithereens_space);

    if DEBUG {
        println!("[MP.plan] Integral space: {:?}", &ispace);
        println!("[MP.plan] Smithereens space: {:?}", &sspace);
    }

    let mut gpu_allocator = GpuAllocator {
        ialloc: integral_allocator::regretting::Allocator::new(ispace, integral_sizes),
        salloc: smithereens_allocator::Allocator::new(sspace),
    };

    let mut ctx = Context {
        gpu_obj2addr: Bijection::new(),
        gpu_addr_mapping: AddrMapping::new(),
        gpu_reg2addr: BTreeMap::new(),
        object_residence: BTreeMap::new(),
        reg_device: BTreeMap::new(),
        reg2mb: BTreeMap::new(),
        inplace_obj: BTreeMap::new(),
    };
    let mut register_types = BTreeMap::new();
    let imctx = ImmutableContext {
        obj_def,
        obj_dies_after,
        obj_dies_after_reversed,
        obj_gpu_next_use,
        vertex_inputs,
    };

    let mut code = Code::new();

    for ((i, &vid), updated_next_uses) in seq
        .iter()
        .enumerate()
        .zip(imctx.obj_gpu_next_use.iter_updates())
    {
        if g.vertex(vid).is_virtual() {
            continue;
        }

        let now = Instant(i);
        gpu_allocator.tick(now);
        let exe_device = devices[&vid];

        if DEBUG {
            println!(
                "[MP.plan] Scheduling {:?} on {:?} at {:?}",
                vid, exe_device, now
            );
        }

        if plan_vertex(
            vid,
            exe_device,
            now,
            updated_next_uses,
            cg,
            g,
            &mut ctx,
            &mut register_types,
            &imctx,
            &mut code,
            &mut gpu_allocator,
            uf_table,
            &mut obj_id_allocator,
            &mut libs,
        )
        .map_err(|e| e.try_with_vid(vid))?
        {
            break;
        }
    }

    let (register_types, reg_id_allocator) = code.1.freeze();

    Ok(Chunk {
        instructions: code.0,
        register_types,
        register_devices: ctx.reg_device,
        gpu_addr_mapping: ctx.gpu_addr_mapping,
        reg_id_allocator,
        reg_memory_blocks: ctx.reg2mb,
        obj_id_allocator,
        libs,
        _phantom: PhantomData,
    })
}
