use std::collections::{BTreeMap, BTreeSet};

use zkpoly_common::{
    bijection::Bijection,
    define_usize_id,
    digraph::internal::Predecessors,
    heap::{Heap, IdAllocator, UsizeId},
};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::type3::{Chunk, DeviceSpecific};

use super::{
    super::type3::{
        Addr, AddrId, AddrMapping, Device as DeterminedDevice, Instruction, InstructionNode,
        IntegralSize, RegisterId, Size, SmithereenSize,
    },
    user_function,
};

use super::{Cg, Device, VertexId};
use object_analysis::{ObjectId, ObjectsDefUse, ObjectsDieAfter, Value};

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
struct Instant(usize);

mod integral_allocator;
mod object_analysis;
mod smithereens_allocator;

trait AddrMappingHandler {
    fn add(&mut self, addr: Addr, size: Size) -> AddrId;
    fn update(&mut self, id: AddrId, addr: Addr, size: Size);
    fn get(&self, id: AddrId) -> (Addr, Size);
}

struct GpuAddrMappingHandler<'m>(&'m mut AddrMapping);

impl<'m> AddrMappingHandler for GpuAddrMappingHandler<'m> {
    fn add(&mut self, addr: Addr, size: Size) -> AddrId {
        self.0.push((addr, size))
    }

    fn update(&mut self, id: AddrId, addr: Addr, size: Size) {
        self.0[id] = (addr, size);
    }

    fn get(&self, id: AddrId) -> (Addr, Size) {
        self.0[id]
    }
}

fn collect_integral_sizes<'s, Rt: RuntimeType>(cg: &Cg<'s, Rt>) -> Vec<IntegralSize> {
    let mut integral_sizes = BTreeSet::<IntegralSize>::new();
    for vid in cg.g.vertices() {
        let v = cg.g.vertex(vid);
        for &size in v.typ().size().iter() {
            if let Ok(is) = IntegralSize::try_from(SmithereenSize(size)) {
                integral_sizes.insert(is);
            }
        }
        if let Some((temp_size, _)) = v.temporary_space_needed() {
            if let Size::Integral(is) = normalize_size(Size::Smithereen(SmithereenSize(temp_size)))
            {
                integral_sizes.insert(is);
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

fn collect_next_uses(
    successors: &Heap<VertexId, BTreeSet<VertexId>>,
    seq: &[VertexId],
) -> Vec<BTreeMap<VertexId, Instant>> {
    let mut next_uses: BTreeMap<VertexId, BTreeMap<Instant, Instant>> = BTreeMap::new();

    let seq_num = seq
        .iter()
        .cloned()
        .enumerate()
        .fold(BTreeMap::new(), |mut seq_num, (i, vid)| {
            seq_num.insert(vid, Instant(i));
            seq_num
        });

    for &vid in seq.iter() {
        for (&vid1, &vid2) in successors[vid].iter().zip(successors[vid].iter().skip(1)) {
            let instant1 = seq_num[&vid1];
            let instant2 = seq_num[&vid2];
            next_uses.entry(vid).or_default().insert(instant1, instant2);
        }
    }

    let mut r = vec![BTreeMap::new(); seq.len()];

    next_uses.into_iter().for_each(|(vid, chain)| {
        chain
            .into_iter()
            .for_each(|(instant1, instant2)| *r[instant1.0].get_mut(&vid).unwrap() = instant2);
    });
    r
}

pub type RegisterPlaces = DeviceSpecific<BTreeSet<RegisterId>>;

impl RegisterPlaces {
    pub fn empty() -> Self {
        Self::default()
    }
}

impl RegisterPlaces {
    pub fn get_any(&self) -> RegisterId {
        self.gpu
            .first()
            .or_else(|| self.cpu.first())
            .or_else(|| self.stack.first())
            .cloned()
            .expect("at least some register is expected to be available")
    }
}

#[derive(Debug)]
struct Context {
    gpu_reg2addr: Bijection<RegisterId, AddrId>,
    gpu_addr_mapping: AddrMapping,
    object_residence: BTreeMap<ObjectId, RegisterPlaces>,
    /// Tuple registers will not be included
    reg_device: BTreeMap<RegisterId, DeterminedDevice>,
    reg2obj: BTreeMap<RegisterId, ObjectId>,
}

#[derive(Debug)]
struct ImmutableContext<'os> {
    obj_sizes: &'os BTreeMap<ObjectId, Size>,
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
}

struct Code<'s>(Vec<Instruction<'s>>, IdAllocator<RegisterId>);

impl<'s> Code<'s> {
    pub fn new() -> Self {
        Self(Vec::new(), IdAllocator::new())
    }

    pub fn alloc_register_id(&mut self) -> RegisterId {
        self.1.alloc()
    }

    pub fn emit(&mut self, inst: Instruction<'s>) {
        self.0.push(inst);
    }
}

struct GpuAllocator {
    ialloc: integral_allocator::regretting::Allocator,
    salloc: smithereens_allocator::Allocator,
}

fn move_victims(victims: &[AddrId], code: &mut Code, ctx: &mut Context, imctx: &ImmutableContext) {
    victims.iter().for_each(|&victim| {
        let reg_id = ctx.gpu_reg2addr.get_backward(&victim).cloned().unwrap();
        let obj_id = ctx.reg2obj[&reg_id];

        if ctx.object_residence[&obj_id].cpu.is_empty() {
            let new_reg = code.alloc_register_id();
            let size = imctx.obj_sizes[&obj_id];
            allocate_cpu(obj_id, new_reg, size, code, ctx);
            code.emit(Instruction::new_no_src(InstructionNode::Transfer {
                id: new_reg,
                from: reg_id,
                rot: 0,
            }));
        }
    });
}

fn allocate_gpu_integral(
    size: IntegralSize,
    obj_id: ObjectId,
    reg_id: RegisterId,
    next_use: Instant,
    gpu_ialloc: &mut integral_allocator::regretting::Allocator,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
) -> AddrId {
    let addr = gpu_ialloc.allocate(
        size,
        next_use,
        &mut GpuAddrMappingHandler(&mut ctx.gpu_addr_mapping),
    );

    if let Some((transfers, addr)) = addr {
        for t in transfers {
            let reg_from = ctx.gpu_reg2addr.get_backward(&t.from).copied().unwrap();
            let reg_to = ctx.gpu_reg2addr.get_backward(&t.to).copied().unwrap();

            code.emit(Instruction::new_no_src(InstructionNode::Transfer {
                id: reg_to,
                from: reg_from,
                rot: 0,
            }));
        }
        return addr;
    }

    let (addr, victims) = gpu_ialloc.decide_and_realloc_victim(
        size,
        next_use,
        &mut GpuAddrMappingHandler(&mut ctx.gpu_addr_mapping),
    );
    move_victims(&victims, code, ctx, imctx);

    addr
}

#[derive(Debug)]
pub struct InsufficientSmithereenSpace;

fn allocate_gpu_smithereen(
    size: SmithereenSize,
    gpu_salloc: &mut smithereens_allocator::Allocator,
    mapping: &mut AddrMapping,
) -> Result<AddrId, InsufficientSmithereenSpace> {
    if let Some(addr) = gpu_salloc.allocate(size, &mut GpuAddrMappingHandler(mapping)) {
        return Ok(addr);
    }

    Err(InsufficientSmithereenSpace)
}

fn whether_ceil_smithereen_to_integral(size: SmithereenSize) -> bool {
    size.0 >= SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD
}

impl GpuAllocator {
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
        let addr = match normalize_size(size) {
            Size::Integral(size) => allocate_gpu_integral(
                size,
                obj_id,
                reg_id,
                next_use,
                &mut self.ialloc,
                code,
                ctx,
                imctx,
            ),
            Size::Smithereen(size) => {
                allocate_gpu_smithereen(size, &mut self.salloc, &mut ctx.gpu_addr_mapping)?
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
            Size::Integral(..) => self
                .ialloc
                .deallocate(addr, &mut GpuAddrMappingHandler(&mut ctx.gpu_addr_mapping)),
            Size::Smithereen(..) => self
                .salloc
                .deallocate(addr, &mut GpuAddrMappingHandler(&mut ctx.gpu_addr_mapping)),
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
}

fn allocate_cpu(obj_id: ObjectId, id: RegisterId, size: Size, code: &mut Code, ctx: &mut Context) {
    code.emit(Instruction::new_no_src(InstructionNode::CpuMalloc {
        id,
        size,
    }));

    ctx.add_residence_for_object(obj_id, id, DeterminedDevice::Cpu);
}

fn allocate_stack(
    obj_id: ObjectId,
    id: RegisterId,
    size: Size,
    code: &mut Code,
    ctx: &mut Context,
) {
    code.emit(Instruction::new_no_src(InstructionNode::StackAlloc { id }));

    ctx.add_residence_for_object(obj_id, id, DeterminedDevice::Stack);
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
        DeterminedDevice::Stack => allocate_stack(obj_id, reg_id, size, code, ctx),
    };
    Ok(())
}

fn deallocate_cpu(obj_id: ObjectId, code: &mut Code, ctx: &mut Context) {
    let reg_id = ctx
        .pop_residence_of_object(obj_id, DeterminedDevice::Cpu)
        .unwrap();

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

fn decide_device<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
    successors: &Heap<VertexId, BTreeSet<VertexId>>,
) -> BTreeMap<VertexId, DeterminedDevice> {
    let mut devices = BTreeMap::new();

    for (vid, v) in cg.g.topology_sort() {
        let device = match v.device() {
            Device::PreferGpu => {
                if successors[vid]
                    .iter()
                    .any(|&vid| cg.g.vertex(vid).device() == Device::Gpu)
                    || cg
                        .g
                        .vertex(vid)
                        .predecessors()
                        .any(|vid| devices[&vid] == DeterminedDevice::Cpu)
                {
                    DeterminedDevice::Gpu
                } else {
                    DeterminedDevice::Cpu
                }
            }
            Device::Gpu => DeterminedDevice::Gpu,
            Device::Cpu => DeterminedDevice::Cpu,
        };

        let device = match device {
            DeterminedDevice::Cpu => {
                if v.typ().stack_allocable() {
                    DeterminedDevice::Stack
                } else {
                    DeterminedDevice::Cpu
                }
            }
            DeterminedDevice::Gpu => DeterminedDevice::Gpu,
            DeterminedDevice::Stack => unreachable!(),
        };

        devices.insert(vid, device);
    }

    devices
}

fn ensure_on_device(
    device: DeterminedDevice,
    obj_id: ObjectId,
    next_use: Instant,
    gpu_allocator: &mut GpuAllocator,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
) -> Result<RegisterId, InsufficientSmithereenSpace> {
    if let Some(reg_id) = ctx.object_residence[&obj_id].get_device(device).first() {
        Ok(*reg_id)
    } else {
        let new_reg = code.alloc_register_id();
        let size = imctx.obj_sizes[&obj_id];
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
        code.emit(Instruction::new_no_src(InstructionNode::Transfer {
            id: new_reg,
            from: ctx.object_residence[&obj_id].get_any(),
            rot: 0,
        }));
        Ok(new_reg)
    }
}

fn ensure_copied(
    device: DeterminedDevice,
    vid: VertexId,
    original_obj_id: ObjectId,
    obj_id: ObjectId,
    now: Instant,
    gpu_allocator: &mut GpuAllocator,
    code: &mut Code,
    ctx: &mut Context,
    imctx: &ImmutableContext,
    obj_dies_after: &ObjectsDieAfter,
) -> Result<RegisterId, InsufficientSmithereenSpace> {
    let need_copy = obj_dies_after.get_device(device)[&original_obj_id] != vid;
    let input_reg = if need_copy {
        let transferred_reg = ensure_on_device(
            device,
            original_obj_id,
            now,
            gpu_allocator,
            code,
            ctx,
            imctx,
        )?;
        let copied_reg = code.alloc_register_id();
        let size = imctx.obj_sizes[&original_obj_id];
        allocate(
            device,
            copied_reg,
            obj_id,
            now,
            size,
            gpu_allocator,
            code,
            ctx,
            &imctx,
        )?;
        code.emit(Instruction::new_no_src(InstructionNode::Transfer {
            id: copied_reg,
            from: transferred_reg,
            rot: 0,
        }));
        copied_reg
    } else {
        let on_device_reg = ensure_on_device(
            device,
            original_obj_id,
            now,
            gpu_allocator,
            code,
            ctx,
            imctx,
        )?;
        let moved_reg = code.alloc_register_id();
        code.emit(Instruction::new_no_src(InstructionNode::Move {
            id: moved_reg,
            from: on_device_reg,
        }));
        ctx.remove_residence_for_object(original_obj_id, device);
        ctx.add_residence_for_object(obj_id, moved_reg, device);
        moved_reg
    };
    Ok(input_reg)
}

fn set_rotation_and_slice(
    value: &Value,
    obj_id: ObjectId,
    device: DeterminedDevice,
    reg_id: RegisterId,
    code: &mut Code,
    ctx: &mut Context,
) -> RegisterId {
    match value {
        Value::Poly {
            rotation, slice, ..
        } => {
            let new_reg = code.alloc_register_id();
            code.emit(Instruction::new_no_src(InstructionNode::RotateAndSlice {
                id: new_reg,
                operand: reg_id,
                rot: *rotation,
                slice: *slice,
            }));
            ctx.add_residence_for_object(obj_id, new_reg, device);
            new_reg
        }
        _ => reg_id,
    }
}

pub fn plan<'s, Rt: RuntimeType>(
    capacity: u64,
    cg: &Cg<'s, Rt>,
    seq: &[VertexId],
    uf_table: &user_function::Table<Rt>,
) -> Result<Chunk<'s, Rt>, InsufficientSmithereenSpace> {
    let integral_sizes = collect_integral_sizes(cg);
    let (ispace, sspace) =
        divide_integral_smithereens(capacity, integral_sizes.last().unwrap().clone());

    let successors = cg.g.successors();
    let next_ueses = collect_next_uses(&successors, seq);
    let devices = decide_device(cg, &successors);
    let (obj_def_use, mut obj_id_allocator) = object_analysis::analyze_def_use(cg);
    let obj_dies_after = object_analysis::analyze_die_after(cg, seq, &devices, &obj_def_use);
    let obj_dies_after_reversed = obj_dies_after.reversed();

    let mut gpu_allocator = GpuAllocator {
        ialloc: integral_allocator::regretting::Allocator::new(ispace, integral_sizes),
        salloc: smithereens_allocator::Allocator::new(sspace),
    };

    let mut ctx = Context {
        gpu_reg2addr: Bijection::new(),
        gpu_addr_mapping: AddrMapping::new(),
        object_residence: BTreeMap::new(),
        reg_device: BTreeMap::new(),
        reg2obj: BTreeMap::new(),
    };
    let mut register_types = BTreeMap::new();
    let imctx = ImmutableContext {
        obj_sizes: &obj_def_use.sizes,
    };

    let mut code = Code::new();

    for ((i, &vid), updated_next_uses) in seq.iter().enumerate().zip(next_ueses.into_iter()) {
        if cg.g.vertex(vid).is_virtual() {
            continue;
        }

        let now = Instant(i);
        gpu_allocator.tick(now);
        let device = devices[&vid];

        // Prepare inputs
        // - Move input objects to desired device
        // - Make copies, if input is mutated and later used on the same device
        //   - If the input's space is used inplace by an output, the copy's object ID is that of the output
        //   - Otherwise, a new temporary object ID is allocated
        // - Build tuples, if needed
        let mutable_uses: Vec<VertexId> = cg.g.vertex(vid).mutable_uses(uf_table).collect();
        let outputs_inplace: Vec<Option<VertexId>> =
            cg.g.vertex(vid).outputs_inplace(uf_table, device).collect();
        let input_registers: Vec<RegisterId> = cg
            .g
            .vertex(vid)
            .uses()
            .map(|input_vid| {
                if mutable_uses.contains(&input_vid) && outputs_inplace.contains(&Some(input_vid)) {
                    // This input's space is used inplace by an output
                    let ouput_obj = obj_def_use.values[&vid]
                        .object_ids()
                        .nth(
                            outputs_inplace
                                .iter()
                                .position(|&x| x == Some(input_vid))
                                .unwrap(),
                        )
                        .unwrap();
                    let input_value = match &obj_def_use.values[&input_vid] {
                        object_analysis::VertexValue::Single(input_value) => input_value.clone(),
                        object_analysis::VertexValue::Tuple(..) => {
                            panic!("an inplace input must not be a tuple")
                        }
                    };
                    let copied_reg = ensure_copied(
                        device,
                        vid,
                        input_value.object_id(),
                        ouput_obj,
                        updated_next_uses[&vid],
                        &mut gpu_allocator,
                        &mut code,
                        &mut ctx,
                        &imctx,
                        &obj_dies_after,
                    )?;
                    Ok(set_rotation_and_slice(
                        &input_value,
                        ouput_obj,
                        device,
                        copied_reg,
                        &mut code,
                        &mut ctx,
                    ))
                } else if mutable_uses.contains(&input_vid) {
                    let input_value = match &obj_def_use.values[&input_vid] {
                        object_analysis::VertexValue::Single(input_value) => input_value.clone(),
                        object_analysis::VertexValue::Tuple(..) => {
                            panic!("an mutable input must not be a tuple")
                        }
                    };
                    // This input is mutated
                    let temp_obj = obj_id_allocator.alloc();
                    let copied_reg = ensure_copied(
                        device,
                        vid,
                        input_value.object_id(),
                        temp_obj,
                        now,
                        &mut gpu_allocator,
                        &mut code,
                        &mut ctx,
                        &imctx,
                        &obj_dies_after,
                    );
                    Ok(set_rotation_and_slice(
                        &input_value,
                        temp_obj,
                        device,
                        copied_reg?,
                        &mut code,
                        &mut ctx,
                    ))
                } else if !mutable_uses.contains(&input_vid)
                    && !outputs_inplace.contains(&Some(input_vid))
                {
                    // The input is not mutated
                    match &obj_def_use.values[&input_vid] {
                        object_analysis::VertexValue::Single(input_value) => {
                            ensure_on_device(
                                device,
                                input_value.object_id(),
                                now,
                                &mut gpu_allocator,
                                &mut code,
                                &mut ctx,
                                &imctx,
                            )
                        }
                        object_analysis::VertexValue::Tuple(input_values) => {
                            let elements = input_values
                                .iter()
                                .map(|input_value| {
                                    ensure_on_device(
                                        device,
                                        input_value.object_id(),
                                        now,
                                        &mut gpu_allocator,
                                        &mut code,
                                        &mut ctx,
                                        &imctx,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let tuple_reg = code.alloc_register_id();
                            code.emit(Instruction::new_no_src(InstructionNode::Tuple {
                                id: tuple_reg,
                                oprands: elements,
                            }));
                            Ok(tuple_reg)
                        }
                    }
                } else {
                    panic!("an inplace output must use an mutable input")
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Allocate outputs
        // - Use correspounding input registe if inplace.
        //   In this case, the object ID of the register is already the output object ID.
        let v = cg.g.vertex(vid);
        let output_registers: Vec<RegisterId> = obj_def_use.values[&vid]
            .object_ids()
            .zip(v.outputs_inplace(uf_table, device))
            .map(|(output_obj, inplace)| {
                if let Some(input_vid) = inplace {
                    let index = v.uses().position(|x| x == input_vid).unwrap();
                    Ok(input_registers[index])
                } else {
                    let output_reg = code.alloc_register_id();
                    let size = imctx.obj_sizes[&output_obj];
                    allocate(
                        device,
                        output_reg,
                        output_obj,
                        updated_next_uses[&vid],
                        size,
                        &mut gpu_allocator,
                        &mut code,
                        &mut ctx,
                        &imctx,
                    )?;
                    Ok(output_reg)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let temp_register_obj =
            if let Some((temp_space_size, temp_space_device)) = v.temporary_space_needed() {
                let temp_register = code.alloc_register_id();
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
                Some((temp_register, temp_obj))
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
            .for_each(|(&output_reg, typ)| {
                register_types.insert(output_reg, typ.clone());
            });

        code.emit(Instruction::new(
            InstructionNode::Type2 {
                ids: output_registers,
                temp: temp_register_obj.map(|(x, _)| x),
                vertex: v3,
            },
            v.src().clone(),
        ));

        // Deallocate temporary space
        if let Some((_, temp_obj)) = temp_register_obj {
            deallocate(device, temp_obj, &mut gpu_allocator, &mut code, &mut ctx);
        }

        // Deallocate dead register
        // - Deallocate dead GPU register right away
        // - Non-GPU register are deallocated if they are never used by any device
        for (&dead_obj, device_collection) in obj_dies_after_reversed.after[&vid].iter() {
            if device_collection.gpu() {
                gpu_allocator.deallocate(dead_obj, &mut code, &mut ctx);
            }

            let gpu_alive = !ctx.object_residence[&dead_obj]
                .get_device(DeterminedDevice::Gpu)
                .is_empty();
            if device_collection.cpu() && gpu_alive {
                deallocate_cpu(dead_obj, &mut code, &mut ctx);
            }
            if device_collection.stack() && gpu_alive {
                deallocate_stack(dead_obj, &mut code, &mut ctx);
            }
        }
    }

    Ok(Chunk {
        instructions: code.0,
        register_types,
        register_devices: ctx.reg_device,
        gpu_addr_mapping: ctx.gpu_addr_mapping,
    })
}
