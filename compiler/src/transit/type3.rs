use std::{collections::BTreeMap, marker::PhantomData};

use crate::transit::{
    self,
    type2::{self, memory_planning::MemoryBlock},
};
use zkpoly_common::{
    arith, define_usize_id,
    heap::{Heap, IdAllocator, RoHeap},
    load_dynamic::Libs,
};
use zkpoly_runtime::args::RuntimeType;

use super::type2::object_analysis::ObjectId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    Gpu(usize),
    Cpu,
    Stack,
}

impl Device {
    pub fn iter(n_gpus: usize) -> impl Iterator<Item = Device> {
        (0..n_gpus)
            .map(Self::Gpu)
            .chain([Device::Cpu, Device::Stack].into_iter())
    }

    /// Returns the parent memory device of current one, if any.
    /// Cold objects are rejected to a memory device's parent device.
    pub fn parent(self) -> Option<Self> {
        use Device::*;
        match self {
            Gpu(..) => Some(Cpu),
            Cpu => None,
            Stack => None,
        }
    }

    pub fn for_execution_on(device: type2::Device) -> Self {
        use type2::Device::*;
        match device {
            Gpu(i) => Device::Gpu(i),
            Cpu => Device::Cpu,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceSpecific<T> {
    pub gpu: Vec<T>,
    pub cpu: T,
    pub stack: T,
}

impl<T> DeviceSpecific<T> {
    pub fn get_device(&self, device: Device) -> &T {
        match device {
            Device::Gpu(i) => &self.gpu[i],
            Device::Cpu => &self.cpu,
            Device::Stack => &self.stack,
        }
    }

    pub fn get_device_mut(&mut self, device: Device) -> &mut T {
        match device {
            Device::Gpu(i) => &mut self.gpu[i],
            Device::Cpu => &mut self.cpu,
            Device::Stack => &mut self.stack,
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> DeviceSpecific<U> {
        DeviceSpecific {
            gpu: self.gpu.into_iter().map(&mut f).collect(),
            cpu: (&mut f)(self.cpu),
            stack: f(self.stack),
        }
    }

    pub fn default(n_gpus: usize) -> Self
    where
        T: Default,
    {
        DeviceSpecific {
            gpu: (0..n_gpus).map(|_| Default::default()).collect(),
            cpu: T::default(),
            stack: T::default(),
        }
    }
}

impl std::ops::Sub<Self> for DeviceSpecific<bool> {
    type Output = Self;
    fn sub(self, rhs: Self::Output) -> Self::Output {
        DeviceSpecific {
            gpu: self
                .gpu
                .into_iter()
                .zip(rhs.gpu)
                .map(|(a, b)| a && !b)
                .collect(),
            cpu: self.cpu && !rhs.cpu,
            stack: self.stack && !rhs.stack,
        }
    }
}

define_usize_id!(RegisterId);

pub mod template {
    use zkpoly_common::typ::PolyType;
    use zkpoly_runtime::instructions::AllocMethod;

    use crate::{ast::PolyInit, transit::type2};

    use type2::object_analysis::size::Size;

    #[derive(Debug, Clone)]
    pub enum InstructionNode<I, V> {
        Type2 {
            /// (a, Some(b)) says that a use b inplace, so we need move b to a when lowering this to
            /// runtime instruction.
            /// (a, None) syas that a needs to be newly allocated, and there needs no move.
            ids: Vec<(I, Option<I>)>,
            temp: Vec<I>,
            vertex: V,
            vid: type2::VertexId,
        },
        GpuMalloc {
            id: I,
            addr: AllocMethod,
        },
        GpuFree {
            id: I,
        },
        CpuMalloc {
            id: I,
            addr: AllocMethod,
        },
        CpuFree {
            id: I,
        },
        StackFree {
            id: I,
        },
        Tuple {
            id: I,
            oprands: Vec<I>,
        },
        Transfer {
            id: I,
            from: I,
        },
        /// Transfer `from` to `to`, but `to` is already defined.
        /// Then `to` is copied to `id`, to indicate that the transfer has completed.
        TransferToDefed {
            id: I,
            to: I,
            from: I,
        },
        SetPolyMeta {
            id: I,
            from: I,
            offset: usize,
            len: usize,
        },
        FillPoly {
            id: I,
            operand: I,
            deg: usize,
            init: PolyInit,
            pty: PolyType,
        },
    }

    impl<I, V> InstructionNode<I, V>
    where
        I: Copy,
    {
        pub fn ids<'s>(&'s self) -> Box<dyn Iterator<Item = I> + 's> {
            use InstructionNode::*;
            match self {
                Type2 { ids, .. } => Box::new(ids.iter().map(|(x, _)| *x)),
                GpuMalloc { id, .. } => Box::new(std::iter::once(*id)),
                GpuFree { .. } => Box::new(std::iter::empty()),
                CpuMalloc { id, .. } => Box::new(std::iter::once(*id)),
                CpuFree { .. } => Box::new(std::iter::empty()),
                StackFree { .. } => Box::new(std::iter::empty()),
                Tuple { id, .. } => Box::new(std::iter::once(*id)),
                Transfer { id, .. } => Box::new(std::iter::once(*id)),
                TransferToDefed { id, .. } => Box::new(std::iter::once(*id)),
                SetPolyMeta { id, .. } => Box::new(std::iter::once(*id)),
                FillPoly { id, .. } => Box::new(std::iter::once(*id)),
            }
        }

        pub fn ids_mut<'s>(&'s mut self) -> Box<dyn Iterator<Item = &'s mut I> + 's> {
            use InstructionNode::*;
            match self {
                Type2 { ids, .. } => Box::new(ids.iter_mut().map(|(x, _)| x)),
                GpuMalloc { id, .. } => Box::new(std::iter::once(id)),
                GpuFree { .. } => Box::new(std::iter::empty()),
                CpuMalloc { id, .. } => Box::new(std::iter::once(id)),
                CpuFree { .. } => Box::new(std::iter::empty()),
                StackFree { .. } => Box::new(std::iter::empty()),
                Tuple { id, .. } => Box::new(std::iter::once(id)),
                TransferToDefed { id, .. } => Box::new(std::iter::once(id)),
                Transfer { id, .. } => Box::new(std::iter::once(id)),
                SetPolyMeta { id, .. } => Box::new(std::iter::once(id)),
                FillPoly { id, .. } => Box::new(std::iter::once(id)),
            }
        }

        /// For each pair (a, b), after the instruction is executed, b's space is used in a for another purpose,
        /// and b can no longer be used syntactically correctly.
        pub fn ids_inplace<'s>(&'s self) -> Box<dyn Iterator<Item = (I, Option<I>)> + 's> {
            use InstructionNode::*;
            match self {
                Type2 { ids, .. } => Box::new(ids.iter().cloned()),
                GpuMalloc { id, .. } => Box::new(std::iter::once((*id, None))),
                GpuFree { .. } => Box::new(std::iter::empty()),
                CpuMalloc { id, .. } => Box::new(std::iter::once((*id, None))),
                CpuFree { .. } => Box::new(std::iter::empty()),
                StackFree { .. } => Box::new(std::iter::empty()),
                Tuple { id, .. } => Box::new(std::iter::once((*id, None))),
                Transfer { id, .. } => Box::new(std::iter::once((*id, None))),
                TransferToDefed { id, to, .. } => Box::new(std::iter::once((*id, Some(*to)))),
                SetPolyMeta { id, .. } => Box::new(std::iter::once((*id, None))),
                FillPoly { id, operand, .. } => Box::new(std::iter::once((*id, Some(*operand)))),
            }
        }

        pub fn is_allloc(&self) -> bool {
            use InstructionNode::*;
            match self {
                GpuMalloc { .. } => true,
                CpuMalloc { .. } => true,
                _ => false,
            }
        }

        pub fn is_dealloc(&self) -> bool {
            use InstructionNode::*;
            match self {
                GpuFree { .. } => true,
                CpuFree { .. } => true,
                StackFree { .. } => true,
                _ => false,
            }
        }
    }
}

pub type VertexNode = type2::template::VertexNode<
    RegisterId,
    arith::ArithGraph<RegisterId, arith::ExprId>,
    type2::ConstantId,
    type2::user_function::Id,
>;

pub type InstructionNode = template::InstructionNode<RegisterId, VertexNode>;

#[derive(Debug, Clone)]
pub struct Instruction<'s> {
    pub node: InstructionNode,
    pub src: Option<transit::SourceInfo<'s>>,
}

impl<'s> Instruction<'s> {
    pub fn new(node: InstructionNode, src: transit::SourceInfo<'s>) -> Self {
        Self {
            node,
            src: Some(src),
        }
    }
    pub fn new_no_src(node: InstructionNode) -> Self {
        Self { node, src: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Track {
    MemoryManagement,
    CoProcess,
    Gpu(usize),
    Cpu,
    ToGpu,
    FromGpu,
    GpuMemory(usize),
}

impl Track {
    pub fn is_gpu(&self) -> bool {
        lowering::Stream::of_track(*self).is_some()
    }

    pub fn is_cpu(&self) -> bool {
        lowering::Stream::of_track(*self).is_none()
    }
}

#[derive(Debug, Clone)]
pub struct TrackSpecific<T> {
    pub(crate) memory_management: T,
    pub(crate) co_process: T,
    pub(crate) gpu: Vec<T>,
    pub(crate) cpu: T,
    pub(crate) to_gpu: T,
    pub(crate) from_gpu: T,
    pub(crate) gpu_memory: Vec<T>,
}

impl<T> TrackSpecific<T> {
    pub fn default(n_gpus: usize) -> Self
    where
        T: Default,
    {
        Self {
            memory_management: T::default(),
            co_process: T::default(),
            gpu: (0..n_gpus).map(|_| T::default()).collect(),
            cpu: T::default(),
            to_gpu: T::default(),
            from_gpu: T::default(),
            gpu_memory: (0..n_gpus).map(|_| T::default()).collect(),
        }
    }

    pub fn get_track(&self, track: Track) -> &T {
        use Track::*;
        match track {
            MemoryManagement => &self.memory_management,
            CoProcess => &self.co_process,
            Gpu(i) => &self.gpu[i],
            Cpu => &self.cpu,
            ToGpu => &self.to_gpu,
            FromGpu => &self.from_gpu,
            GpuMemory(i) => &self.gpu_memory[i],
        }
    }

    pub fn get_track_mut(&mut self, track: Track) -> &mut T {
        use Track::*;
        match track {
            MemoryManagement => &mut self.memory_management,
            CoProcess => &mut self.co_process,
            Gpu(i) => &mut self.gpu[i],
            Cpu => &mut self.cpu,
            ToGpu => &mut self.to_gpu,
            FromGpu => &mut self.from_gpu,
            GpuMemory(i) => &mut self.gpu_memory[i],
        }
    }

    pub fn iter(&self, n_gpus: usize) -> impl Iterator<Item = (Track, &T)> {
        use Track::*;
        vec![
            (MemoryManagement, &self.memory_management),
            (CoProcess, &self.co_process),
            (Cpu, &self.cpu),
            (ToGpu, &self.to_gpu),
            (FromGpu, &self.from_gpu),
        ]
        .into_iter()
        .chain((0..n_gpus).map(|i| (Gpu(i), &self.gpu[i])))
        .chain((0..n_gpus).map(|i| (GpuMemory(i), &self.gpu_memory[i])))
    }

    pub fn new(t: T, n_gpus: usize) -> Self
    where
        T: Clone,
    {
        Self {
            memory_management: t.clone(),
            co_process: t.clone(),
            gpu: vec![t.clone(); n_gpus],
            cpu: t.clone(),
            to_gpu: t.clone(),
            from_gpu: t.clone(),
            gpu_memory: vec![t.clone(); n_gpus],
        }
    }
}

impl Track {
    pub fn on_device(device: type2::Device) -> Track {
        use type2::Device::*;
        match device {
            Gpu(i) => Track::Gpu(i),
            Cpu => Track::Cpu,
        }
    }
}

fn determine_transfer_track(from: Device, to: Device) -> Track {
    use Device::*;
    match (from, to) {
        (Gpu(..), Cpu) => Track::FromGpu,
        (Gpu(..), Stack) => Track::FromGpu,
        (Gpu(i), Gpu(j)) if (i == j) => Track::GpuMemory(i),
        (Gpu(_), Gpu(_)) => Track::ToGpu,
        (Cpu, Gpu(..)) => Track::ToGpu,
        (Cpu, Stack) => panic!("Cpu cannot transfer to Stack"),
        (Cpu, Cpu) => Track::Cpu,
        (Stack, Gpu(..)) => Track::ToGpu,
        (Stack, Cpu) => panic!("Stack cannot transfer to Cpu"),
        (Stack, Stack) => Track::Cpu,
    }
}

impl<'s> Instruction<'s> {
    pub fn track(&self, devices: impl Fn(RegisterId) -> Device) -> Track {
        use template::InstructionNode::*;
        use Track::*;

        let executor_of = |md| match md {
            Device::Cpu | Device::Stack => type2::Device::Cpu,
            Device::Gpu(i) => type2::Device::Gpu(i),
        };

        match &self.node {
            Type2 {
                vertex: type2::template::VertexNode::Return(..),
                ..
            } => MemoryManagement,
            Type2 { vertex, ids, .. } => vertex.track(executor_of(devices(ids[0].0))),
            GpuMalloc { .. } => MemoryManagement,
            GpuFree { .. } => MemoryManagement,
            CpuMalloc { .. } => MemoryManagement,
            CpuFree { .. } => MemoryManagement,
            StackFree { .. } => MemoryManagement,
            Tuple { .. } => Cpu,
            Transfer { from, id: to, .. } | TransferToDefed { to, from, .. } => {
                determine_transfer_track(devices(*from), devices(*to))
            }
            SetPolyMeta { .. } => Cpu,
            FillPoly { id, .. } => Track::on_device(executor_of(devices(*id))),
        }
    }

    pub fn defs<'a>(&'a self) -> Box<dyn Iterator<Item = RegisterId> + 'a> {
        self.node.ids()
    }

    pub fn uses<'a>(&'a self) -> Box<dyn Iterator<Item = RegisterId> + 'a> {
        use template::InstructionNode::*;

        match &self.node {
            Type2 { vertex, temp, .. } => Box::new(vertex.uses().chain(temp.iter().copied())),
            GpuFree { id } => Box::new(std::iter::once(*id)),
            CpuFree { id } => Box::new(std::iter::once(*id)),
            StackFree { id } => Box::new(std::iter::once(*id)),
            Tuple { oprands, .. } => Box::new(oprands.iter().copied()),
            Transfer { from, .. } => Box::new(std::iter::once(*from)),
            TransferToDefed { from, to, .. } => {
                Box::new(std::iter::once(*from).chain(std::iter::once(*to)))
            }
            SetPolyMeta { from, .. } => Box::new(std::iter::once(*from)),
            FillPoly { operand, .. } => Box::new(std::iter::once(*operand)),
            _ => Box::new(std::iter::empty()),
        }
    }

    pub fn mem_reads<'a>(&'a self) -> Box<dyn Iterator<Item = RegisterId> + 'a> {
        use template::InstructionNode::*;

        match &self.node {
            Type2 { vertex, .. } => {
                let mem_writes: Vec<RegisterId> = self.mem_writes().collect();
                Box::new(vertex.uses().filter(move |x| !mem_writes.contains(x)))
            }
            Transfer { from, .. } => Box::new(std::iter::once(*from)),
            TransferToDefed { from, to, .. } => {
                Box::new(std::iter::once(*from).chain(std::iter::once(*to)))
            }
            _ => Box::new(std::iter::empty()),
        }
    }

    pub fn mem_writes<'a>(&'a self) -> Box<dyn Iterator<Item = RegisterId> + 'a> {
        use template::InstructionNode::*;

        match &self.node {
            Type2 { ids, .. } => Box::new(ids.iter().filter_map(|(_, x)| x.clone())),
            _ => Box::new(std::iter::empty()),
        }
    }
}
define_usize_id!(InstructionIndex);

pub struct RegisterAllocator {
    register_types: Heap<RegisterId, typ::Typ>,
    register_devices: BTreeMap<RegisterId, Device>,
    register_memory_blocks: BTreeMap<RegisterId, MemoryBlock>,
    obj_id_allocator: IdAllocator<ObjectId>,
}

impl RegisterAllocator {
    fn alloc(&mut self, typ: typ::Typ, device: Device) -> RegisterId {
        let id = self.register_types.push(typ);
        self.register_devices.insert(id, device);
        id
    }

    pub fn inherit_memory_block(
        &mut self,
        typ: typ::Typ,
        device: Device,
        memory_from: RegisterId,
    ) -> RegisterId {
        let id = self.register_types.push(typ);
        self.register_devices.insert(id, device);

        let memory_block = self.register_memory_blocks[&memory_from].clone();
        self.register_memory_blocks.insert(id, memory_block);

        id
    }

    pub fn inherit(&mut self, r: RegisterId) -> RegisterId {
        let device = self.device_of(r);
        let typ = self.typ_of(r);
        let mb = self.register_memory_blocks[&r].clone();

        let r1 = self.alloc(typ.clone(), device);

        self.register_memory_blocks.insert(r1, mb);

        r1
    }

    pub fn device_of(&self, id: RegisterId) -> Device {
        self.register_devices[&id]
    }

    pub fn typ_of(&self, id: RegisterId) -> &typ::Typ {
        &self.register_types[id]
    }

    pub fn inherit_device_typ_address(&mut self, r: RegisterId) -> RegisterId {
        let device = self.device_of(r);
        let obj_id = self.obj_id_allocator.alloc();
        let typ = self.typ_of(r);
        let mb = self.register_memory_blocks[&r]
            .clone()
            .with_object_id(obj_id);

        let r1 = self.alloc(typ.clone(), device);

        self.register_memory_blocks.insert(r1, mb);

        r1
    }
}

#[derive(Debug)]
pub struct Chunk<'s, Rt: RuntimeType> {
    pub(crate) instructions: Vec<Instruction<'s>>,
    pub(crate) register_types: RoHeap<RegisterId, typ::Typ>,
    pub(crate) register_devices: BTreeMap<RegisterId, Device>,
    pub(crate) reg_id_allocator: IdAllocator<RegisterId>,
    pub(crate) reg_memory_blocks: BTreeMap<RegisterId, MemoryBlock>,
    pub(crate) obj_id_allocator: IdAllocator<ObjectId>,
    pub(crate) libs: Libs,
    pub(crate) _phantom: PhantomData<Rt>,
}

impl<'s, Rt: RuntimeType> std::ops::Index<InstructionIndex> for Chunk<'s, Rt> {
    type Output = Instruction<'s>;
    fn index(&self, index: InstructionIndex) -> &Self::Output {
        &self.instructions[index.0]
    }
}
impl<'s, Rt: RuntimeType> std::ops::IndexMut<InstructionIndex> for Chunk<'s, Rt> {
    fn index_mut(&mut self, index: InstructionIndex) -> &mut Self::Output {
        &mut self.instructions[index.0]
    }
}

impl<'s, Rt: RuntimeType> Chunk<'s, Rt> {
    pub fn iter_instructions(&self) -> impl Iterator<Item = (InstructionIndex, &Instruction<'s>)> {
        self.instructions
            .iter()
            .enumerate()
            .map(|(i, instr)| (InstructionIndex(i), instr))
    }

    pub fn iter_instructions_mut(
        &mut self,
    ) -> impl Iterator<Item = (InstructionIndex, &mut Instruction<'s>)> {
        self.instructions
            .iter_mut()
            .enumerate()
            .map(|(i, instr)| (InstructionIndex(i), instr))
    }

    pub fn rewrite_instructions<'a, I2>(
        &'a mut self,
        f: impl FnOnce(Box<dyn Iterator<Item = (InstructionIndex, Instruction<'s>)> + 's>) -> I2,
    ) where
        I2: Iterator<Item = Instruction<'s>>,
    {
        let insts = std::mem::take(&mut self.instructions);
        let insts = f(Box::new(
            insts
                .into_iter()
                .enumerate()
                .map(|(i, inst)| (InstructionIndex(i), inst)),
        ));
        let insts = insts.collect();
        self.instructions = insts;
    }

    pub fn assigned_at(&self) -> BTreeMap<RegisterId, InstructionIndex> {
        self.iter_instructions()
            .filter(|(_, inst)| !inst.node.is_allloc())
            .fold(BTreeMap::new(), |mut acc, (i, instr)| {
                for id in instr.defs() {
                    acc.insert(id, i);
                }
                acc
            })
    }

    pub fn malloc_at(&self) -> BTreeMap<RegisterId, InstructionIndex> {
        self.instructions
            .iter()
            .enumerate()
            .fold(BTreeMap::new(), |mut acc, (i, instr)| {
                use template::InstructionNode::*;
                match &instr.node {
                    GpuMalloc { id, .. } => {
                        acc.insert(*id, InstructionIndex(i));
                    }
                    CpuMalloc { id, .. } => {
                        acc.insert(*id, InstructionIndex(i));
                    }
                    _ => {}
                };
                acc
            })
    }

    pub fn use_not_deallocate_at(&self) -> BTreeMap<RegisterId, Vec<InstructionIndex>> {
        self.iter_instructions()
            .fold(BTreeMap::new(), |mut acc, (i, instr)| {
                for id in instr.uses() {
                    acc.entry(id).or_default().push(i);
                }
                acc
            })
    }

    pub fn take_reg_id_allocator(&mut self) -> IdAllocator<RegisterId> {
        let mut x = IdAllocator::new();
        std::mem::swap(&mut self.reg_id_allocator, &mut x);
        x
    }

    pub fn with_reg_id_allocator_taken(
        mut self,
        f: impl FnOnce(Self, RegisterAllocator) -> (Self, RegisterAllocator),
    ) -> Self {
        let reg_id_allocator = self.take_reg_id_allocator();
        let regster_types = std::mem::take(&mut self.register_types);
        let ra = RegisterAllocator {
            register_devices: std::mem::take(&mut self.register_devices),
            register_memory_blocks: std::mem::take(&mut self.reg_memory_blocks),
            obj_id_allocator: std::mem::take(&mut self.obj_id_allocator),
            register_types: regster_types.to_mutable(reg_id_allocator),
        };

        let (mut self2, ra) = f(self, ra);
        let (register_types, reg_id_allocator) = ra.register_types.freeze();

        self2.reg_id_allocator = reg_id_allocator;
        self2.register_types = register_types;
        self2.reg_memory_blocks = ra.register_memory_blocks;
        self2.register_devices = ra.register_devices;
        self2.obj_id_allocator = ra.obj_id_allocator;
        self2
    }

    pub fn take_libs(&mut self) -> Libs {
        let mut x = Libs::new();
        std::mem::swap(&mut self.libs, &mut x);
        x
    }
}

pub mod lowering;
pub mod pretty_print;
pub mod rewrite_extend;
pub mod track_splitting;
pub mod typ;
