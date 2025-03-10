use std::{collections::BTreeMap, marker::PhantomData};

use crate::transit::{self, type2};
use crate::utils::log2_ceil;
use zkpoly_common::{
    arith, define_usize_id,
    heap::{Heap, IdAllocator, RoHeap},
    load_dynamic::Libs,
};
use zkpoly_runtime::args::RuntimeType;

define_usize_id!(AddrId);

pub type AddrMapping = Heap<AddrId, (Addr, Size)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct IntegralSize(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmithereenSize(pub u64);

impl IntegralSize {
    pub fn double(self) -> Self {
        Self(self.0 + 1)
    }
}
impl From<IntegralSize> for SmithereenSize {
    fn from(size: IntegralSize) -> Self {
        Self(2u64.pow(size.0))
    }
}

impl TryFrom<SmithereenSize> for IntegralSize {
    type Error = ();
    fn try_from(value: SmithereenSize) -> Result<Self, Self::Error> {
        if let Some(l) = value.0.checked_ilog2() {
            if 2u64.pow(l) == value.0 {
                Ok(IntegralSize(l))
            } else {
                Err(())
            }
        } else {
            Err(())
        }
    }
}

impl TryFrom<Size> for IntegralSize {
    type Error = ();
    fn try_from(value: Size) -> Result<Self, Self::Error> {
        match value {
            Size::Integral(size) => Ok(size),
            Size::Smithereen(ss) => ss.try_into(),
        }
    }
}

impl IntegralSize {
    pub fn ceiling(size: SmithereenSize) -> Self {
        Self(log2_ceil(size.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    Integral(IntegralSize),
    Smithereen(SmithereenSize),
}

impl std::fmt::Display for Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Size::Integral(size) => write!(f, "2^{}", size.0),
            Size::Smithereen(size) => write!(f, "{}", size.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Addr(pub(crate) u64);

impl Addr {
    pub fn offset(self, x: u64) -> Addr {
        Addr(self.0 + x)
    }

    pub fn unoffset(self, x: u64) -> Addr {
        Addr(self.0 - x)
    }
}

impl Size {
    pub fn new(s: u64) -> Self {
        let ss = SmithereenSize(s);
        if let Ok(is) = IntegralSize::try_from(ss) {
            Self::Integral(is)
        } else {
            Self::Smithereen(ss)
        }
    }
}

impl From<u64> for Size {
    fn from(size: u64) -> Self {
        Self::new(size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
    Stack,
}

impl Device {
    pub fn iter() -> impl Iterator<Item = Device> {
        [Device::Gpu, Device::Cpu, Device::Stack].into_iter()
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeviceSpecific<T> {
    pub gpu: T,
    pub cpu: T,
    pub stack: T,
}

impl<T> DeviceSpecific<T> {
    pub fn get_device(&self, device: Device) -> &T {
        match device {
            Device::Gpu => &self.gpu,
            Device::Cpu => &self.cpu,
            Device::Stack => &self.stack,
        }
    }

    pub fn get_device_mut(&mut self, device: Device) -> &mut T {
        match device {
            Device::Gpu => &mut self.gpu,
            Device::Cpu => &mut self.cpu,
            Device::Stack => &mut self.stack,
        }
    }
}

define_usize_id!(RegisterId);

pub mod template {
    use zkpoly_common::typ::PolyMeta;

    use crate::transit::type2;

    use super::{typ, Size};

    #[derive(Debug, Clone)]
    pub enum InstructionNode<I, A, V> {
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
            addr: A,
        },
        GpuFree {
            id: I,
        },
        CpuMalloc {
            id: I,
            size: Size,
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
        SetPolyMeta {
            id: I,
            from: I,
            offset: u64,
            len: u64,
        },
    }

    impl<I, A, V> InstructionNode<I, A, V>
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
                Tuple { id, oprands, .. } => {
                    Box::new(std::iter::once(*id).chain(oprands.iter().copied()))
                }
                Transfer { id, .. } => Box::new(std::iter::once(*id)),
                SetPolyMeta { id, .. } => Box::new(std::iter::once(*id)),
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
    }
}

pub type VertexNode = type2::template::VertexNode<
    RegisterId,
    arith::ArithGraph<RegisterId, arith::ExprId>,
    type2::ConstantId,
    type2::user_function::Id,
>;

pub type InstructionNode = template::InstructionNode<RegisterId, AddrId, VertexNode>;

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
    Gpu,
    Cpu,
    ToGpu,
    FromGpu,
    GpuMemory,
}

#[derive(Debug, Clone, Default)]
pub struct TrackSpecific<T> {
    pub(crate) memory_management: T,
    pub(crate) co_process: T,
    pub(crate) gpu: T,
    pub(crate) cpu: T,
    pub(crate) to_gpu: T,
    pub(crate) from_gpu: T,
    pub(crate) gpu_memory: T,
}

impl<T> TrackSpecific<T> {
    pub fn get_track(&self, track: Track) -> &T {
        use Track::*;
        match track {
            MemoryManagement => &self.memory_management,
            CoProcess => &self.co_process,
            Gpu => &self.gpu,
            Cpu => &self.cpu,
            ToGpu => &self.to_gpu,
            FromGpu => &self.from_gpu,
            GpuMemory => &self.gpu_memory,
        }
    }

    pub fn get_track_mut(&mut self, track: Track) -> &mut T {
        use Track::*;
        match track {
            MemoryManagement => &mut self.memory_management,
            CoProcess => &mut self.co_process,
            Gpu => &mut self.gpu,
            Cpu => &mut self.cpu,
            ToGpu => &mut self.to_gpu,
            FromGpu => &mut self.from_gpu,
            GpuMemory => &mut self.gpu_memory,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Track, &T)> {
        use Track::*;
        vec![
            (MemoryManagement, &self.memory_management),
            (CoProcess, &self.co_process),
            (Gpu, &self.gpu),
            (Cpu, &self.cpu),
            (ToGpu, &self.to_gpu),
            (FromGpu, &self.from_gpu),
            (GpuMemory, &self.gpu_memory),
        ]
        .into_iter()
    }

    pub fn new(t: T) -> Self
    where
        T: Clone,
    {
        Self {
            memory_management: t.clone(),
            co_process: t.clone(),
            gpu: t.clone(),
            cpu: t.clone(),
            to_gpu: t.clone(),
            from_gpu: t.clone(),
            gpu_memory: t,
        }
    }
}

impl Track {
    pub fn on_device(device: type2::Device) -> Track {
        use type2::Device::*;
        match device {
            Gpu => Track::Gpu,
            Cpu => Track::Cpu,
            PreferGpu => panic!("PreferGpu should have been resolved"),
        }
    }
}

fn determine_transfer_track(from: Device, to: Device) -> Track {
    use Device::*;
    match (from, to) {
        (Gpu, Cpu) => Track::FromGpu,
        (Gpu, Stack) => Track::FromGpu,
        (Gpu, Gpu) => Track::GpuMemory,
        (Cpu, Gpu) => Track::ToGpu,
        (Cpu, Stack) => panic!("Cpu cannot transfer to Stack"),
        (Cpu, Cpu) => Track::Cpu,
        (Stack, Gpu) => Track::ToGpu,
        (Stack, Cpu) => panic!("Stack cannot transfer to Cpu"),
        (Stack, Stack) => Track::Cpu,
    }
}

impl<'s> Instruction<'s> {
    pub fn track(&self, devices: impl Fn(RegisterId) -> Device) -> Track {
        use template::InstructionNode::*;
        use Track::*;

        match &self.node {
            Type2 { vertex, ids, .. } => vertex.track(match devices(ids[0].0) {
                Device::Cpu | Device::Stack => type2::Device::Cpu,
                Device::Gpu => type2::Device::Gpu,
            }),
            GpuMalloc { .. } => MemoryManagement,
            GpuFree { .. } => MemoryManagement,
            CpuMalloc { .. } => MemoryManagement,
            CpuFree { .. } => MemoryManagement,
            StackFree { .. } => MemoryManagement,
            Tuple { .. } => Cpu,
            Transfer { from, id, .. } => determine_transfer_track(devices(*from), devices(*id)),
            SetPolyMeta { .. } => Cpu,
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
            Transfer { id, .. } => Box::new(std::iter::once(*id)),
            _ => Box::new(std::iter::empty()),
        }
    }
}
define_usize_id!(InstructionIndex);

#[derive(Debug)]
pub struct Chunk<'s, Rt: RuntimeType> {
    pub(crate) instructions: Vec<Instruction<'s>>,
    pub(crate) register_types: RoHeap<RegisterId, typ::Typ>,
    pub(crate) register_devices: BTreeMap<RegisterId, Device>,
    pub(crate) gpu_addr_mapping: AddrMapping,
    pub(crate) reg_id_allocator: IdAllocator<RegisterId>,
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

    pub fn take_reg_id_allocator(&mut self) -> IdAllocator<RegisterId> {
        let mut x = IdAllocator::new();
        std::mem::swap(&mut self.reg_id_allocator, &mut x);
        x
    }

    pub fn take_libs(&mut self) -> Libs {
        let mut x = Libs::new();
        std::mem::swap(&mut self.libs, &mut x);
        x
    }
}

pub mod lowering;
pub mod pretty_print;
pub mod track_splitting;
pub mod typ;
