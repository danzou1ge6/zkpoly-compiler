use std::collections::BTreeMap;

use crate::transit::{self, type2};
use zkpoly_common::{arith, define_usize_id, heap::Heap};
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

fn log2_ceil(x: u64) -> u32 {
    if x == 0 {
        panic!("log2(0) is undefined");
    }
    64 - x.leading_zeros()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Addr(pub(crate) u64);

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
    use super::Size;

    #[derive(Debug, Clone)]
    pub enum InstructionNode<I, A, R, V> {
        Type2 {
            ids: Vec<I>,
            temp: Option<I>,
            vertex: V,
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
        StackAlloc {
            id: I,
        },
        StackFree {
            id: I,
        },
        Tuple {
            id: I,
            oprands: Vec<I>,
        },
        RotateAndSlice {
            id: I,
            operand: I,
            rot: R,
            slice: (u64, u64),
        },
        Transfer {
            id: I,
            from: I,
            rot: R,
        },
        Move {
            id: I,
            from: I,
        },
    }

    impl<I, A, R, V> InstructionNode<I, A, R, V>
    where
        I: Copy,
    {
        pub fn ids<'s>(&'s self) -> Box<dyn Iterator<Item = I> + 's> {
            use InstructionNode::*;
            match self {
                Type2 { ids, .. } => Box::new(ids.iter().copied()),
                GpuMalloc { id, .. } => Box::new(std::iter::once(*id)),
                GpuFree { id, .. } => Box::new(std::iter::once(*id)),
                CpuMalloc { id, .. } => Box::new(std::iter::once(*id)),
                CpuFree { id, .. } => Box::new(std::iter::once(*id)),
                StackAlloc { id, .. } => Box::new(std::iter::once(*id)),
                StackFree { id, .. } => Box::new(std::iter::once(*id)),
                Tuple { id, oprands, .. } => {
                    Box::new(std::iter::once(*id).chain(oprands.iter().copied()))
                }
                RotateAndSlice {
                    id,
                    ..
                } => Box::new(std::iter::once(*id)),
                Transfer { id, .. } => Box::new(std::iter::once(*id)),
                Move { id, .. } => Box::new(std::iter::once(*id)),
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

pub type InstructionNode = template::InstructionNode<RegisterId, AddrId, i32, VertexNode>;

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
    CoProcess,
    Gpu,
    Cpu,
    Pcie,
}

#[derive(Debug, Clone, Default)]
pub struct TrackSpecific<T> {
    pub(crate) co_process: T,
    pub(crate) gpu: T,
    pub(crate) cpu: T,
    pub(crate) pcie: T,
}

impl<T> TrackSpecific<T> {
    pub fn get_track(&self, track: Track) -> &T {
        use Track::*;
        match track {
            CoProcess => &self.co_process,
            Gpu => &self.gpu,
            Cpu => &self.cpu,
            Pcie => &self.pcie,
        }
    }

    pub fn get_track_mut(&mut self, track: Track) -> &mut T {
        use Track::*;
        match track {
            CoProcess => &mut self.co_process,
            Gpu => &mut self.gpu,
            Cpu => &mut self.cpu,
            Pcie => &mut self.pcie,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Track, &T)> {
        use Track::*;
        vec![
            (CoProcess, &self.co_process),
            (Gpu, &self.gpu),
            (Cpu, &self.cpu),
            (Pcie, &self.pcie),
        ]
        .into_iter()
    }
}

impl Track {
    pub fn on_device(device: Device) -> Track {
        use Device::*;
        match device {
            Gpu => Track::Gpu,
            Cpu => Track::Cpu,
            Stack => Track::Cpu,
        }
    }
}

fn determine_transfer_track(from: Device, to: Device) -> Track {
    use Device::*;
    match (from, to) {
        (Gpu, Cpu) => Track::Pcie,
        (Gpu, Stack) => Track::Pcie,
        (Gpu, Gpu) => Track::Gpu,
        (Cpu, Gpu) => Track::Pcie,
        (Cpu, Stack) => panic!("Cpu cannot transfer to Stack"),
        (Cpu, Cpu) => Track::Cpu,
        (Stack, Gpu) => Track::Pcie,
        (Stack, Cpu) => panic!("Stack cannot transfer to Cpu"),
        (Stack, Stack) => Track::Cpu,
    }
}

impl<'s> Instruction<'s> {
    pub fn track(&self, devices: impl Fn(RegisterId) -> Device) -> Track {
        use template::InstructionNode::*;
        use Track::*;

        match &self.node {
            Type2 { vertex, ids, .. } => vertex.track(devices(ids[0])),
            GpuMalloc { .. } => Cpu,
            GpuFree { .. } => Cpu,
            CpuMalloc { .. } => Cpu,
            CpuFree { .. } => Cpu,
            StackAlloc { .. } => Cpu,
            StackFree { .. } => Cpu,
            Tuple { .. } => Cpu,
            RotateAndSlice { .. } => Cpu,
            Transfer { from, id, .. } => determine_transfer_track(devices(*from), devices(*id)),
            Move { .. } => Cpu,
        }
    }

    pub fn defs<'a>(&'a self) -> Box<dyn Iterator<Item = RegisterId> + 'a> {
        self.node.ids()
    }
}
define_usize_id!(InstructionIndex);

#[derive(Debug, Clone)]
pub struct Chunk<'s, Rt: RuntimeType> {
    pub(crate) instructions: Vec<Instruction<'s>>,
    pub(crate) register_types: BTreeMap<RegisterId, type2::Typ<Rt>>,
    pub(crate) register_devices: BTreeMap<RegisterId, Device>,
    pub(crate) gpu_addr_mapping: AddrMapping,
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
        self.instructions
            .iter()
            .enumerate()
            .fold(BTreeMap::new(), |mut acc, (i, instr)| {
                for id in instr.node.ids() {
                    acc.insert(id, InstructionIndex(i));
                }
                acc
            })
    }
}

pub mod track_splitting;
