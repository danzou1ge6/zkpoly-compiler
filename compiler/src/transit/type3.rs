use std::collections::BTreeMap;

use crate::transit::{self, type2};
use zkpoly_common::{define_usize_id, heap::Heap, arith};
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

#[derive(Debug, Clone)]
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
            id: I
        },
        CpuMalloc {
            id: I,
            size: Size,
        },
        CpuFree {
            id: I,
        },
        StackAlloc {
            id: I
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
            rot: R,
        },
        Move {
            id: I,
            from: I,
        },
    }
}

pub type VertexNode = 
    type2::template::VertexNode<
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

#[derive(Debug, Clone)]
pub struct Chunk<'s, Rt: RuntimeType> {
    pub(crate) instructions: Vec<Instruction<'s>>,
    pub(crate) register_types: BTreeMap<RegisterId, type2::Typ<Rt>>,
    pub(crate) register_devices: BTreeMap<RegisterId, Device>,
    pub(crate) gpu_addr_mapping: AddrMapping,
}
