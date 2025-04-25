use std::{collections::BTreeMap, marker::PhantomData};

use crate::transit::{self, type2};
use crate::utils::{log2, log2_ceil};
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

    pub fn unwrap_integral(self) -> IntegralSize {
        match self {
            Size::Integral(size) => size,
            Size::Smithereen(..) => panic!("unwrap_integral on Size::Smithereen")
        }
    }
}

impl From<u64> for Size {
    fn from(size: u64) -> Self {
        Self::new(size)
    }
}

impl std::ops::Div<u64> for Size {
    type Output = Size;
    fn div(self, rhs: u64) -> Self::Output {
        match self {
            Size::Integral(IntegralSize(is)) => {
                if let Some(log) = log2(rhs) {
                    Self::Integral(IntegralSize(is - log))
                } else {
                    panic!("can only divide by power of 2")
                }
            }
            Size::Smithereen(SmithereenSize(ss)) => Self::Smithereen(SmithereenSize(ss / rhs)),
        }
    }
}

impl std::ops::Mul<u64> for Size {
    type Output = Size;
    fn mul(self, rhs: u64) -> Self::Output {
        match self {
            Size::Integral(IntegralSize(is)) => {
                if let Some(log) = log2(rhs) {
                    Self::Integral(IntegralSize(is + log))
                } else {
                    panic!("can only multiply by power of 2")
                }
            }
            Size::Smithereen(SmithereenSize(ss)) => Self::Smithereen(SmithereenSize(ss * rhs)),
        }
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
    use zkpoly_common::typ::PolyType;

    use crate::{ast::PolyInit, transit::type2};

    use super::Size;

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

impl Track {
    pub fn is_gpu(&self) -> bool {
        lowering::Stream::of_track(*self).is_some()
    }

    pub fn is_cpu(&self) -> bool {
        lowering::Stream::of_track(*self).is_none()
    }
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

        let executor_of = |md| match md {
            Device::Cpu | Device::Stack => type2::Device::Cpu,
            Device::Gpu => type2::Device::Gpu,
        };

        match &self.node {
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
}
define_usize_id!(InstructionIndex);

pub struct RegisterAllocator {
    register_types: Heap<RegisterId, typ::Typ>,
    register_devices: BTreeMap<RegisterId, Device>,
}

impl RegisterAllocator {
    pub fn alloc(&mut self, typ: typ::Typ, device: Device) -> RegisterId {
        let id = self.register_types.push(typ);
        self.register_devices.insert(id, device);
        id
    }

    pub fn device_of(&self, id: RegisterId) -> Device {
        self.register_devices[&id]
    }

    pub fn typ_of(&self, id: RegisterId) -> &typ::Typ {
        &self.register_types[id]
    }

    pub fn inherit(&mut self, r: RegisterId) -> RegisterId {
        let device = self.device_of(r);
        let typ = self.typ_of(r);
        self.alloc(typ.clone(), device)
    }
}

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
            register_types: regster_types.to_mutable(reg_id_allocator),
        };

        let (mut self2, ra) = f(self, ra);
        let (register_types, reg_id_allocator) = ra.register_types.freeze();

        self2.reg_id_allocator = reg_id_allocator;
        self2.register_types = register_types;
        self2.register_devices = ra.register_devices;
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
