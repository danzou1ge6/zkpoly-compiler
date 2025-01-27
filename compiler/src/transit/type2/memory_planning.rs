use std::collections::{BTreeMap, BTreeSet};

use zkpoly_common::{
    bijection::Bijection,
    define_usize_id,
    digraph::internal::Predecessors,
    heap::{Heap, IdAllocator, UsizeId},
};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::type2::Device;

use super::{Cg, VertexId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Addr(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct IntegralSize(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmithereenSize(u64);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Instant(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    Integral(IntegralSize),
    Smithereen(SmithereenSize),
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

define_usize_id!(RuntimeAddrId);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalAddr {
    Cpu(RuntimeAddrId, Size),
    Gpu(Addr, Size),
}

impl GlobalAddr {
    pub fn size(&self) -> Size {
        match self {
            GlobalAddr::Cpu(_, size) => *size,
            GlobalAddr::Gpu(_, size) => *size,
        }
    }

    pub fn is_on_cpu(&self) -> bool {
        match self {
            GlobalAddr::Cpu(_, _) => true,
            GlobalAddr::Gpu(_, _) => false,
        }
    }

    pub fn is_on_gpu(&self) -> bool {
        match self {
            GlobalAddr::Cpu(_, _) => false,
            GlobalAddr::Gpu(_, _) => true,
        }
    }
}

define_usize_id!(AddrId);

pub type AddrMapping = Heap<AddrId, GlobalAddr>;

trait AddrMappingHandler {
    fn add(&mut self, addr: Addr, size: Size) -> AddrId;
    fn update(&mut self, id: AddrId, addr: Addr, size: Size);
    fn get(&self, id: AddrId) -> (Addr, Size);
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

struct GpuAddrMappingHandler<'m>(&'m mut AddrMapping);

impl<'m> AddrMappingHandler for GpuAddrMappingHandler<'m> {
    fn add(&mut self, addr: Addr, size: Size) -> AddrId {
        self.0.push(GlobalAddr::Gpu(addr, size))
    }

    fn update(&mut self, id: AddrId, addr: Addr, size: Size) {
        self.0[id] = GlobalAddr::Gpu(addr, size);
    }

    fn get(&self, id: AddrId) -> (Addr, Size) {
        match self.0[id] {
            GlobalAddr::Gpu(addr, size) => (addr, size),
            GlobalAddr::Cpu(_, _) => panic!("invalid global addr"),
        }
    }
}

impl IntegralSize {
    pub fn double(self) -> Self {
        Self(self.0 + 1)
    }
}

mod integral_allocator;
mod smithereens_allocator;

#[derive(Debug, Clone)]
pub enum Work<I: UsizeId, A> {
    Vertex {
        id: I,
        args: Vec<A>,
        outputs: Vec<A>,
        temp: Option<A>,
    },
    CpuMalloc {
        id: I,
        addr: A,
    },
    CpuFree {
        id: I,
        addr: A,
    },
    Transfer {
        id: I,
        src: A,
        dst: A,
    },
}

impl<I: UsizeId> Work<I, AddrId> {
    pub fn to_global_addr(self, mapping: &AddrMapping) -> Work<I, GlobalAddr> {
        match self {
            Work::Vertex {
                id,
                args,
                outputs,
                temp,
            } => Work::Vertex {
                id,
                args: args.into_iter().map(|id| mapping[id]).collect(),
                outputs: outputs.into_iter().map(|id| mapping[id]).collect(),
                temp: temp.map(|id| mapping[id]),
            },
            Work::CpuMalloc { id, addr } => Work::CpuMalloc {
                id,
                addr: mapping[addr],
            },
            Work::CpuFree { id, addr } => Work::CpuFree {
                id,
                addr: mapping[addr],
            },
            Work::Transfer { id, src, dst } => Work::Transfer {
                id,
                src: mapping[src],
                dst: mapping[dst],
            },
        }
    }
}

fn collect_integral_sizes<'s, Rt: RuntimeType>(cg: &Cg<'s, Rt>) -> Vec<IntegralSize> {
    let mut integral_sizes = BTreeSet::<IntegralSize>::new();
    for vid in cg.g.vertices() {
        let v = cg.g.vertex(vid);
        let temp_size = SmithereenSize(v.temporary_space_needed());
        for &size in v.typ().size().iter() {
            if let Ok(is) = IntegralSize::try_from(SmithereenSize(size)) {
                integral_sizes.insert(is);
            }
        }
        if let Ok(is) = IntegralSize::try_from(temp_size) {
            integral_sizes.insert(is);
        }
    }

    integral_sizes.into_iter().collect()
}

const MIN_SMITHEREEN_SPACE: u64 = 2u64.pow(26);
const SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD: u64 = 2u64.pow(12);

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

struct CpuAllocator {
    id_alloc: IdAllocator<RuntimeAddrId>,
}

impl CpuAllocator {
    pub fn new() -> Self {
        Self {
            id_alloc: IdAllocator::new(),
        }
    }

    pub fn allocate(
        &mut self,
        size: Size,
        vid: VertexId,
        mapping: &mut AddrMapping,
        outputs: &mut Vec<Work<VertexId, AddrId>>,
    ) -> AddrId {
        let rt_addr_id = self.id_alloc.alloc();
        let addr = mapping.push(GlobalAddr::Cpu(rt_addr_id, size));
        outputs.push(Work::CpuMalloc { id: vid, addr });
        addr
    }

    pub fn dealloc(
        &mut self,
        addr: AddrId,
        vid: VertexId,
        outputs: &mut Vec<Work<VertexId, AddrId>>,
    ) {
        outputs.push(Work::CpuFree { id: vid, addr });
    }
}

fn move_victims(
    victims: &[AddrId],
    cpu_allocator: &mut CpuAllocator,
    mapping: &mut AddrMapping,
    outputs: &mut Vec<Work<VertexId, AddrId>>,
    av: &mut Bijection<VertexId, AddrId>,
) {
    victims.iter().for_each(|&victim| {
        let vid = av.get_backward(&victim).cloned().unwrap();
        let move_to = cpu_allocator.allocate(mapping[victim].size(), vid, mapping, outputs);
        outputs.push(Work::Transfer {
            id: vid,
            src: victim,
            dst: move_to,
        });
    });
}

fn allocate_gpu_integral(
    size: IntegralSize,
    vid: VertexId,
    next_use: Instant,
    gpu_ialloc: &mut integral_allocator::regretting::Allocator,
    cpu_allocator: &mut CpuAllocator,
    mapping: &mut AddrMapping,
    outputs: &mut Vec<Work<VertexId, AddrId>>,
    av: &mut Bijection<VertexId, AddrId>,
) -> AddrId {
    let addr = gpu_ialloc.allocate(size, next_use, &mut GpuAddrMappingHandler(mapping));

    if let Some((transfers, addr)) = addr {
        for t in transfers {
            outputs.push(Work::Transfer {
                id: vid,
                src: t.from,
                dst: t.to,
            });
        }
        return addr;
    }

    let (addr, victims) =
        gpu_ialloc.decide_and_realloc_victim(size, next_use, &mut GpuAddrMappingHandler(mapping));
    move_victims(&victims, cpu_allocator, mapping, outputs, av);

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

fn allocate_gpu(
    size: Size,
    vid: VertexId,
    next_use: Instant,
    gpu_ialloc: &mut integral_allocator::regretting::Allocator,
    gpu_salloc: &mut smithereens_allocator::Allocator,
    cpu_allocator: &mut CpuAllocator,
    mapping: &mut AddrMapping,
    outputs: &mut Vec<Work<VertexId, AddrId>>,
    av: &mut Bijection<VertexId, AddrId>,
) -> Result<AddrId, InsufficientSmithereenSpace> {
    match size {
        Size::Integral(size) => Ok(allocate_gpu_integral(
            size,
            vid,
            next_use,
            gpu_ialloc,
            cpu_allocator,
            mapping,
            outputs,
            av,
        )),
        Size::Smithereen(size) => {
            if size.0 >= SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD
                || IntegralSize::try_from(size).is_ok()
            {
                let size = IntegralSize::ceiling(size);
                Ok(allocate_gpu_integral(
                    size,
                    vid,
                    next_use,
                    gpu_ialloc,
                    cpu_allocator,
                    mapping,
                    outputs,
                    av,
                ))
            } else {
                allocate_gpu_smithereen(size, gpu_salloc, mapping)
            }
        }
    }
}

fn deallocate(
    addr_id: AddrId,
    vid: VertexId,
    gpu_ialloc: &mut integral_allocator::regretting::Allocator,
    gpu_salloc: &mut smithereens_allocator::Allocator,
    cpu_allocator: &mut CpuAllocator,
    mapping: &mut AddrMapping,
    outputs: &mut Vec<Work<VertexId, AddrId>>,
) {
    match mapping[addr_id] {
        GlobalAddr::Gpu(_, size) => match size {
            Size::Integral(..) => {
                gpu_ialloc.deallocate(addr_id, &mut GpuAddrMappingHandler(mapping));
            }
            Size::Smithereen(..) => {
                gpu_salloc.deallocate(addr_id, &mut GpuAddrMappingHandler(mapping));
            }
        },
        GlobalAddr::Cpu(..) => {
            cpu_allocator.dealloc(addr_id, vid, outputs);
        }
    }
}

pub fn plan<'s, Rt: RuntimeType>(
    capacity: u64,
    cg: &Cg<'s, Rt>,
    seq: &[VertexId],
    die_at: &BTreeMap<VertexId, usize>,
) -> Result<
    (Vec<Work<VertexId, GlobalAddr>>, BTreeMap<VertexId, Device>),
    InsufficientSmithereenSpace,
> {
    let integral_sizes = collect_integral_sizes(cg);
    let (ispace, sspace) =
        divide_integral_smithereens(capacity, integral_sizes.last().unwrap().clone());

    let successors = cg.g.successors();
    let next_ueses = collect_next_uses(&successors, seq);

    let mut mapping = AddrMapping::new();
    let mut cpu_alloc = CpuAllocator::new();
    let mut gpu_ialloc = integral_allocator::regretting::Allocator::new(ispace, integral_sizes);
    let mut gpu_salloc = smithereens_allocator::Allocator::new(sspace);
    let mut av: Bijection<VertexId, AddrId> = Bijection::new();

    let mut outputs = vec![];
    let mut devices = BTreeMap::new();

    for ((i, &vid), updated_next_uses) in seq.iter().enumerate().zip(next_ueses.into_iter()) {
        let now = Instant(i);

        let device = match cg.g.vertex(vid).device() {
            Device::PreferGpu => {
                if cg
                    .g
                    .vertex(vid)
                    .predecessors()
                    .map(|p| av.get_forward(&p).unwrap())
                    .any(|&addr| mapping[addr].is_on_gpu())
                    || successors[vid]
                        .iter()
                        .any(|succ| cg.g.vertex(*succ).device() == Device::Gpu)
                {
                    Device::Gpu
                } else {
                    Device::Cpu
                }
            }
            x => x,
        };
        devices.insert(vid, device);

        let (output_addrs, temp_addr) = match device {
            Device::Gpu => {
                let output_addrs =
                    cg.g.vertex(vid)
                        .typ()
                        .size()
                        .iter()
                        .zip(cg.g.vertex(vid).predecessors())
                        .map(|(size, arg)| {
                            allocate_gpu(
                                Size::Smithereen(SmithereenSize(*size)),
                                vid,
                                updated_next_uses[&arg],
                                &mut gpu_ialloc,
                                &mut gpu_salloc,
                                &mut cpu_alloc,
                                &mut mapping,
                                &mut outputs,
                                &mut av,
                            )
                        })
                        .collect::<Result<_, _>>()?;
                let temp_space = cg.g.vertex(vid).temporary_space_needed();
                let temp_addr = if temp_space != 0 {
                    Some(allocate_gpu(
                        Size::Smithereen(SmithereenSize(temp_space)),
                        vid, // using vid here is ok, as temproary space will never be transfered
                        now,
                        &mut gpu_ialloc,
                        &mut gpu_salloc,
                        &mut cpu_alloc,
                        &mut mapping,
                        &mut outputs,
                        &mut av,
                    )?)
                } else {
                    None
                };
                (output_addrs, temp_addr)
            }
            Device::Cpu => {
                let output_addrs =
                    cg.g.vertex(vid)
                        .typ()
                        .size()
                        .iter()
                        .zip(cg.g.vertex(vid).predecessors())
                        .map(|(size, arg)| {
                            cpu_alloc.allocate(
                                Size::Smithereen(SmithereenSize(*size)),
                                arg,
                                &mut mapping,
                                &mut outputs,
                            )
                        })
                        .collect();
                let temp_space = cg.g.vertex(vid).temporary_space_needed();
                let temp_addr = if temp_space != 0 {
                    Some(cpu_alloc.allocate(
                        Size::Smithereen(SmithereenSize(temp_space)),
                        vid,
                        &mut mapping,
                        &mut outputs,
                    ))
                } else {
                    None
                };
                (output_addrs, temp_addr)
            }
            _ => unreachable!(),
        };

        outputs.push(Work::Vertex {
            id: vid,
            args: cg
                .g
                .vertex(vid)
                .predecessors()
                .map(|vid| av.get_forward(&vid).cloned().unwrap())
                .collect(),
            outputs: output_addrs,
            temp: temp_addr,
        });

        cg.g.vertex(vid).predecessors().for_each(|pred| {
            if die_at[&pred] == i {
                deallocate(
                    *av.get_forward(&pred).unwrap(),
                    pred,
                    &mut gpu_ialloc,
                    &mut gpu_salloc,
                    &mut cpu_alloc,
                    &mut mapping,
                    &mut outputs,
                );
            }
        });
    }

    let outputs = outputs
        .into_iter()
        .map(|work| work.to_global_addr(&mapping))
        .collect();

    Ok((outputs, devices))
}
