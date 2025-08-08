mod prelude {
    pub(super) use std::{
        collections::{BTreeMap, BTreeSet},
        marker::PhantomData,
    };

    pub(super) use zkpoly_common::{
        arith::Mutability,
        bijection::Bijection,
        define_usize_id,
        digraph::internal::SubDigraph,
        heap::{Heap, IdAllocator, RoHeap, UsizeId},
        load_dynamic::Libs,
        mm_heap::MmHeap,
        typ::Slice,
    };

    pub(super) use zkpoly_runtime::{
        args::RuntimeType,
        instructions::{AllocMethod, AllocVariant},
    };

    pub(super) use crate::driver::HardwareInfo;
    pub(super) use crate::transit::{
        type2::{
            self,
            object_analysis::{
                self, liveness, object_info,
                size::{IntegralSize, LogBlockSizes, Size, SmithereenSize},
                template::{Operation, OperationSeq, ResidentalValue},
                Index, ObjectId, Value, ValueNode, VertexInput,
            },
            VertexId,
        },
        type3::{self, Device, DeviceSpecific, RegisterId},
    };

    pub(super) use super::{
        address::Addr,
        allocator::{
            self, Allocator, AllocatorCollection, AllocatorHandle, AllocatorRealizer, Completeness,
            Cpu, DeviceMarker, Disk, Gpu,
        },
        allocators,
        auxiliary::AuxiliaryInfo,
        continuations::{self, Continuation, Response},
        planning,
        realization::{self, RealizationResponse},
        Error,
    };
}

use prelude::*;

use crate::driver;

pub mod address;
mod allocator;
mod continuations;

#[derive(Debug, Clone)]
pub enum Error<'s> {
    VertexInputsOutputsNotAccommodated(Option<(VertexId, super::SourceInfo<'s>)>),
    InsufficientSmithereenSpace,
}

impl<'s> Error<'s> {
    pub fn with_vid_src(self, vid: VertexId, src: super::SourceInfo<'s>) -> Self {
        match self {
            Self::VertexInputsOutputsNotAccommodated(_) => {
                Self::VertexInputsOutputsNotAccommodated(Some((vid, src)))
            }
            _ => self,
        }
    }
}

define_usize_id!(Pointer);

pub type MemoryBlock = realization::MemoryBlock<Pointer>;

pub mod allocators;
pub mod auxiliary;
pub mod planning;
pub mod realization;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Statistics {
    pub realization_statistics: realization::machine::Statistics,
    pub disk_peak_usage: u64,
    pub cpu_peak_usage: u64,
    pub disk_constants_size: u64,
    pub cpu_constants_size: u64,
}

impl std::fmt::Display for Statistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::utils::human_readable_size;

        writeln!(f, "Realization statistics:")?;
        self.realization_statistics.pretty_print(f, 2)?;
        writeln!(
            f,
            "Disk peak usage           : {}",
            human_readable_size(self.disk_peak_usage)
        )?;
        writeln!(
            f,
            "CPU peak usage            : {}",
            human_readable_size(self.cpu_peak_usage)
        )?;
        writeln!(
            f,
            "Disk constants size       : {}",
            human_readable_size(self.disk_constants_size)
        )?;
        writeln!(
            f,
            "CPU constants size        : {}",
            human_readable_size(self.cpu_constants_size)
        )
    }
}

pub fn plan<'s, Rt: RuntimeType>(
    cg: &type2::Cg<'s, Rt>,
    g: &SubDigraph<type2::VertexId, type2::Vertex<'s, Rt>>,
    seq: &[VertexId],
    def_use: &object_analysis::cg_def_use::DefUse,
    mut obj_id_allocator: IdAllocator<ObjectId>,
    execution_device: impl Fn(VertexId) -> type2::Device,
    hd_info: &driver::HardwareInfo,
    libs: &mut Libs,
) -> Result<(type3::Chunk<'s, Rt>, Statistics), Error<'s>> {
    let ops: OperationSeq<'_, ObjectId, Pointer> = OperationSeq::construct(
        cg,
        g,
        seq,
        &execution_device,
        &def_use,
        &mut obj_id_allocator,
        libs,
    );

    let obj_info = object_info::Info::<Rt>::collect(&ops);

    use allocators::{
        ConstantWrapper, PageAllocator, SlabAllocator, SmithereenWrapper, SuperAllocator,
    };

    let lbss = allocators::collect_integral_sizes(obj_info.sizes());
    let mut gpu_allocators: Vec<
        SmithereenWrapper<SlabAllocator<Pointer, Rt, Gpu>, Pointer, Rt, Gpu>,
    > = hd_info
        .gpus()
        .map(|gpu| {
            SmithereenWrapper::new(
                SlabAllocator::new(gpu.integral_space(), gpu.smithereen_space(), lbss.clone()),
                gpu.smithereen_space(),
                0,
            )
        })
        .collect();

    let mut cpu_allocator: SmithereenWrapper<
        '_,
        SlabAllocator<'_, Pointer, Rt, Cpu>,
        Pointer,
        Rt,
        Cpu,
    > = SmithereenWrapper::new(
        SlabAllocator::new(
            hd_info.cpu().integral_space(),
            hd_info.cpu().smithereen_space(),
            lbss.clone(),
        ),
        hd_info.cpu().smithereen_space(),
        0,
    );
    let mut disk_allocator = SuperAllocator::<Pointer, Rt, Disk>::new();

    let cpu_constant_objects = def_use
        .immortal_objects()
        .iter()
        .copied()
        .filter(|obj| def_use.def_on(*obj) == Device::Cpu)
        .map(|obj| (obj, cpu_allocator.allcate_pointer()))
        .collect::<Vec<_>>();
    let disk_constant_objects = def_use
        .immortal_objects()
        .iter()
        .copied()
        .filter(|obj| def_use.def_on(*obj) == Device::Disk)
        .map(|obj| (obj, disk_allocator.allcate_pointer()))
        .collect::<Vec<_>>();

    let cpu_constants_size = cpu_constant_objects
        .iter()
        .map(|(obj, _)| u64::from(obj_info.size(*obj)))
        .sum();
    let disk_constants_size = disk_constant_objects
        .iter()
        .map(|(obj, _)| u64::from(obj_info.size(*obj)))
        .sum();

    let mut cpu_allocator =
        ConstantWrapper::<_, _, _, Rt, Cpu>::new(cpu_allocator, cpu_constant_objects.into_iter());
    let mut disk_allocator = ConstantWrapper::<_, _, _, Rt, Disk>::new(
        disk_allocator,
        disk_constant_objects.into_iter(),
    );

    let ops = planning::transform_ops(
        ops,
        &mut gpu_allocators,
        &mut cpu_allocator,
        &mut disk_allocator,
        &obj_info,
        hd_info,
    )?;

    let allocators: AllocatorCollection<ObjectId, Pointer, Rt> = gpu_allocators
        .iter_mut()
        .enumerate()
        .map(|(i, alloc)| {
            (
                Device::Gpu(i),
                alloc as &mut dyn Allocator<ObjectId, Pointer, Rt>,
            )
        })
        .chain(std::iter::once((
            Device::Cpu,
            &mut cpu_allocator as &mut dyn Allocator<ObjectId, Pointer, Rt>,
        )))
        .chain(std::iter::once((
            Device::Disk,
            &mut disk_allocator as &mut dyn Allocator<ObjectId, Pointer, Rt>,
        )))
        .collect();

    let (chunk, realization_statistics) = realization::realize(
        ops,
        allocators,
        obj_id_allocator,
        &obj_info,
        hd_info,
        LogBlockSizes::new(lbss),
        &execution_device,
    )?;

    let disk_allocator = disk_allocator.unwrap();

    if hd_info.disks_available() == 0 && disk_allocator.peak_memory_usage() > 0 {
        panic!("disk is not enabled but some objects are spilled to disk; this indicates insufficient CPU space");
    }

    Ok((
        chunk,
        Statistics {
            realization_statistics,
            disk_peak_usage: disk_allocator.peak_memory_usage(),
            cpu_peak_usage: cpu_allocator.unwrap().unwrap().peak_memory_usage()
                + hd_info.cpu().smithereen_space(),
            disk_constants_size,
            cpu_constants_size,
        },
    ))
}
