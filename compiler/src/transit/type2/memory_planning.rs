static DEBUG: bool = true;

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

    pub(super) use zkpoly_runtime::args::RuntimeType;

    pub(super) use crate::transit::{
        type2::{
            self,
            object_analysis::{
                self, liveness, object_info,
                size::{IntegralSize, Size, SmithereenSize},
                template::{Operation, OperationSeq, ResidentalValue},
                Index, ObjectId, Value, ValueNode, VertexInput,
            },
            VertexId,
        },
        type3::{self, Device, DeviceSpecific, RegisterId},
    };
    pub(super) use crate::driver::HardwareInfo;

    pub(super) use super::{
        address::{Addr, AddrId, AddrMapping},
        allocator::{
            Allocator, AllocatorCollection, AllocatorHandle, AllocatorRealizer, Completeness,
        },
        allocators,
        auxiliary::AuxiliaryInfo,
        continuations::{self, Continuation, Response},
        planning::{self, machine::PlanningResponse},
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

pub fn plan<'s, Rt: RuntimeType>(
    cg: &type2::Cg<'s, Rt>,
    g: &SubDigraph<type2::VertexId, type2::Vertex<'s, Rt>>,
    seq: &[VertexId],
    execution_device: impl Fn(VertexId) -> type2::Device,
    uf_table: &type2::user_function::Table<Rt>,
    hd_info: &driver::HardwareInfo,
    mut libs: Libs,
) -> Result<type3::Chunk<'s, Rt>, Error<'s>> {
    let (def_use, mut obj_id_allocator) = object_analysis::cg_def_use::DefUse::analyze(
        g,
        uf_table,
        seq,
        cg.output,
        &execution_device,
        hd_info,
    );

    let ops: OperationSeq<'_, ObjectId, Pointer> = OperationSeq::construct(
        cg,
        g,
        seq,
        &execution_device,
        &def_use,
        &mut obj_id_allocator,
        &mut libs,
    );

    let obj_info = object_info::Info::<Rt>::collect(&ops);

    use allocators::{CpuAllocator, GpuAllocator};

    let lbss = allocators::gpu_allocator::collect_integral_sizes(obj_info.sizes());
    let mut gpu_allocators: Vec<GpuAllocator<Pointer>> = hd_info
        .gpus()
        .map(|gpu| GpuAllocator::new(gpu.gpu_memory_limit, gpu.gpu_smithereen_space, lbss.clone()))
        .collect();
    let mut cpu_allocator = CpuAllocator::<Pointer>::new();

    let ops = planning::transform_ops(
        ops,
        &mut gpu_allocators,
        &mut cpu_allocator,
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
        .collect();

    Ok(realization::realize(
        ops,
        allocators,
        libs,
        obj_id_allocator,
        &obj_info,
        hd_info
    ))
}
