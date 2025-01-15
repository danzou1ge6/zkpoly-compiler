use group::ff::Field;
use zkpoly_cuda_api::mem::CudaAllocator;
use zkpoly_memory_pool::PinnedMemoryPool;

use crate::{
    args::{RuntimeType, Variable, VariableId},
    devices::DeviceType,
    only_cpu,
    poly::Polynomial,
    run::RuntimeInfo,
    typ::Typ,
};

pub fn allocate<T: RuntimeType>(
    info: &RuntimeInfo<T>,
    device: DeviceType,
    typ: Typ,
    id: VariableId,
    offset: Option<usize>,
    mem_allocator: &Option<PinnedMemoryPool>,
    gpu_allocator: &Option<Vec<CudaAllocator>>,
) {
    let mut target = info.variable[id].write().unwrap();
    match typ {
        Typ::Poly {
            typ: ref poly_typ,
            log_n,
        } => {
            let poly = match device {
                DeviceType::CPU => Polynomial::<T::Field>::new(
                    poly_typ.clone(),
                    log_n,
                    mem_allocator.as_ref().unwrap().allocate(log_n.clone()),
                    device.clone(),
                ),
                DeviceType::GPU { device_id } => Polynomial::<T::Field>::new(
                    poly_typ.clone(),
                    log_n,
                    gpu_allocator.as_ref().unwrap()[device_id as usize].allocate(offset.unwrap()),
                    device.clone(),
                ),
                DeviceType::Disk => todo!(),
            };
            *target = Some(Variable::Poly(poly));
        }
        Typ::PointBase { log_n } => todo!(),
        Typ::Scalar => {
            only_cpu!(device.clone());
            *target = Some(Variable::Scalar(T::Field::ZERO));
        }
        Typ::Transcript => todo!(),
        Typ::Point => todo!(),
        Typ::Tuple(vec) => todo!(),
        Typ::Array(typ, _) => todo!(),
        Typ::Any(type_id, _) => todo!(),
    };
}

pub fn deallocate<T: RuntimeType>(
    info: &RuntimeInfo<T>,
    id: VariableId,
    mem_allocator: &Option<PinnedMemoryPool>,
) {
    let mut target = info.variable[id].write().unwrap();
    if let Some(var) = target.as_ref() {
        match var {
            Variable::Poly(poly) => match poly.device {
                DeviceType::CPU => {
                    mem_allocator.as_ref().unwrap().deallocate(poly.values);
                    *target = None;
                }
                DeviceType::GPU { device_id } => {
                    *target = None;
                }
                DeviceType::Disk => todo!(),
            },
            Variable::PointBase => todo!(),
            Variable::Scalar(_) => todo!(),
            Variable::Transcript => todo!(),
            Variable::Point => todo!(),
            Variable::Tuple(vec) => todo!(),
            Variable::Array(_) => todo!(),
            Variable::Any(any) => todo!(),
        }
    }
}
