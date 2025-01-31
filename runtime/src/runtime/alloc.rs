use group::prime::PrimeCurveAffine;
use zkpoly_common::typ::Typ;
use zkpoly_cuda_api::{mem::CudaAllocator, stream::CudaStream};
use zkpoly_memory_pool::PinnedMemoryPool;

use crate::{
    args::{RuntimeType, Variable},
    devices::DeviceType,
    gpu_buffer::GpuBuffer,
    point::PointArray,
    runtime::RuntimeInfo,
    scalar::Scalar,
    scalar::ScalarArray,
};

impl<T: RuntimeType> RuntimeInfo<T> {
    pub(super) fn allocate(
        &self,
        device: DeviceType,
        typ: Typ,
        offset: Option<usize>,
        mem_allocator: &Option<PinnedMemoryPool>,
        gpu_allocator: &Option<Vec<CudaAllocator>>,
    ) -> Variable<T> {
        match typ {
            Typ::ScalarArray { typ: _, len } => {
                let poly = match device {
                    DeviceType::CPU => ScalarArray::<T::Field>::new(
                        len,
                        mem_allocator.as_ref().unwrap().allocate(len.clone()),
                        device.clone(),
                    ),
                    DeviceType::GPU { device_id } => ScalarArray::<T::Field>::new(
                        len,
                        gpu_allocator.as_ref().unwrap()[device_id as usize]
                            .allocate(offset.unwrap()),
                        device.clone(),
                    ),
                    DeviceType::Disk => todo!(),
                };
                Variable::ScalarArray(poly)
            }
            Typ::PointBase { len } => {
                let point_base = match device {
                    DeviceType::CPU => PointArray::<T::PointAffine>::new(
                        len,
                        mem_allocator.as_ref().unwrap().allocate(len),
                        device.clone(),
                    ),
                    DeviceType::GPU { device_id } => PointArray::<T::PointAffine>::new(
                        len,
                        gpu_allocator.as_ref().unwrap()[device_id as usize]
                            .allocate(offset.unwrap()),
                        device.clone(),
                    ),
                    DeviceType::Disk => todo!(),
                };
                Variable::PointArray(point_base)
            }
            Typ::Scalar => match device {
                DeviceType::CPU => Variable::Scalar(Scalar::new_cpu()),
                DeviceType::GPU { device_id } => Variable::Scalar(Scalar::new_gpu(
                    gpu_allocator.as_ref().unwrap()[device_id as usize].allocate(offset.unwrap()),
                    device_id,
                )),
                DeviceType::Disk => unreachable!(),
            },
            Typ::Transcript => unreachable!(),
            Typ::Point => {
                assert!(device.is_cpu());
                Variable::Point(crate::point::Point::new(T::PointAffine::identity()))
            }
            Typ::Tuple(vec) => {
                let mut vars = Vec::new();
                for typ in vec {
                    vars.push(self.allocate(
                        device.clone(),
                        typ,
                        offset,
                        mem_allocator,
                        gpu_allocator,
                    ));
                }
                Variable::Tuple(vars)
            }
            Typ::Array(typ, len) => {
                let typ = *typ;
                let mut vars = Vec::new();
                for _ in 0..len {
                    vars.push(self.allocate(
                        device.clone(),
                        typ.clone(),
                        offset,
                        mem_allocator,
                        gpu_allocator,
                    ));
                }
                Variable::Array(vars.into_boxed_slice())
            }
            Typ::Any(_, _) => unreachable!(),
            Typ::Stream => {
                let device = device.unwrap_gpu();
                Variable::Stream(CudaStream::new(device))
            }
            Typ::GpuBuffer(size) => {
                let device = device.unwrap_gpu();
                Variable::GpuBuffer(GpuBuffer {
                    ptr: gpu_allocator.as_ref().unwrap()[device as usize].allocate(offset.unwrap()),
                    size,
                })
            }
            Typ::Rng => todo!(),
        }
    }

    pub(super) fn deallocate(
        &self,
        var: &mut Variable<T>,
        mem_allocator: &Option<PinnedMemoryPool>,
    ) {
        match var {
            Variable::ScalarArray(poly) => match poly.device {
                DeviceType::CPU => {
                    mem_allocator.as_ref().unwrap().free(poly.values);
                }
                _ => {}
            },
            Variable::PointArray(point_base) => match point_base.device {
                DeviceType::CPU => {
                    mem_allocator.as_ref().unwrap().free(point_base.values);
                }
                _ => {}
            },
            Variable::Tuple(vec) => {
                for var in vec {
                    self.deallocate(var, mem_allocator);
                }
            }
            Variable::Array(array) => {
                for var in array.iter_mut() {
                    self.deallocate(var, mem_allocator);
                }
            }
            _ => {}
        }
    }
}
