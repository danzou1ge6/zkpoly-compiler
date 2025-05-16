use group::prime::PrimeCurveAffine;
use zkpoly_common::typ::Typ;
use zkpoly_cuda_api::{mem::CudaAllocator, stream::CudaStream};
use zkpoly_memory_pool::CpuMemoryPool;

use crate::{
    any::AnyWrapper,
    args::{RuntimeType, Variable, VariableId},
    devices::DeviceType,
    gpu_buffer::GpuBuffer,
    point::PointArray,
    runtime::RuntimeInfo,
    scalar::{Scalar, ScalarArray},
    transcript::TranscriptObject,
};

impl<T: RuntimeType> RuntimeInfo<T> {
    pub(super) fn allocate(
        &self,
        device: DeviceType,
        typ: Typ,
        offset: Option<usize>,
        mem_allocator: &mut Option<&mut CpuMemoryPool>,
        gpu_allocator: &mut Option<&mut Vec<CudaAllocator>>,
    ) -> Variable<T> {
        match typ {
            Typ::ScalarArray { len, meta: _ } => {
                let poly = match device {
                    DeviceType::CPU => ScalarArray::<T::Field>::new(
                        len as usize,
                        mem_allocator.as_mut().unwrap().allocate(len as usize),
                        device.clone(),
                    ),
                    DeviceType::GPU { device_id } => ScalarArray::<T::Field>::new(
                        len as usize,
                        gpu_allocator.as_mut().unwrap()[device_id as usize]
                            .allocate(offset.unwrap(), len as usize),
                        device.clone(),
                    ),
                    DeviceType::Disk => todo!(),
                };
                Variable::ScalarArray(poly)
            }
            Typ::PointBase { len } => {
                let point_base = match device {
                    DeviceType::CPU => PointArray::<T::PointAffine>::new(
                        len as usize,
                        mem_allocator.as_mut().unwrap().allocate(len as usize),
                        device.clone(),
                    ),
                    DeviceType::GPU { device_id } => PointArray::<T::PointAffine>::new(
                        len as usize,
                        gpu_allocator.as_mut().unwrap()[device_id as usize]
                            .allocate(offset.unwrap(), len as usize),
                        device.clone(),
                    ),
                    DeviceType::Disk => todo!(),
                };
                Variable::PointArray(point_base)
            }
            Typ::Scalar => match device {
                DeviceType::CPU => Variable::Scalar(Scalar::new_cpu()),
                DeviceType::GPU { device_id } => Variable::Scalar(Scalar::new_gpu(
                    gpu_allocator.as_mut().unwrap()[device_id as usize]
                        .allocate(offset.unwrap(), 1),
                    device_id,
                )),
                DeviceType::Disk => unreachable!(),
            },
            Typ::Transcript => {
                assert!(device.is_cpu());
                Variable::Transcript(TranscriptObject::new_raw())
            }
            Typ::Point => {
                assert!(device.is_cpu());
                Variable::Point(crate::point::Point::new(T::PointAffine::identity()))
            }
            Typ::Tuple => unreachable!("Tuple can only be assembled"),
            Typ::Any(_, _) => {
                assert!(device.is_cpu());
                Variable::Any(AnyWrapper::new(Box::new(0))) // default payload
            }
            Typ::Stream => {
                let device = device.unwrap_gpu();
                Variable::Stream(CudaStream::new(device))
            }
            Typ::GpuBuffer(size) => {
                let device_id = device.unwrap_gpu();
                Variable::GpuBuffer(GpuBuffer {
                    ptr: gpu_allocator.as_mut().unwrap()[device_id as usize]
                        .allocate(offset.unwrap(), size),
                    size: size as usize,
                    device: device,
                })
            }
        }
    }

    pub(super) fn deallocate(
        &self,
        var: &mut Variable<T>,
        var_id: VariableId,
        mem_allocator: &mut Option<&mut CpuMemoryPool>,
        gpu_allocator: &mut Option<&mut Vec<CudaAllocator>>,
    ) {
        match var {
            Variable::ScalarArray(poly) => match poly.device {
                DeviceType::CPU => {
                    mem_allocator.as_mut().unwrap().free(poly.values);
                }
                DeviceType::GPU { device_id } => {
                    gpu_allocator.as_mut().unwrap()[device_id as usize].free(poly.values);
                }
                _ => unimplemented!(),
            },
            Variable::PointArray(point_base) => match point_base.device {
                DeviceType::CPU => {
                    mem_allocator.as_mut().unwrap().free(point_base.values);
                }
                DeviceType::GPU { device_id } => {
                    gpu_allocator.as_mut().unwrap()[device_id as usize].free(point_base.values);
                }
                _ => unimplemented!(),
            },
            Variable::Tuple(vec) => {
                for var in vec {
                    self.deallocate(var, var_id, mem_allocator, gpu_allocator);
                }
            }
            Variable::Stream(stream) => {
                stream.destroy();
            }
            Variable::Point(point) => {
                point.deallocate();
            }
            Variable::Transcript(transcript) => {
                transcript.deallocate();
            }
            Variable::Scalar(scalar) => match scalar.device {
                DeviceType::CPU => {
                    scalar.deallocate();
                }
                DeviceType::GPU { device_id } => {
                    gpu_allocator.as_mut().unwrap()[device_id as usize].free(scalar.value);
                }
                _ => unimplemented!(),
            },
            Variable::Any(any) => {
                // deallocate the payload
                any.dealloc();
            }
            Variable::GpuBuffer(gpu_buffer) => {
                let device_id = gpu_buffer.device.unwrap_gpu();
                gpu_allocator.as_mut().unwrap()[device_id as usize].free(gpu_buffer.ptr);
            }
            _ => {}
        }
    }
}
