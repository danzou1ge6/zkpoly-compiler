use group::prime::PrimeCurveAffine;
use zkpoly_common::typ::Typ;
use zkpoly_cuda_api::{mem::CudaAllocator, stream::CudaStream};
use zkpoly_memory_pool::PinnedMemoryPool;

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
        mem_allocator: &Option<PinnedMemoryPool>,
        gpu_allocator: &Option<Vec<CudaAllocator>>,
    ) -> Variable<T> {
        match typ {
            Typ::ScalarArray { len, meta: _ } => {
                let poly = match device {
                    DeviceType::CPU => ScalarArray::<T::Field>::new(
                        len as usize,
                        mem_allocator.as_ref().unwrap().allocate(len as usize),
                        device.clone(),
                    ),
                    DeviceType::GPU { device_id } => ScalarArray::<T::Field>::new(
                        len as usize,
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
                        len as usize,
                        mem_allocator.as_ref().unwrap().allocate(len as usize),
                        device.clone(),
                    ),
                    DeviceType::GPU { device_id } => PointArray::<T::PointAffine>::new(
                        len as usize,
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
                let device = device.unwrap_gpu();
                Variable::GpuBuffer(GpuBuffer {
                    ptr: gpu_allocator.as_ref().unwrap()[device as usize].allocate(offset.unwrap()),
                    size: size as usize,
                })
            }
        }
    }

    pub(super) fn deallocate(
        &self,
        var: &mut Variable<T>,
        var_id: VariableId,
        mem_allocator: &Option<PinnedMemoryPool>,
    ) {
        match var {
            Variable::ScalarArray(poly) => {
                match poly.device {
                    DeviceType::CPU => {
                        mem_allocator.as_ref().unwrap().free(poly.values);
                    }
                    _ => {}
                }
            }
            Variable::PointArray(point_base) => match point_base.device {
                DeviceType::CPU => {
                    mem_allocator.as_ref().unwrap().free(point_base.values);
                }
                _ => {}
            },
            Variable::Tuple(vec) => {
                for var in vec {
                    self.deallocate(var, var_id, mem_allocator);
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
                _ => {}
            },
            Variable::Any(any) => {
                // deallocate the payload
                any.dealloc();
            }
            _ => {}
        }
    }
}
