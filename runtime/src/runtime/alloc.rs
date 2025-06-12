use std::collections::HashMap;

use group::prime::PrimeCurveAffine;
use zkpoly_common::{devices::DeviceType, typ::Typ};
use zkpoly_cuda_api::{
    mem::{page_allocator::CudaPageAllocator, CudaAllocator},
    stream::CudaStream,
};
use zkpoly_memory_pool::{static_allocator::CpuStaticAllocator, BuddyDiskPool};

use crate::{
    any::AnyWrapper,
    args::{RuntimeType, Variable, VariableId},
    gpu_buffer::GpuBuffer,
    instructions::AllocMethod,
    point::PointArray,
    runtime::RuntimeInfo,
    scalar::{Scalar, ScalarArray},
    transcript::TranscriptObject,
};

fn unsupported_alloc_method(am: AllocMethod, on: DeviceType) -> ! {
    panic!("unsupported alloc method {:?} on {:?}", am, on)
}

impl<T: RuntimeType> RuntimeInfo<T> {
    pub(super) fn allocate(
        &self,
        device: DeviceType,
        typ: Typ,
        alloc_method: AllocMethod,
        mem_allocator: &mut Option<&mut CpuStaticAllocator>,
        gpu_allocator: &mut Option<&mut HashMap<i32, CudaAllocator>>,
        disk_allocator: &mut Option<&mut Vec<BuddyDiskPool>>,
        page_allocator: &mut Option<&mut Vec<CudaPageAllocator>>,
    ) -> Variable<T> {
        match typ {
            Typ::ScalarArray { len, meta: _ } => {
                let poly = match device {
                    DeviceType::CPU => match alloc_method {
                        AllocMethod::Offset(offset, size) => ScalarArray::<T::Field>::new(
                            len as usize,
                            mem_allocator.as_mut().unwrap().allocate(offset, size) as *mut T::Field,
                            DeviceType::CPU,
                        ),
                        otherwise => unsupported_alloc_method(otherwise, device),
                    },
                    DeviceType::GPU { device_id } => {
                        let device_id = (self.gpu_mapping)(device_id);
                        match alloc_method {
                        AllocMethod::Offset(offset, ..) => ScalarArray::<T::Field>::new(
                            len as usize,
                            gpu_allocator
                                .as_mut()
                                .unwrap()
                                .get_mut(&device_id)
                                .unwrap()
                                .allocate(offset, len as usize),
                            DeviceType::GPU { device_id },
                        ),
                        AllocMethod::Paged { va_size, pa } => ScalarArray::<T::Field>::new(
                            len as usize,
                            page_allocator.as_mut().unwrap()[device_id as usize]
                                .allocate(va_size, pa),
                            DeviceType::GPU { device_id },
                        ),
                        otherwise => unsupported_alloc_method(otherwise, device),
                    }},
                    DeviceType::Disk => match alloc_method {
                        AllocMethod::Dynamic(..) => ScalarArray::<T::Field>::alloc_disk(
                            len as usize,
                            disk_allocator.as_mut().unwrap(),
                        ),
                        otherwise => unsupported_alloc_method(otherwise, device),
                    },
                };
                Variable::ScalarArray(poly)
            }
            Typ::PointBase { len } => {
                let point_base = match device {
                    DeviceType::CPU => match alloc_method {
                        AllocMethod::Offset(offset, size) => PointArray::<T::PointAffine>::new(
                            len as usize,
                            mem_allocator.as_mut().unwrap().allocate(offset, size)
                                as *mut T::PointAffine,
                            DeviceType::CPU,
                        ),
                        otherwise => unsupported_alloc_method(otherwise, device),
                    },
                    DeviceType::GPU { device_id } => {
                        let device_id = (self.gpu_mapping)(device_id);
                        match alloc_method {
                        AllocMethod::Offset(offset, ..) => PointArray::<T::PointAffine>::new(
                            len as usize,
                            gpu_allocator
                                .as_mut()
                                .unwrap()
                                .get_mut(&device_id)
                                .unwrap()
                                .allocate(offset, len as usize),
                            DeviceType::GPU { device_id },
                        ),
                        AllocMethod::Paged { va_size, pa } => PointArray::<T::PointAffine>::new(
                            len as usize,
                            page_allocator.as_mut().unwrap()[device_id as usize]
                                .allocate(va_size, pa),
                            DeviceType::GPU { device_id },
                        ),
                        otherwise => unsupported_alloc_method(otherwise, device),
                    }},
                    DeviceType::Disk => todo!(),
                };
                Variable::PointArray(point_base)
            }
            Typ::Scalar => match device {
                DeviceType::CPU => Variable::Scalar(Scalar::new_cpu()),
                DeviceType::GPU { device_id } => {
                    let device_id = (self.gpu_mapping)(device_id);
                    match alloc_method {
                    AllocMethod::Offset(offset, ..) => Variable::Scalar(Scalar::new_gpu(
                        gpu_allocator
                            .as_mut()
                            .unwrap()
                            .get_mut(&device_id)
                            .unwrap().allocate(offset, 1),
                        device_id,
                    )),
                    AllocMethod::Paged { va_size, pa } => Variable::Scalar(Scalar::new_gpu(
                        page_allocator.as_mut().unwrap()[device_id as usize].allocate(va_size, pa),
                        device_id,
                    )),
                    otherwise => unsupported_alloc_method(otherwise, device),
                }},
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
                let device_id = (self.gpu_mapping)(device.unwrap_gpu());
                Variable::Stream(CudaStream::new(device_id))
            }
            Typ::GpuBuffer(size) => {
                let device_id = (self.gpu_mapping)(device.unwrap_gpu());
                match alloc_method {
                    AllocMethod::Offset(offset, ..) => Variable::GpuBuffer(GpuBuffer {
                        ptr: gpu_allocator
                            .as_mut()
                            .unwrap()
                            .get_mut(&device_id)
                            .unwrap()
                            .allocate(offset, size),
                        size: size as usize,
                        device: DeviceType::GPU { device_id },
                    }),
                    AllocMethod::Paged { va_size, pa } => Variable::GpuBuffer(GpuBuffer {
                        ptr: page_allocator.as_mut().unwrap()[device_id as usize]
                            .allocate(va_size, pa),
                        size: size as usize,
                        device: DeviceType::GPU { device_id },
                    }),
                    otherwise => unsupported_alloc_method(otherwise, device),
                }
            }
        }
    }

    pub(super) fn deallocate(
        &self,
        var: &mut Variable<T>,
        var_id: VariableId,
        mem_allocator: &mut Option<&mut CpuStaticAllocator>,
        gpu_allocator: &mut Option<&mut HashMap<i32, CudaAllocator>>,
        disk_allocator: &mut Option<&mut Vec<BuddyDiskPool>>,
    ) {
        match var {
            Variable::ScalarArray(poly) => match poly.device {
                DeviceType::CPU => {
                    mem_allocator
                        .as_mut()
                        .unwrap()
                        .deallocate(poly.values as *mut u8);
                }
                DeviceType::GPU { device_id } => {
                    gpu_allocator
                        .as_mut()
                        .unwrap()
                        .get_mut(&device_id)
                        .unwrap()
                        .free(poly.values);
                }
                DeviceType::Disk => {
                    let bytes = poly.len() * size_of::<T::Field>() / poly.disk_pos.len();
                    disk_allocator
                        .as_mut()
                        .unwrap()
                        .iter_mut()
                        .zip(poly.disk_pos.iter())
                        .for_each(|(disk_pool, (_, offset))| {
                            disk_pool
                                .deallocate(*offset, bytes)
                                .expect("deallocation failed");
                        });
                }
            },
            Variable::PointArray(point_base) => match point_base.device {
                DeviceType::CPU => {
                    mem_allocator
                        .as_mut()
                        .unwrap()
                        .deallocate(point_base.values as *mut u8);
                }
                DeviceType::GPU { device_id } => {
                    gpu_allocator
                        .as_mut()
                        .unwrap()
                        .get_mut(&device_id)
                        .unwrap()
                        .free(point_base.values);
                }
                _ => unimplemented!(),
            },
            Variable::Tuple(vec) => {
                for var in vec {
                    self.deallocate(var, var_id, mem_allocator, gpu_allocator, disk_allocator);
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
                    gpu_allocator
                        .as_mut()
                        .unwrap()
                        .get_mut(&device_id)
                        .unwrap()
                        .free(scalar.value);
                }
                _ => unimplemented!(),
            },
            Variable::Any(any) => {
                // deallocate the payload
                any.dealloc();
            }
            Variable::GpuBuffer(gpu_buffer) => {
                let device_id = gpu_buffer.device.unwrap_gpu();
                gpu_allocator
                    .as_mut()
                    .unwrap()
                    .get_mut(&device_id)
                    .unwrap()
                    .free(gpu_buffer.ptr);
            }
        }
    }
}
