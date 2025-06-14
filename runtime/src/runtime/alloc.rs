use std::collections::HashMap;

use group::prime::PrimeCurveAffine;
use zkpoly_common::{devices::DeviceType, typ::Typ};
use zkpoly_cuda_api::{mem::CudaAllocator, stream::CudaStream};
use zkpoly_memory_pool::{static_allocator::CpuStaticAllocator, BuddyDiskPool};

use crate::{
    any::AnyWrapper,
    args::{RuntimeType, Variable, VariableId},
    gpu_buffer::GpuBuffer,
    instructions::{AllocMethod, AllocVariant},
    point::PointArray,
    runtime::RuntimeInfo,
    scalar::{Scalar, ScalarArray},
    transcript::TranscriptObject,
};

fn unsupported_alloc_method(am: AllocMethod, on: DeviceType) -> ! {
    panic!("unsupported alloc method {:?} on {:?}", am, on)
}

fn unsupported_alloc_variant(av: AllocVariant, on: DeviceType) -> ! {
    panic!("unsupported alloc variant {:?} on {:?}", av, on)
}

fn get_gpu_allocator<'a, 'b, 'c>(
    i: i32,
    gpu_allocator: &'a mut Option<&'b mut HashMap<i32, CudaAllocator>>,
) -> &'c mut CudaAllocator
where
    'b: 'c,
    'a: 'c,
{
    gpu_allocator.as_mut().unwrap().get_mut(&i).unwrap()
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
                                get_gpu_allocator(device_id, gpu_allocator)
                                    .statik
                                    .allocate(offset, len as usize),
                                DeviceType::GPU { device_id },
                            ),
                            AllocMethod::Paged { va_size, pa } => ScalarArray::<T::Field>::new(
                                len as usize,
                                get_gpu_allocator(device_id, gpu_allocator)
                                    .page
                                    .allocate(va_size, pa),
                                DeviceType::GPU { device_id },
                            ),
                            otherwise => unsupported_alloc_method(otherwise, device),
                        }
                    }
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
                                get_gpu_allocator(device_id, gpu_allocator)
                                    .statik
                                    .allocate(offset, len as usize),
                                DeviceType::GPU { device_id },
                            ),
                            AllocMethod::Paged { va_size, pa } => {
                                PointArray::<T::PointAffine>::new(
                                    len as usize,
                                    get_gpu_allocator(device_id, gpu_allocator)
                                        .page
                                        .allocate(va_size, pa),
                                    DeviceType::GPU { device_id },
                                )
                            }
                            otherwise => unsupported_alloc_method(otherwise, device),
                        }
                    }
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
                            get_gpu_allocator(device_id, gpu_allocator)
                                .statik
                                .allocate(offset, 1),
                            device_id,
                        )),
                        AllocMethod::Paged { va_size, pa } => Variable::Scalar(Scalar::new_gpu(
                            get_gpu_allocator(device_id, gpu_allocator)
                                .page
                                .allocate(va_size, pa),
                            device_id,
                        )),
                        otherwise => unsupported_alloc_method(otherwise, device),
                    }
                }
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
                        ptr: get_gpu_allocator(device_id, gpu_allocator)
                            .statik
                            .allocate(offset, size),
                        size: size as usize,
                        device: DeviceType::GPU { device_id },
                    }),
                    AllocMethod::Paged { va_size, pa } => Variable::GpuBuffer(GpuBuffer {
                        ptr: get_gpu_allocator(device_id, gpu_allocator)
                            .page
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
        _var_id: VariableId,
        alloc_method: AllocVariant,
        mem_allocator: &mut Option<&mut CpuStaticAllocator>,
        gpu_allocator: &mut Option<&mut HashMap<i32, CudaAllocator>>,
        disk_allocator: &mut Option<&mut Vec<BuddyDiskPool>>,
    ) {
        match var {
            Variable::ScalarArray(poly) => match poly.device {
                DeviceType::CPU => match alloc_method {
                    AllocVariant::Offset => {
                        mem_allocator
                            .as_mut()
                            .unwrap()
                            .deallocate(poly.values as *mut u8);
                    }
                    otherwise => unsupported_alloc_variant(otherwise, poly.device.clone()),
                },
                DeviceType::GPU { device_id } => match alloc_method {
                    AllocVariant::Offset => {
                        get_gpu_allocator(device_id, gpu_allocator)
                            .statik
                            .free(poly.values);
                    }
                    AllocVariant::Paged => {
                        // For now, we don't unmap pages
                    }
                    otherwise => unsupported_alloc_variant(otherwise, poly.device.clone()),
                },
                DeviceType::Disk => match alloc_method {
                    AllocVariant::Dynamic => {
                        let bytes = poly.len() * size_of::<T::Field>() / poly.disk_pos.len();
                        disk_allocator
                            .as_mut()
                            .unwrap()
                            .iter_mut()
                            .zip(poly.disk_pos.iter())
                            .for_each(|(disk_pool, dai)| {
                                disk_pool
                                    .deallocate(dai.offset)
                                    .expect("deallocation failed");
                            });
                    }
                    otherwise => unsupported_alloc_variant(otherwise, poly.device.clone()),
                },
            },
            Variable::PointArray(point_base) => match point_base.device {
                DeviceType::CPU => match alloc_method {
                    AllocVariant::Offset => {
                        mem_allocator
                            .as_mut()
                            .unwrap()
                            .deallocate(point_base.values as *mut u8);
                    }
                    otherwise => unsupported_alloc_variant(otherwise, point_base.device.clone()),
                },
                DeviceType::GPU { device_id } => match alloc_method {
                    AllocVariant::Offset => {
                        get_gpu_allocator(device_id, gpu_allocator)
                            .statik
                            .free(point_base.values);
                    }
                    AllocVariant::Paged => {
                        // For now, we don't unmap pages
                    }
                    otherwise => unsupported_alloc_variant(otherwise, point_base.device.clone()),
                },
                _ => unimplemented!(),
            },
            Variable::Tuple(..) => {
                panic!("tuple must be deallocated element by element")
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
            Variable::Scalar(scalar) => {
                if alloc_method != AllocVariant::Offset {
                    panic!("scalars should be allocated using offset");
                }

                match scalar.device {
                    DeviceType::CPU => {
                        scalar.deallocate();
                    }
                    DeviceType::GPU { device_id } => {
                        gpu_allocator
                            .as_mut()
                            .unwrap()
                            .get_mut(&device_id)
                            .unwrap()
                            .statik
                            .free(scalar.value);
                    }
                    _ => unimplemented!(),
                }
            }
            Variable::Any(any) => {
                // deallocate the payload
                any.dealloc();
            }
            Variable::GpuBuffer(gpu_buffer) => {
                let device_id = gpu_buffer.device.unwrap_gpu();
                match alloc_method {
                    AllocVariant::Offset => {
                        gpu_allocator
                            .as_mut()
                            .unwrap()
                            .get_mut(&device_id)
                            .unwrap()
                            .statik
                            .free(gpu_buffer.ptr);
                    }
                    AllocVariant::Paged => {
                        // For now, we don't unmap pages
                    }
                    otherwise => {
                        unsupported_alloc_variant(otherwise, DeviceType::GPU { device_id })
                    }
                }
            }
        }
    }
}
