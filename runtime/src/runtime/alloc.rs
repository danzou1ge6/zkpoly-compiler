use group::prime::PrimeCurveAffine;
use zkpoly_common::{devices::DeviceType, typ::Typ};
use zkpoly_cuda_api::{
    mem::{page_allocator::CudaPageAllocator, CudaAllocator},
    stream::CudaStream,
};
use zkpoly_memory_pool::{BuddyDiskPool, CpuMemoryPool, SwapPagePool};

use crate::{
    any::AnyWrapper,
    args::{RuntimeType, Variable, VariableId},
    gpu_buffer::GpuBuffer,
    instructions::GpuAlloc,
    point::PointArray,
    runtime::RuntimeInfo,
    scalar::{PolyPtr, Scalar, ScalarArray},
    transcript::TranscriptObject,
};

impl<T: RuntimeType> RuntimeInfo<T> {
    pub(super) fn allocate(
        &self,
        device: DeviceType,
        typ: Typ,
        gpu_alloc: Option<GpuAlloc>,
        mem_allocator: &mut Option<&mut CpuMemoryPool>,
        gpu_allocator: &mut Option<&mut Vec<CudaAllocator>>,
        disk_allocator: &mut Option<&mut Vec<BuddyDiskPool>>,
        page_allocator: &mut Option<&mut Vec<CudaPageAllocator>>,
        swap_page_pool: &mut Option<&mut SwapPagePool>,
    ) -> Variable<T> {
        match typ {
            Typ::ScalarArray { len, meta: _ } => {
                let poly = match device {
                    DeviceType::CPU => ScalarArray::<T::Field>::new(
                        len as usize,
                        crate::scalar::PolyPtr::Swap(
                            swap_page_pool
                                .as_mut()
                                .unwrap()
                                .allocate(len as usize * size_of::<T::Field>())
                                .unwrap(),
                        ),
                        // mem_allocator.as_mut().unwrap().allocate(len as usize),
                        device.clone(),
                    ),
                    DeviceType::GPU { device_id } => match gpu_alloc.unwrap() {
                        GpuAlloc::Offset(offset) => ScalarArray::<T::Field>::new(
                            len as usize,
                            crate::scalar::PolyPtr::Raw(
                                gpu_allocator.as_mut().unwrap()[device_id as usize]
                                    .allocate(offset, len as usize),
                            ),
                            device.clone(),
                        ),
                        GpuAlloc::PageInfo { va_size, pa } => ScalarArray::<T::Field>::new(
                            len as usize,
                            crate::scalar::PolyPtr::Raw(
                                page_allocator.as_mut().unwrap()[device_id as usize]
                                    .allocate(va_size, &pa),
                            ),
                            device.clone(),
                        ),
                    },
                    DeviceType::Disk => ScalarArray::<T::Field>::alloc_disk(
                        len as usize,
                        disk_allocator.as_mut().unwrap(),
                    ),
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
                    DeviceType::GPU { device_id } => match gpu_alloc.unwrap() {
                        GpuAlloc::Offset(offset) => PointArray::<T::PointAffine>::new(
                            len as usize,
                            gpu_allocator.as_mut().unwrap()[device_id as usize]
                                .allocate(offset, len as usize),
                            device.clone(),
                        ),
                        GpuAlloc::PageInfo { va_size, pa } => PointArray::<T::PointAffine>::new(
                            len as usize,
                            page_allocator.as_mut().unwrap()[device_id as usize]
                                .allocate(va_size, &pa),
                            device.clone(),
                        ),
                    },
                    DeviceType::Disk => todo!(),
                };
                Variable::PointArray(point_base)
            }
            Typ::Scalar => match device {
                DeviceType::CPU => Variable::Scalar(Scalar::new_cpu()),
                DeviceType::GPU { device_id } => match gpu_alloc.unwrap() {
                    GpuAlloc::Offset(offset) => Variable::Scalar(Scalar::new_gpu(
                        gpu_allocator.as_mut().unwrap()[device_id as usize].allocate(offset, 1),
                        device_id,
                    )),
                    GpuAlloc::PageInfo { va_size, pa } => Variable::Scalar(Scalar::new_gpu(
                        page_allocator.as_mut().unwrap()[device_id as usize].allocate(va_size, &pa),
                        device_id,
                    )),
                },
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
                match gpu_alloc.unwrap() {
                    GpuAlloc::Offset(offset) => Variable::GpuBuffer(GpuBuffer {
                        ptr: gpu_allocator.as_mut().unwrap()[device_id as usize]
                            .allocate(offset, size),
                        size: size as usize,
                        device: device,
                    }),
                    GpuAlloc::PageInfo { va_size, pa } => Variable::GpuBuffer(GpuBuffer {
                        ptr: page_allocator.as_mut().unwrap()[device_id as usize]
                            .allocate(va_size, &pa),
                        size: size as usize,
                        device: device,
                    }),
                }
            }
        }
    }

    pub(super) fn deallocate(
        &self,
        var: &mut Variable<T>,
        var_id: VariableId,
        mem_allocator: &mut Option<&mut CpuMemoryPool>,
        gpu_allocator: &mut Option<&mut Vec<CudaAllocator>>,
        disk_allocator: &mut Option<&mut Vec<BuddyDiskPool>>,
        swap_page_pool: &mut Option<&mut SwapPagePool>,
    ) {
        match var {
            Variable::ScalarArray(poly) => match poly.device {
                DeviceType::CPU => match &poly.values {
                    PolyPtr::Swap(handle) => {
                        swap_page_pool.as_mut().unwrap().deallocate(handle.clone());
                    }
                    PolyPtr::Raw(ptr) => {
                        mem_allocator.as_mut().unwrap().free(*ptr);
                    }
                },
                DeviceType::GPU { device_id } => {
                    if let PolyPtr::Raw(ptr) = poly.values {
                        gpu_allocator.as_mut().unwrap()[device_id as usize].free(ptr);
                    }
                }
                DeviceType::Disk => {
                    let bytes = poly.len() * size_of::<T::Field>() / poly.disk_pos.len();
                    disk_allocator
                        .as_mut()
                        .unwrap()
                        .iter_mut()
                        .zip(poly.disk_pos.iter())
                        .for_each(|(disk_pool, (_, offset))| {
                            disk_pool.deallocate(*offset, bytes).unwrap();
                        });
                }
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
                    self.deallocate(var, var_id, mem_allocator, gpu_allocator, disk_allocator, swap_page_pool);
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
        }
    }
}
