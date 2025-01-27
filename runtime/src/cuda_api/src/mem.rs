use crate::bindings::*;
use std::ffi::c_void;

use crate::cuda_check;

pub fn alloc_pinned<T: Sized>(len: usize) -> *mut T {
    let mut ptr: *mut T = std::ptr::null_mut();
    unsafe {
        cuda_check!(cudaMallocHost(
            &mut ptr as *mut *mut T as *mut *mut c_void,
            len * std::mem::size_of::<T>()
        ));
    }
    ptr
}

pub fn free_pinned<T: Sized>(ptr: *mut T) {
    unsafe {
        cuda_check!(cudaFreeHost(ptr as *mut c_void));
    }
}

pub struct CudaAllocator {
    device_id: i32,
    base_ptr: *mut std::ffi::c_void,
    max_size: usize,
}

impl CudaAllocator {
    pub fn new(device_id: i32, max_size: usize) -> Self {
        let mut base_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            cuda_check!(cudaSetDevice(device_id));
            cuda_check!(cudaMalloc(&mut base_ptr, max_size));
        }
        Self {
            device_id,
            base_ptr,
            max_size,
        }
    }

    pub fn allocate<F: Sized>(&self, offset: usize) -> *mut F {
        assert!(offset < self.max_size);
        unsafe { self.base_ptr.offset(offset.try_into().unwrap()) as *mut F }
    }
}

impl Drop for CudaAllocator {
    fn drop(&mut self) {
        unsafe {
            cuda_check!(cudaSetDevice(self.device_id));
            cuda_check!(cudaFree(self.base_ptr));
        }
    }
}

pub fn cuda_h2d(dst: *mut c_void, src: *const c_void, size: usize) {
    unsafe {
        cuda_check!(cudaMemcpy(
            dst,
            src,
            size,
            cudaMemcpyKind_cudaMemcpyHostToDevice
        ));
    }
}

pub fn cuda_d2h(dst: *mut c_void, src: *const c_void, size: usize) {
    unsafe {
        cuda_check!(cudaMemcpy(
            dst,
            src,
            size,
            cudaMemcpyKind_cudaMemcpyDeviceToHost
        ));
    }
}

#[test]
fn test_cuda_allocator() {
    let device_id = 0;
    let max_size = 1 << 20;
    let allocator = CudaAllocator::new(device_id, max_size);
    let offset = 0;
    let ptr: *mut c_void = allocator.allocate(offset);
    assert!(!ptr.is_null());
    let va = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut vb = [9, 8, 7, 6, 5, 4, 3, 2, 1];
    cuda_h2d(ptr, va.as_ptr() as *const c_void, 9 * size_of::<i32>());
    cuda_d2h(vb.as_mut_ptr() as *mut c_void, ptr, 9 * size_of::<i32>());
    assert_eq!(va, vb);
}
