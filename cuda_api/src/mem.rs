use std::ffi::c_void;

pub use cuda_allocator::CudaAllocator;

use crate::{
    bindings::{cudaFreeHost, cudaMallocHost},
    cuda_check,
};

pub mod cuda_allocator;
pub mod page_allocator;

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
