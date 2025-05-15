use crate::bindings::*;
use std::{collections::BTreeMap, ffi::c_void};

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
    check_overlap: bool,
    ranges: BTreeMap<usize, usize>, // [left, right)
}

impl CudaAllocator {
    pub fn new(device_id: i32, max_size: usize, check_overlap: bool) -> Self {
        let mut base_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            cuda_check!(cudaSetDevice(device_id));
            cuda_check!(cudaMalloc(&mut base_ptr, max_size));
        }
        Self {
            device_id,
            base_ptr,
            max_size,
            check_overlap,
            ranges: BTreeMap::new(),
        }
    }

    pub fn allocate<F: Sized>(&mut self, offset: usize, len: usize) -> *mut F {
        assert!(len > 0);
        assert!(offset < self.max_size);
        let left = offset;
        let right = offset + len * std::mem::size_of::<F>();
        assert!(right <= self.max_size);
        if self.check_overlap {
            // println!("Allocating range: {} - {}", left, right);
            let gap = self.ranges.lower_bound(std::ops::Bound::Included(&left));
            if let Some((pred_start, pre_end)) = gap.peek_prev() {
                if !(pre_end <= &left) {
                    panic!(
                        "overlap detected: trying to allocate [{}, {}), overlapping with [{}, {})",
                        left, right, pred_start, pre_end
                    )
                }
            }
            if let Some((next_start, next_end)) = gap.peek_next() {
                if !(next_start >= &right) {
                    panic!(
                        "overlap detected: trying to allocate [{}, {}), overlapping with [{}, {})",
                        left, right, next_start, next_end
                    )
                }
            }
            self.ranges.insert(left, right);
        }
        unsafe { self.base_ptr.offset(offset.try_into().unwrap()) as *mut F }
    }

    pub fn free<F: Sized>(&mut self, ptr: *mut F) {
        if self.check_overlap {
            let offset = unsafe { (ptr as *mut c_void).offset_from(self.base_ptr) } as usize;
            assert!(self.ranges.contains_key(&offset));
            // println!("Freeing range: {}", offset);
            self.ranges.remove(&offset);
        }
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
    let mut allocator = CudaAllocator::new(device_id, max_size, false);
    let offset = 0;
    let ptr: *mut c_void = allocator.allocate(offset, 9 * size_of::<i32>());
    assert!(!ptr.is_null());
    let va = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut vb = [9, 8, 7, 6, 5, 4, 3, 2, 1];
    cuda_h2d(ptr, va.as_ptr() as *const c_void, 9 * size_of::<i32>());
    cuda_d2h(vb.as_mut_ptr() as *mut c_void, ptr, 9 * size_of::<i32>());
    allocator.free(ptr);
    assert_eq!(va, vb);
}

#[test]
fn test_cuda_allocator_overlap() {
    let device_id = 0;
    let max_size = 1024;
    let mut allocator = CudaAllocator::new(device_id, max_size, true);

    // 测试1：正常分配不重叠的内存区域
    let ptr1: *mut i32 = allocator.allocate(0, 4); // 0-16 bytes
    let ptr2: *mut i32 = allocator.allocate(16, 4); // 16-32 bytes
    assert!(!ptr1.is_null());
    assert!(!ptr2.is_null());

    // 测试3：释放内存后可以重新分配
    allocator.free(ptr1);
    let ptr3: *mut i32 = allocator.allocate(0, 4); // 重新分配0-16 bytes
    assert!(!ptr3.is_null());

    // 测试4：相邻分配
    allocator.free(ptr2);
    allocator.free(ptr3);
    let ptr4: *mut i32 = allocator.allocate(0, 4); // 0-16 bytes
    let ptr5: *mut i32 = allocator.allocate(16, 4); // 16-32 bytes，与前一个分配相邻
    assert!(!ptr4.is_null());
    assert!(!ptr5.is_null());

    // 清理
    allocator.free(ptr4);
    allocator.free(ptr5);
}

#[test]
#[should_panic]
fn test_cuda_allocator_overlap_should_panic() {
    let device_id = 0;
    let max_size = 1024;
    let mut allocator = CudaAllocator::new(device_id, max_size, true);

    // 首先分配一块内存
    let _ptr1: *mut i32 = allocator.allocate(0, 4); // 0-16 bytes

    // 尝试分配重叠区域，这会导致panic
    allocator.allocate::<i32>(8, 4); // 8-24 bytes，与前面的分配有重叠
}

#[test]
#[should_panic]
fn test_cuda_allocator_zero_size_should_panic() {
    let device_id = 0;
    let max_size = 1024;
    let mut allocator = CudaAllocator::new(device_id, max_size, true);

    // 尝试分配大小为0的内存，这会导致panic
    allocator.allocate::<i32>(32, 0);
}
