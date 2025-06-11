use std::{collections::BTreeMap, ffi::{CStr, OsStr}, os::raw::c_void};

use zkpoly_cuda_api::{bindings::{cudaError_cudaSuccess, cudaFreeHost, cudaGetErrorString, cudaMallocHost}, cuda_check};

struct SanityChecker {
    /// A mapping from start to end of allocated ranges.
    ranges: BTreeMap<usize, usize>,
}

impl SanityChecker {
    pub fn new() -> Self {
        Self {
            ranges: BTreeMap::new(),
        }
    }

    pub fn allocate(&mut self, left: usize, right: usize) {
        let gap = self.ranges.lower_bound(std::ops::Bound::Included(&left));
        if let Some((pred_start, pred_end)) = gap.peek_prev() {
            if !(pred_end <= &left) {
                panic!(
                    "range collision detected: trying to allocate [{}, {}), overlapping with [{}, {})",
                    left, right, pred_start, pred_end
                )
            }
        }
        if let Some((next_start, next_end)) = gap.peek_next() {
            if !(next_start >= &right) {
                panic!(
                    "range collision detected: trying to allocate [{}, {}), overlapping with [{}, {})",
                    left, right, next_start, next_end
                )
            }
        }
    }

    pub fn free(&mut self, left: usize) {
        if !self.ranges.contains_key(&left) {
            panic!("freeing unallocated range [{},?)", left);
        }
        self.ranges.remove(&left);
    }
}

pub struct StaticAllocator {
    capacity: usize,
    checker: Option<SanityChecker>,
    base_ptr: *mut u8,
}



fn allocate_pinned_memory(capacity: usize) -> *mut u8 {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        cuda_check!(cudaMallocHost(&mut ptr, capacity))
    };

    ptr as *mut u8
}

fn free_pinned_memory(ptr: *mut u8) {
    unsafe {
        cuda_check!(cudaFreeHost(ptr as *mut c_void));
    }
}

impl StaticAllocator {
    pub fn new(capacity: usize, check: bool) -> Self {
        Self {
            capacity,
            checker: if check {
                Some(SanityChecker::new())
            } else {
                None
            },
            base_ptr: allocate_pinned_memory(capacity),
        }
    }

    pub fn allocate(&mut self, offset: usize, size: usize) -> *mut u8 {
        let left = offset;
        let right = offset + size;
        if let Some(checker) = &mut self.checker {
            checker.allocate(left, right);
        }
        if right > self.capacity {
            panic!(
                "trying to allocate [{}, {}), but capacity is {}",
                left, right, self.capacity
            );
        }

        self.base_ptr.with_addr(left)
    }

    pub fn deallocate(&mut self, ptr: *mut u8) {
        if let Some(checker) = &mut self.checker {
            let offset = ptr.addr() - self.base_ptr.addr();
            checker.free(offset);
        }
    }
}

impl Drop for StaticAllocator {
    fn drop(&mut self) {
        free_pinned_memory(self.base_ptr);
    }
}
