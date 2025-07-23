use std::{collections::BTreeMap, os::raw::c_void};

use zkpoly_cuda_api::{
    bindings::{cudaFreeHost, cudaMallocHost},
    cuda_check,
};

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
        self.ranges.insert(left, right);
    }

    pub fn free(&mut self, left: usize) {
        if !self.ranges.contains_key(&left) {
            panic!("freeing unallocated range [{},?)", left);
        }
        self.ranges.remove(&left);
    }
}

pub struct CpuStaticAllocator {
    capacity: usize,
    checker: Option<SanityChecker>,
    base_ptr: *mut u8,
    primary: bool,
}

unsafe impl Send for CpuStaticAllocator {}

fn allocate_pinned_memory(capacity: usize) -> *mut u8 {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe { cuda_check!(cudaMallocHost(&mut ptr, capacity)) };

    ptr as *mut u8
}

fn free_pinned_memory(ptr: *mut u8) {
    unsafe {
        cuda_check!(cudaFreeHost(ptr as *mut c_void));
    }
}

impl CpuStaticAllocator {
    pub fn new(capacity: usize, check: bool) -> Self {
        Self {
            capacity,
            checker: if check {
                Some(SanityChecker::new())
            } else {
                None
            },
            base_ptr: allocate_pinned_memory(capacity),
            primary: true,
        }
    }

    pub fn slice(&self, offset: usize, size: usize) -> Self {
        Self {
            capacity: size,
            checker: self.checker.as_ref().map(|_| SanityChecker::new()),
            base_ptr: unsafe { self.base_ptr.byte_add(offset) },
            primary: false,
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

        unsafe { self.base_ptr.add(left) }
    }

    pub fn deallocate(&mut self, ptr: *mut u8) {
        if let Some(checker) = &mut self.checker {
            let offset = unsafe { ptr.offset_from(self.base_ptr) };
            checker.free(offset.try_into().expect("negative offset"));
        }
    }
}

impl Drop for CpuStaticAllocator {
    fn drop(&mut self) {
        if self.primary {
            free_pinned_memory(self.base_ptr);
        }
    }
}
