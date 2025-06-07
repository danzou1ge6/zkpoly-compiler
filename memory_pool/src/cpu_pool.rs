use super::buddy_memory_pool::MemoryPool; // Assuming MemoryPool is pub in lib.rs or its module
use std::ffi::c_void; // For NonNull<c_void> from MemoryPool
use std::mem;
use std::ptr::NonNull;

/// A memory pool that provides pinned host memory allocations,
/// wrapping the underlying buddy system MemoryPool.
#[derive(Debug)]
pub struct CpuMemoryPool {
    pool: MemoryPool,
    max_log_factor: u32, // Store for log_factor calculation and validation
    base_size: usize,    // Store for log_factor calculation
}

impl CpuMemoryPool {
    /// Creates a new PinnedMemoryPool.
    ///
    /// # Arguments
    /// * `max_log_factor`: The maximum log factor for slab sizes. The largest slab will be `base_size * 2^max_log_factor`.
    /// * `base_size`: The size of the smallest allocatable unit (slab with log_factor 0).
    pub fn new(max_log_factor: u32, base_size: usize) -> Self {
        Self {
            pool: MemoryPool::new(max_log_factor, base_size),
            max_log_factor,
            base_size,
        }
    }

    pub fn use_mmap(mut self) -> Self {
        self.pool.use_mmap();
        self
    }

    /// Allocates a block of memory suitable for `len` elements of type `T`.
    ///
    /// The actual allocated size will be rounded up to the nearest power-of-two
    /// multiple of `base_size`.
    ///
    /// # Panics
    /// Panics if the required `log_factor` exceeds `max_log_factor` or if allocation fails.
    pub fn allocate<T: Sized>(&mut self, len: usize) -> *mut T {
        let size_in_bytes = mem::size_of::<T>()
            .checked_mul(len)
            .expect("Allocation size overflow");
        if size_in_bytes == 0 {
            // Consistent with some allocators, return a dangling but non-null pointer for zero-size allocations.
            // Or, one might choose to return null or panic.
            // For now, let's align with returning a unique pointer.
            // However, the underlying pool might not support zero-size log_factor.
            // Let's require a non-zero size for meaningful log_factor calculation.
            if len > 0 && mem::size_of::<T>() == 0 {
                // ZSTs of non-zero length
                return NonNull::<T>::dangling().as_ptr();
            }
            // For truly zero bytes, it's tricky. Let's assume non-zero allocations for now.
            // Or handle by allocating smallest possible block if size_in_bytes is 0.
            // For simplicity, if size_in_bytes is 0, we might panic or return null.
            // Let's proceed assuming size_in_bytes > 0 for log_factor calculation.
            if size_in_bytes == 0 {
                // To avoid issues with log_factor of 0-size, if T is ZST, len > 0, return dangling.
                // If len is 0, or T is not ZST but size_in_bytes is 0 (e.g. T is ZST, len is 0),
                // it's a bit ambiguous. Let's return dangling for ZSTs.
                // If not ZST and size is 0, it's like asking for 0 bytes.
                // The C++ version's allocate(log_factor) implies log_factor >= 0.
                // A size of 0 would lead to issues with div_ceil or log_factor calculation.
                // For now, let's assume valid, positive size_in_bytes for pool allocation.
                // If T is a ZST, size_of::<T>() is 0.
                if mem::size_of::<T>() == 0 {
                    return NonNull::<T>::dangling().as_ptr();
                } else {
                    // size_of::<T>() > 0 but len is 0.
                    // This means 0 bytes requested. What log_factor?
                    // The original C++ code takes log_factor directly.
                    // Let's assume if len is 0, we return null or panic.
                    // For now, to match *mut T, let's return a null-like dangling pointer if it's what's expected.
                    // Or, more robustly, panic or return Option.
                    // Given the *mut T return, panic on invalid input is one way.
                    panic!("Cannot allocate 0 elements of a non-ZST type or 0 total bytes meaningfully with this allocator design.");
                }
            }
        }

        // Calculate log_factor based on total size in bytes
        let num_base_chunks = size_in_bytes.saturating_add(self.base_size - 1) / self.base_size; // Equivalent to div_ceil

        let log_factor = if num_base_chunks == 0 {
            // If size_in_bytes is 0 (and base_size > 0), num_base_chunks is 0.
            // The original user code: (size.div_ceil(self.base_size)).next_power_of_two().trailing_zeros();
            // If size is 0, div_ceil(0, X) is 0. 0.next_power_of_two() is 1. trailing_zeros() is 0.
            // So log_factor should be 0 for 0 size.
            0
        } else {
            num_base_chunks.next_power_of_two().trailing_zeros()
        };

        if log_factor > self.max_log_factor {
            panic!(
                "Required log_factor {} (for size {} bytes, {} elements of size {}) exceeds max_log_factor {}. Base size: {}",
                log_factor, size_in_bytes, len, mem::size_of::<T>(), self.max_log_factor, self.base_size
            );
        }

        match self.pool.allocate(log_factor) {
            Ok(ptr_nn) => ptr_nn.as_ptr() as *mut T,
            Err(e) => panic!(
                "PinnedMemoryPool: allocation failed for log_factor {}: {}",
                log_factor, e
            ),
        }
    }

    /// Pre-allocates a number of largest possible slabs and immediately frees them.
    /// This is intended to warm up the pool or reserve physical memory.
    pub fn preallocate(&mut self, num_slabs: usize) {
        let mut alloc_ptrs: Vec<NonNull<c_void>> = Vec::with_capacity(num_slabs);
        for _ in 0..num_slabs {
            match self.pool.allocate(self.max_log_factor) {
                Ok(ptr_nn) => {
                    // if is mmaped, we have to visit the memory to ensure it's allocated
                    if self.pool.is_mmaped() {
                        for i in (0..self.base_size * (1 << self.max_log_factor)).step_by(4 * 1024)
                        // 4KB
                        {
                            unsafe {
                                // SAFETY: This is a no-op, just to ensure the memory is touched.
                                *(ptr_nn.as_ptr().add(i) as *mut u8) = 0;
                            }
                        }
                    }
                    alloc_ptrs.push(ptr_nn)
                }
                Err(e) => {
                    // Log or handle preallocation failure for one slab
                    // For simplicity, we can panic or just print an error and continue
                    eprintln!(
                        "PinnedMemoryPool: preallocate failed to allocate a slab: {}",
                        e
                    );
                    // Depending on desired behavior, we might stop or try to free what was allocated.
                    // For now, just break if one fails, after freeing successful ones.
                    break;
                }
            }
        }
        for ptr_nn in alloc_ptrs {
            if let Err(e) = self.pool.deallocate(ptr_nn) {
                eprintln!(
                    "PinnedMemoryPool: preallocate failed to deallocate a slab: {}",
                    e
                );
                // This is more problematic as memory might be leaked by the pool's logic if dealloc fails.
            }
        }
    }

    /// Frees a previously allocated block of memory.
    ///
    /// # Safety
    /// The pointer `ptr` must have been previously allocated by `allocate` from this pool
    /// and not yet freed. The type `T` must be the same as when allocated.
    pub fn free<T: Sized>(&mut self, ptr: *mut T) {
        if ptr.is_null() {
            // Or handle as no-op, depending on desired behavior for null pointers.
            // The underlying pool might panic or error if ptr is not from an active allocation.
            // For ZSTs, ptr might be dangling.
            if mem::size_of::<T>() == 0 {
                // For ZSTs, ptr might be dangling. Freeing it is a no-op.
                return;
            }
            // For non-ZSTs, a null ptr typically means nothing to free or an error.
            // Let's assume non-null for actual deallocation by the pool.
            // If the user passes null, and it wasn't from a ZST allocation, it's likely an error.
            // The underlying `deallocate` expects a `NonNull`.
            return;
        }
        // SAFETY: Caller guarantees ptr is valid and from this pool.
        match NonNull::new(ptr as *mut c_void) {
            Some(ptr_nn) => {
                if let Err(e) = self.pool.deallocate(ptr_nn) {
                    panic!("PinnedMemoryPool: deallocation failed: {}", e);
                }
            }
            None => {
                // This case should ideally not be reached if ptr is not null,
                // but as a safeguard if ptr was somehow non-null but became null for c_void.
                // However, if ptr is non-null, NonNull::new will succeed.
                // This path is more for if ptr was already null and we didn't early exit.
                panic!("PinnedMemoryPool: tried to free a null pointer that wasn't caught by initial check.");
            }
        }
    }

    /// Clears the memory pool, freeing all allocated physical memory.
    pub fn clear(&mut self) {
        if let Err(e) = self.pool.clear() {
            panic!("PinnedMemoryPool: clear failed: {}", e);
        }
    }

    /// Tries to shrink the memory pool by freeing unused top-level slabs.
    pub fn shrink(&mut self) {
        if let Err(e) = self.pool.shrink() {
            panic!("PinnedMemoryPool: shrink failed: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use std::slice;

    #[test]
    fn test_memory_pool_pinned_wrapper() {
        let max_log = 10; // Smaller than original test's 16 for speed
        let base_s = mem::size_of::<u32>();
        let mut pool = CpuMemoryPool::new(max_log, base_s);

        let iters = 10; // Reduced iterations
        let items_per_iter = 100; // Reduced items

        for iter_num in 0..iters {
            let mut rng = rand::thread_rng();
            let mut allocations: Vec<(*mut u32, usize)> = Vec::new(); // ptr, len

            for item_num in 0..items_per_iter {
                // Ensure len results in log_factor <= max_log
                // Max size = base_s * 2^max_log
                // len = Max size / sizeof(u32)
                let max_len_for_type = (base_s * (1 << max_log)) / mem::size_of::<u32>();
                let len: usize = rng.gen_range(0..=max_len_for_type.min(1 << 15)); // Cap len for practical test slice sizes
                                                                                   // min(1<<15) is arbitrary cap for test speed

                if mem::size_of::<u32>() == 0 && len > 0 {
                    // ZST case
                    let ptr = pool.allocate::<u32>(len);
                    assert!(
                        !ptr.is_null(),
                        "ZST allocation should return non-null (dangling) pointer"
                    );
                    // Can't really write to ZST slices in a meaningful way for this test structure.
                    // We'll just allocate and free.
                    allocations.push((ptr, len));
                    continue;
                } else if len == 0 {
                    // Zero length non-ZST
                    // The current allocate panics on 0 len for non-ZST.
                    // If we want to test this, we need to expect panic or change allocate.
                    // For now, let's generate len > 0 for non-ZSTs.
                    // Or, if allocate is changed to return Option/Result, test that.
                    // Let's adjust gen_range to be 1..=max_len for non-ZSTs for this test.
                    continue; // Skip 0-len for non-ZSTs in this test version
                }

                let ptr = pool.allocate::<u32>(len);
                assert!(
                    !ptr.is_null(),
                    "Allocation failed for iter {}, item {}, len {}",
                    iter_num,
                    item_num,
                    len
                );
                allocations.push((ptr, len));

                // Write to the allocated memory
                // SAFETY: ptr is assumed valid and allocated for `len` u32s.
                let slice = unsafe { slice::from_raw_parts_mut(ptr, len) };
                for (idx, val) in slice.iter_mut().enumerate() {
                    *val = idx as u32;
                }
            }

            allocations.shuffle(&mut rng);

            for (ptr, len) in &allocations {
                if mem::size_of::<u32>() == 0 && *len > 0 {
                    // ZST
                    // No data to check for ZSTs
                    continue;
                }
                if *len == 0 {
                    continue;
                }

                // SAFETY: ptr was valid, len matches.
                let slice = unsafe { slice::from_raw_parts(*ptr, *len) };
                for (idx, val) in slice.iter().enumerate() {
                    assert_eq!(*val, idx as u32, "Data mismatch before free");
                }
            }

            for (ptr, _len) in allocations {
                pool.free(ptr);
            }
        }
        pool.shrink();
        pool.clear();
    }

    #[test]
    fn test_preallocate_pinned() {
        let mut pool = CpuMemoryPool::new(10, mem::size_of::<u32>());
        pool.preallocate(5); // Preallocate 5 largest slabs
                             // Test basic allocation after preallocate
        let ptr = pool.allocate::<u32>(10);
        assert!(!ptr.is_null());
        // SAFETY: ptr is valid from allocate
        unsafe {
            *ptr = 123;
            assert_eq!(*ptr, 123);
        }
        pool.free(ptr);
    }

    #[test]
    #[should_panic]
    fn test_allocate_zero_len_non_zst() {
        let mut pool = CpuMemoryPool::new(5, mem::size_of::<u32>());
        pool.allocate::<u32>(0); // Should panic as per current allocate logic for non-ZSTs
    }
}
