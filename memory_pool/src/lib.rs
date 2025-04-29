include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub struct PinnedMemoryPool {
    handle: SlabMangerHandle,
    max_log_factor: u32,
    base_size: usize,
}

unsafe impl Send for PinnedMemoryPool {}

impl PinnedMemoryPool {
    pub fn new(max_log_factor: u32, base_size: usize) -> Self {
        Self {
            handle: unsafe { create_slab_manager(max_log_factor, base_size as u64) },
            max_log_factor,
            base_size,
        }
    }

    pub fn allocate<T: Sized>(&self, len: usize) -> *mut T {
        let size = std::mem::size_of::<T>() * len;
        let log_factor = (size.div_ceil(self.base_size))
            .next_power_of_two()
            .trailing_zeros();
        assert!(log_factor <= self.max_log_factor);
        unsafe { allocate(self.handle, log_factor) as *mut T }
    }

    pub fn preallocate(&self, num: usize) {
        // Preallocate memory for the pool, num: number of chunks
        let alloc_ptrs = (0..num)
            .into_iter()
            .map(|_| unsafe { allocate(self.handle, self.max_log_factor) }).collect::<Vec<_>>();
        alloc_ptrs.into_iter().for_each(|ptr| unsafe { deallocate(self.handle, ptr) });
    }

    pub fn free<T: Sized>(&self, ptr: *mut T) {
        unsafe { deallocate(self.handle, ptr as *mut std::ffi::c_void) }
    }

    pub fn clear(&self) {
        unsafe { clear(self.handle) }
    }

    pub fn shrink(&self) {
        unsafe { shrink(self.handle) }
    }
}

impl Drop for PinnedMemoryPool {
    fn drop(&mut self) {
        unsafe { destroy_slab_manager(self.handle) }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::seq::SliceRandom;
    use rand::Rng;

    #[test]
    fn test_memory_pool() {
        let range = 16;
        let iters = 100;
        let items = 1000;
        let pool = PinnedMemoryPool::new(range, std::mem::size_of::<u32>());

        for _ in 0..iters {
            let mut rng = rand::thread_rng();
            let mut slices: Vec<&mut [u32]> = (0..items)
                .map(|_| unsafe {
                    let len: usize = rng.gen_range(0..=(1 << range));
                    let ptr = pool.allocate::<u32>(len);
                    std::slice::from_raw_parts_mut(ptr, len)
                })
                .collect();
            for slice in slices.iter_mut() {
                for (x, id) in slice.iter_mut().zip(0..) {
                    *x = id
                }
            }
            slices.shuffle(&mut rng);
            for slice in slices.iter() {
                for (x, id) in slice.iter().zip(0..) {
                    assert_eq!(*x, id)
                }
            }
            for slice in slices.iter_mut() {
                pool.free(slice.as_mut_ptr());
            }
        }
    }

    #[test]
    fn test_preallocate() {
        let pool = PinnedMemoryPool::new(16, std::mem::size_of::<u32>());
        pool.preallocate(20);
    }
}
