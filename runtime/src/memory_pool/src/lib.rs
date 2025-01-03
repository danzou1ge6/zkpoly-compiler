include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub struct PinnedMemoryPool {
    handle: SlabMangerHandle,
    max_log_factor: u32,
    base_size: usize,
}

impl PinnedMemoryPool {
    pub fn new(max_log_factor: u32, base_size: usize) -> Self {
        Self {
            handle: unsafe { create_slab_manager(max_log_factor, base_size as u64) },
            max_log_factor,
            base_size,
        }
    }

    pub fn allocate<T: Sized>(&self, log_len: u32) -> *mut T {
        assert!(std::mem::size_of::<T>() % self.base_size == 0);
        let factor: u32 = (std::mem::size_of::<T>() / self.base_size).try_into().unwrap();
        assert!(factor.is_power_of_two());
        let log_factor = log_len + factor.ilog2();
        assert!(log_factor <= self.max_log_factor);
        unsafe { allocate(self.handle, log_factor) as *mut T }
    }

    pub fn deallocate<T: Sized>(&self, ptr: *mut T) {
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
    use rand::Rng;
    use rand::seq::SliceRandom;

    #[test]
    fn test_memory_pool() {
        let range = 16;
        let iters = 100;
        let items = 1000;
        let pool = PinnedMemoryPool::new(range, std::mem::size_of::<u32>());

        for _ in 0..iters {
            let mut rng = rand::thread_rng();
            let mut slices: Vec<&mut [u32]> = (0..items).map(|_| 
                unsafe{
                    let log_len = rng.gen_range(0..=range);
                    let ptr = pool.allocate::<u32>(log_len);
                    std::slice::from_raw_parts_mut(ptr, 1 << log_len)
                }).collect();
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
                pool.deallocate(slice.as_mut_ptr());
            }
        }
    }
}