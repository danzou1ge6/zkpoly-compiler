include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub struct MemoryPool {
    handle: SlabMangerHandle,
    max_log_factor: u32,
    base_size: usize,
}

impl MemoryPool {
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

impl Drop for MemoryPool {
    fn drop(&mut self) {
        unsafe { destroy_slab_manager(self.handle) }
    }
}

#[test]
fn test_memory_pool() {
    use rand::Rng;
    use rand::seq::SliceRandom;

    let range = 16;
    let iters = 1000;
    let items = 1000;
    let pool = MemoryPool::new(range, std::mem::size_of::<u32>());

    for _ in 0..iters {
        let mut rng = rand::thread_rng();
        let mut ptrs: Vec<_> = (0..items).map(|_| pool.allocate::<u32>(rng.gen_range(1..=range))).collect();
        for ptr in ptrs.iter() {
            unsafe { **ptr = 42 }
        }
        ptrs.shuffle(&mut rng);
        for ptr in ptrs.iter() {
            assert_eq!(unsafe { **ptr }, 42);
        }
        for ptr in ptrs.iter() {
            pool.deallocate(*ptr);
        }
    }
}