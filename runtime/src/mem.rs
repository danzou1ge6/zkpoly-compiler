use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_cuda_api::stream::CudaStream;

pub trait Storage {
    fn allocate<T: Sized>(&self, len: u64) -> *mut T;
    fn deallocate<T: Sized>(&self, ptr: *mut T);
}

impl Storage for PinnedMemoryPool {
    fn allocate<T: Sized>(&self, len: u64) -> *mut T {
        assert!(len.is_power_of_two());
        let log_len = len.ilog2();
        self.allocate(log_len)
    }

    fn deallocate<T: Sized>(&self, ptr: *mut T) {
        self.deallocate(ptr)
    }
}

impl Storage for CudaStream {
    fn allocate<T: Sized>(&self, len: u64) -> *mut T {
        self.allocate(len)
    }
    
    fn deallocate<T: Sized>(&self, ptr: *mut T) {
        self.deallocate(ptr)
    }
}