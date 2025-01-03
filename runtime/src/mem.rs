use crate::devices::DeviceType;
use zkpoly_cuda_api::stream::CudaStream;
use zkpoly_memory_pool::PinnedMemoryPool;

pub trait Allocator<T: Sized> {
    fn allocate(&self, len: u64) -> *mut T;
    fn deallocate(&self, ptr: *mut T);
    fn device_type(&self) -> DeviceType;
}

impl<T: Sized> Allocator<T> for PinnedMemoryPool {
    fn allocate(&self, len: u64) -> *mut T {
        assert!(len.is_power_of_two());
        let log_len = len.ilog2();
        self.allocate(log_len)
    }

    fn deallocate(&self, ptr: *mut T) {
        self.deallocate(ptr)
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::CPU
    }
}

impl<T: Sized> Allocator<T> for CudaStream {
    fn allocate(&self, len: u64) -> *mut T {
        self.allocate(len)
    }

    fn deallocate(&self, ptr: *mut T) {
        self.deallocate(ptr)
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::GPU {
            device_id: self.get_gpu_id() as u32,
            stream: self.clone(),
        }
    }
}
