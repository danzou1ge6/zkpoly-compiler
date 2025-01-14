#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(unused)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
use crate::cuda_check;

#[derive(Clone)]
pub struct CudaEvent {
    event: cudaEvent_t,
}

impl CudaEvent {
    pub fn new() -> Self {
        let mut event: cudaEvent_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(cudaEventCreate(&mut event));
        }
        Self { event }
    }

    pub fn record(&self, stream: &CudaStream) {
        unsafe {
            cuda_check!(cudaEventRecord(self.event, stream.as_ptr()));
        }
    }

    pub fn sync(&self) {
        unsafe {
            cuda_check!(cudaEventSynchronize(self.event));
        }
    }

    pub fn as_ptr(&self) -> cudaEvent_t {
        self.event
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            cuda_check!(cudaEventDestroy(self.event));
        }
    }
}

#[derive(Clone)]
pub struct CudaStream {
    stream: cudaStream_t,
    gpu_id: i32,
}

impl CudaStream {
    pub fn new(device: i32) -> Self {
        let mut stream: cudaStream_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(cudaSetDevice(device));
            cuda_check!(cudaStreamCreate(&mut stream));
        }
        Self {
            stream,
            gpu_id: device,
        }
    }

    pub fn sync(&self) {
        unsafe {
            cuda_check!(cudaStreamSynchronize(self.stream));
        }
    }

    pub fn as_ptr(&self) -> cudaStream_t {
        self.stream
    }

    pub fn get_device(&self) -> i32 {
        self.gpu_id
    }

    pub fn allocate<T: Sized>(&self, len: u64) -> *mut T {
        let mut ptr: *mut T = std::ptr::null_mut();
        let size = std::mem::size_of::<T>() * len as usize;
        unsafe {
            cuda_check!(cudaMallocAsync(
                &mut ptr as *mut *mut T as *mut *mut std::ffi::c_void,
                size,
                self.stream
            ));
        }
        ptr
    }

    pub fn deallocate<T: Sized>(&self, ptr: *mut T) {
        unsafe {
            cuda_check!(cudaFreeAsync(ptr as *mut std::ffi::c_void, self.stream));
        }
    }

    pub fn record(&self, event: &CudaEvent) {
        unsafe {
            cuda_check!(cudaEventRecord(event.as_ptr(), self.stream));
        }
    }

    pub fn wait(&self, event: &CudaEvent) {
        unsafe {
            cuda_check!(cudaStreamWaitEvent(
                self.stream,
                event.as_ptr(),
                cudaEventWaitDefault
            ));
        }
    }

    pub fn memcpy_h2d<T: Sized>(&self, dst: *mut T, src: *const T, len: u64) {
        let size = std::mem::size_of::<T>() * len as usize;
        unsafe {
            cuda_check!(cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                cudaMemcpyKind_cudaMemcpyHostToDevice,
                self.stream,
            ));
        }
    }

    pub fn memcpy_d2h<T: Sized>(&self, dst: *mut T, src: *const T, len: u64) {
        let size = std::mem::size_of::<T>() * len as usize;
        unsafe {
            cuda_check!(cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                cudaMemcpyKind_cudaMemcpyDeviceToHost,
                self.stream,
            ));
        }
    }

    pub fn memcpy_d2d<T: Sized>(&self, dst: *mut T, src: *const T, len: u64) {
        let size = std::mem::size_of::<T>() * len as usize;
        unsafe {
            cuda_check!(cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                cudaMemcpyKind_cudaMemcpyDeviceToDevice,
                self.stream,
            ));
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            cuda_check!(cudaStreamDestroy(self.stream));
        }
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

#[test]
fn test_cuda_stream() {
    let stream = CudaStream::new(0);
    let event = CudaEvent::new();
    stream.record(&event);
    stream.wait(&event);
    stream.sync();
    let ptr = stream.allocate::<u32>(1024 * 1024 * 1024);
    stream.sync();
    stream.deallocate(ptr);
}
