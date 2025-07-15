use zkpoly_common::cpu_event::CpuEvent;

use crate::bindings::*;
use crate::cuda_check;

// the sematic of CudaEvent is slightly different from cudaEvent_t
// to enforce the recording of the event before using it
// we will use a cond var with a mutex to enforce the order
// note that this event can only be used once
pub struct CudaEvent {
    event: cudaEvent_t,
    event_ready: CpuEvent,
    device_id: i32,
}

impl CudaEvent {
    pub fn get_device(&self) -> i32 {
        self.device_id
    }

    pub fn reset(&mut self) {
        self.event_ready.reset();
    }

    pub fn new(device_id: i32) -> Self {
        let mut event: cudaEvent_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(cudaSetDevice(device_id));
            cuda_check!(cudaEventCreateWithFlags(&mut event, cudaEventBlockingSync));
        }
        Self {
            event,
            event_ready: CpuEvent::new(),
            device_id,
        }
    }

    pub fn wait_ready(&self) {
        self.event_ready.wait();
    }

    pub fn record(&self, stream: &CudaStream) {
        unsafe {
            cuda_check!(cudaSetDevice(self.device_id));
            cuda_check!(cudaEventRecord(self.event, stream.raw()));
        }
        self.event_ready.notify();
    }

    pub fn sync(&self) {
        self.wait_ready();
        unsafe {
            cuda_check!(cudaSetDevice(self.device_id));
            cuda_check!(cudaEventSynchronize(self.event));
        }
    }

    pub fn as_ptr(&self) -> cudaEvent_t {
        self.event
    }

    pub fn elapsed(&self, other: &CudaEvent) -> f32 {
        let mut elapsed: f32 = 0.0;
        unsafe {
            cuda_check!(cudaSetDevice(self.device_id));
            cuda_check!(cudaEventElapsedTime(&mut elapsed, self.event, other.event));
        }
        elapsed
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            cuda_check!(cudaSetDevice(self.device_id));
            cuda_check!(cudaEventDestroy(self.event));
        }
    }
}

// Assume the event is recorded on the current device, so we don't need to set the device again
pub struct CudaEventRaw {
    event: cudaEvent_t,
}

impl CudaEventRaw {
    pub fn new() -> Self {
        let mut event: cudaEvent_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(cudaEventCreateWithFlags(&mut event, cudaEventBlockingSync));
        }
        Self { event }
    }

    pub fn record(&self, stream: &CudaStream) {
        unsafe {
            cuda_check!(cudaEventRecord(self.event, stream.raw()));
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

    pub fn elapsed(&self, other: &CudaEventRaw) -> f32 {
        let mut elapsed: f32 = 0.0;
        unsafe {
            cuda_check!(cudaEventElapsedTime(&mut elapsed, self.event, other.event));
        }
        elapsed
    }
}

impl Drop for CudaEventRaw {
    fn drop(&mut self) {
        unsafe {
            cuda_check!(cudaEventDestroy(self.event));
        }
    }
}

#[derive(Debug, Clone)]
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

    pub fn raw(&self) -> cudaStream_t {
        self.stream
    }

    pub fn get_device(&self) -> i32 {
        self.gpu_id
    }

    pub fn allocate<T: Sized>(&self, len: usize) -> *mut T {
        let mut ptr: *mut T = std::ptr::null_mut();
        let size = std::mem::size_of::<T>() * len as usize;
        unsafe {
            cuda_check!(cudaMalloc(
                &mut ptr as *mut *mut T as *mut *mut std::ffi::c_void,
                size,
            ));
        }
        ptr
    }

    pub fn free<T: Sized>(&self, ptr: *mut T) {
        unsafe {
            cuda_check!(cudaFree(ptr as *mut std::ffi::c_void));
        }
    }

    pub fn record(&self, event: &CudaEvent) {
        event.record(self);
    }

    pub fn wait(&self, event: &CudaEvent) {
        event.wait_ready();
        unsafe {
            cuda_check!(cudaStreamWaitEvent(
                self.stream,
                event.as_ptr(),
                0,
            ));
        }
    }

    pub fn wait_raw(&self, event: &CudaEventRaw) {
        unsafe {
            cuda_check!(cudaStreamWaitEvent(
                self.stream,
                event.as_ptr(),
                0,
            ));
        }
    }

    pub fn memcpy_h2d<T: Sized>(&self, dst: *mut T, src: *const T, len: usize) {
        let size = std::mem::size_of::<T>() * len;
        unsafe {
            cuda_check!(cudaSetDevice(self.gpu_id));
            cuda_check!(cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                cudaMemcpyKind_cudaMemcpyHostToDevice,
                self.stream,
            ));
        }
    }

    pub fn memcpy_d2h<T: Sized>(&self, dst: *mut T, src: *const T, len: usize) {
        let size = std::mem::size_of::<T>() * len;
        unsafe {
            cuda_check!(cudaSetDevice(self.gpu_id));
            cuda_check!(cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                cudaMemcpyKind_cudaMemcpyDeviceToHost,
                self.stream,
            ));
        }
    }

    pub fn memcpy_d2d<T: Sized>(&self, dst: *mut T, src: *const T, len: usize) {
        let size = std::mem::size_of::<T>() * len;
        unsafe {
            cuda_check!(cudaSetDevice(self.gpu_id));
            cuda_check!(cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                cudaMemcpyKind_cudaMemcpyDeviceToDevice,
                self.stream,
            ));
        }
    }

    pub fn destroy(&self) {
        unsafe {
            cuda_check!(cudaStreamDestroy(self.stream));
        }
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

unsafe impl Send for CudaEventRaw {}
unsafe impl Sync for CudaEventRaw {}

#[test]
fn test_cuda_stream() {
    let stream = CudaStream::new(0);
    let event = CudaEvent::new(0);
    stream.record(&event);
    stream.wait(&event);
    stream.sync();
    let ptr = stream.allocate::<u32>(1024 * 1024 * 1024);
    stream.sync();
    stream.free(ptr);
}
