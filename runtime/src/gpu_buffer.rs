#[derive(Debug, Clone)]
pub struct GpuBuffer {
    pub ptr: *mut u8,
    pub size: usize,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    pub fn new(ptr: *mut u8, size: usize) -> Self {
        Self { ptr, size }
    }
}
