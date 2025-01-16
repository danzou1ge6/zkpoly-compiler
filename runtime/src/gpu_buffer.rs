#[derive(Debug)]
pub struct GpuBuffer {
    pub ptr: *mut u8,
    pub size: usize,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}
