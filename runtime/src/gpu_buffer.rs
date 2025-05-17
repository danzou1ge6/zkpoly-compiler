use zkpoly_common::devices::DeviceType;

#[derive(Debug, Clone)]
pub struct GpuBuffer {
    pub ptr: *mut u8,
    pub size: usize,
    pub device: DeviceType,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    pub fn new(ptr: *mut u8, size: usize, device: DeviceType) -> Self {
        Self { ptr, size, device }
    }
}
