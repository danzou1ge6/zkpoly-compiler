
pub enum DeviceType {
    CPU,
    GPU{device_id: u32, stream: u32}, // stream to be modified to type
    Disk,
}

// This is a trait that will be implemented by all the types that can be moved between different devices.
pub trait DeviceTransfer {
    fn tansfer_to(&self, from: DeviceType, to: DeviceType);
}