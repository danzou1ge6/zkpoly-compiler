use crate::mem::Allocator;
use std::rc::Rc;
use zkpoly_cuda_api::stream::CudaStream;

pub enum DeviceType {
    CPU,
    GPU { device_id: u32, stream: CudaStream }, // stream to be modified to type
    Disk,
}
