use zkpoly_cuda_api::stream::CudaStream;

pub enum DeviceType {
    CPU,
    GPU { device_id: u32 },
    Disk,
}

pub enum StreamType {
    GpuStream { stream_id: u32 },
    Thread { thread_id: u32 },
}

pub struct Stream {
    typ: StreamType,
    stream_id: u32,
}

pub enum EventType {
    GpuEvent,
    ThreadEvent,
}

pub struct Event {
    typ: EventType,
    event_id: u32,
}
