use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq, PartialOrd, Ord)]
pub enum DeviceType {
    CPU,
    GPU { device_id: i32 },
    Disk,
}

impl DeviceType {
    pub fn unwrap_gpu(&self) -> i32 {
        match self {
            DeviceType::GPU { device_id } => *device_id,
            _ => panic!("unwrap_gpu: not a GPU device"),
        }
    }

    pub fn is_gpu(&self) -> bool {
        match self {
            DeviceType::GPU { .. } => true,
            _ => false,
        }
    }

    pub fn is_cpu(&self) -> bool {
        match self {
            DeviceType::CPU => true,
            _ => false,
        }
    }

    pub fn is_disk(&self) -> bool {
        match self {
            DeviceType::Disk => true,
            _ => false,
        }
    }
}
