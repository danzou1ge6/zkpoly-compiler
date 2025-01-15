use crate::{devices::DeviceType, transport::Transport};
use group::ff::Field;
use zkpoly_cuda_api::stream::CudaStream;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyType {
    Coef,
    Lagrange,
    ExtendedLagrange,
}

#[derive(Debug)]
pub struct Polynomial<F: Field> {
    pub values: *mut F,
    pub typ: PolyType,
    pub log_n: u32,
    pub rotate: u64,
    pub device: DeviceType,
}

unsafe impl<F: Field> Send for Polynomial<F> {}
unsafe impl<F: Field> Sync for Polynomial<F> {}

impl<F: Field> Polynomial<F> {
    pub fn new(typ: PolyType, log_n: u32, ptr: *mut F, device: DeviceType) -> Self {
        Self {
            values: ptr,
            typ,
            log_n,
            rotate: 0,
            device,
        }
    }
}

impl<F: Field> Transport for Polynomial<F> {
    fn cpu2cpu(&self, target: &mut Self) {
        assert!(self.log_n <= target.log_n);
        assert!(self.device == DeviceType::CPU);
        assert!(target.device == DeviceType::CPU);
        target.rotate = self.rotate;
        unsafe {
            std::ptr::copy_nonoverlapping(self.values, target.values, 1 << self.log_n);
        }
    }

    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.log_n <= target.log_n);
        assert!(self.device == DeviceType::CPU);
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        target.rotate = self.rotate;
        stream.memcpy_h2d(target.values, self.values, 1 << self.log_n);
    }

    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.log_n <= target.log_n);
        assert!(target.device == DeviceType::CPU);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        target.rotate = self.rotate;
        stream.memcpy_d2h(target.values, self.values, 1 << self.log_n);
    }

    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        // currently, we do not support copying between two different GPUs
        assert!(self.log_n <= target.log_n);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        target.rotate = self.rotate;
        stream.memcpy_d2d(target.values, self.values, 1 << self.log_n);
    }

    fn cpu2disk(&self, target: &mut Self) {
        todo!();
    }

    fn disk2cpu(&self, target: &mut Self) {
        todo!();
    }
}
