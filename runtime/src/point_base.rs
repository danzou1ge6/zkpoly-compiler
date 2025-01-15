use crate::{devices::DeviceType, transport::Transport};
use pasta_curves::arithmetic::CurveAffine;
use zkpoly_cuda_api::stream::CudaStream;

#[derive(Debug)]
pub struct PointBase<P: CurveAffine> {
    pub values: *mut P,
    pub log_n: u32,
    pub device: DeviceType,
}

unsafe impl<P: CurveAffine> Send for PointBase<P> {}
unsafe impl<P: CurveAffine> Sync for PointBase<P> {}

impl<P: CurveAffine> PointBase<P> {
    pub fn new(log_n: u32, ptr: *mut P, device: DeviceType) -> Self {
        Self {
            values: ptr,
            log_n,
            device,
        }
    }
}

impl<P: CurveAffine> Transport for PointBase<P> {
    fn cpu2cpu(&self, target: &mut Self) {
        assert!(self.log_n == target.log_n);
        assert!(self.device == DeviceType::CPU);
        assert!(target.device == DeviceType::CPU);

        unsafe {
            std::ptr::copy_nonoverlapping(self.values, target.values, 1 << self.log_n);
        }
    }

    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.log_n == target.log_n);
        assert!(self.device == DeviceType::CPU);
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );

        stream.memcpy_h2d(target.values, self.values, 1 << self.log_n);
    }

    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.log_n == target.log_n);
        assert!(target.device == DeviceType::CPU);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );

        stream.memcpy_d2h(target.values, self.values, 1 << self.log_n);
    }

    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        // currently, we do not support copying between two different GPUs
        assert!(self.log_n == target.log_n);
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
        stream.memcpy_d2d(target.values, self.values, 1 << self.log_n);
    }

    fn cpu2disk(&self, target: &mut Self) {
        todo!();
    }

    fn disk2cpu(&self, target: &mut Self) {
        todo!();
    }
}
