use crate::{devices::DeviceType, runtime::transfer::Transfer};
use pasta_curves::arithmetic::CurveAffine;
use zkpoly_cuda_api::stream::CudaStream;

#[derive(Debug, Clone)]
pub struct Point<P: CurveAffine> {
    pub value: P,
}

impl<P: CurveAffine> Point<P> {
    pub fn new(value: P) -> Self {
        Self { value }
    }

    pub fn as_ref(&self) -> &P {
        &self.value
    }

    pub fn as_mut(&mut self) -> &mut P {
        &mut self.value
    }
}

impl<P: CurveAffine> Transfer for Point<P> {
    fn cpu2cpu(&self, target: &mut Self) {
        target.value = self.value.clone();
    }
}

#[derive(Debug, Clone)]
pub struct PointArray<P: CurveAffine> {
    pub values: *mut P,
    pub len: usize,
    pub device: DeviceType,
}

unsafe impl<P: CurveAffine> Send for PointArray<P> {}
unsafe impl<P: CurveAffine> Sync for PointArray<P> {}

impl<P: CurveAffine> PointArray<P> {
    pub fn new(len: usize, ptr: *mut P, device: DeviceType) -> Self {
        Self {
            values: ptr,
            len,
            device,
        }
    }
}

impl<P: CurveAffine> Transfer for PointArray<P> {
    fn cpu2cpu(&self, target: &mut Self) {
        assert!(self.len == target.len);
        assert!(self.device == DeviceType::CPU);
        assert!(target.device == DeviceType::CPU);

        unsafe {
            std::ptr::copy_nonoverlapping(self.values, target.values, self.len);
        }
    }

    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.len == target.len);
        assert!(self.device == DeviceType::CPU);
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );

        stream.memcpy_h2d(target.values, self.values, self.len);
    }

    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.len == target.len);
        assert!(target.device == DeviceType::CPU);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );

        stream.memcpy_d2h(target.values, self.values, self.len);
    }

    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        // currently, we do not support copying between two different GPUs
        assert!(self.len == target.len);
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
        stream.memcpy_d2d(target.values, self.values, self.len);
    }
}
