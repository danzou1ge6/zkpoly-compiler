use std::ptr::null_mut;

use group::ff::Field;
use zkpoly_cuda_api::{
    mem::{alloc_pinned, free_pinned},
    stream::CudaStream,
};

use crate::{devices::DeviceType, transfer::Transfer};

#[derive(Debug)]
pub struct Scalar<F: Field> {
    pub value: *mut F,
    pub device: DeviceType,
}

unsafe impl<F: Field> Send for Scalar<F> {}
unsafe impl<F: Field> Sync for Scalar<F> {}

impl<F: Field> Scalar<F> {
    pub fn new_cpu() -> Self {
        Self {
            value: alloc_pinned(1),
            device: DeviceType::CPU,
        }
    }

    pub fn new_gpu(value: *mut F, device_id: i32) -> Self {
        Self {
            value,
            device: DeviceType::GPU { device_id },
        }
    }
}

impl<F: Field> Drop for Scalar<F> {
    fn drop(&mut self) {
        if self.device == DeviceType::CPU {
            free_pinned(self.value);
        }
    }
}

impl<F: Field> Transfer for Scalar<F> {
    fn cpu2cpu(&self, target: &mut Self) {
        assert!(self.device == DeviceType::CPU);
        assert!(target.device == DeviceType::CPU);

        unsafe {
            std::ptr::copy_nonoverlapping(self.value, target.value, 1);
        }
    }

    fn cpu2disk(&self, target: &mut Self) {
        unreachable!("scalar doesn't need to be transferred to disk");
    }

    fn disk2cpu(&self, target: &mut Self) {
        unreachable!("scalar doesn't need to be transferred from disk");
    }

    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.device == DeviceType::CPU);
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );

        stream.memcpy_h2d(target.value, self.value, 1);
    }

    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(target.device == DeviceType::CPU);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );

        stream.memcpy_d2h(target.value, self.value, 1);
    }

    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
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

        stream.memcpy_d2d(target.value, self.value, 1);
    }
}
