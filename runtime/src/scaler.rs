use std::slice::{from_raw_parts, from_raw_parts_mut};

use group::ff::Field;
use zkpoly_cuda_api::{
    mem::{alloc_pinned, free_pinned},
    stream::CudaStream,
};

use crate::{devices::DeviceType, runtime::transfer::Transfer};

#[derive(Debug)]
pub struct Scalar<F: Field> {
    pub value: *mut F,
    pub len: usize,
    pub device: DeviceType,
}

unsafe impl<F: Field> Send for Scalar<F> {}
unsafe impl<F: Field> Sync for Scalar<F> {}

impl<F: Field> Scalar<F> {
    pub fn new_cpu(len: usize) -> Self {
        Self {
            value: alloc_pinned(len),
            len,
            device: DeviceType::CPU,
        }
    }

    pub fn new_gpu(value: *mut F, device_id: i32, len: usize) -> Self {
        Self {
            value,
            len,
            device: DeviceType::GPU { device_id },
        }
    }

    pub fn as_ref(&self) -> &[F] {
        unsafe { from_raw_parts(self.value, self.len) }
    }

    pub fn as_mut(&mut self) -> &mut [F] {
        unsafe { from_raw_parts_mut(self.value, self.len) }
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
        assert!(self.len == target.len);

        unsafe {
            std::ptr::copy_nonoverlapping(self.value, target.value, self.len);
        }
    }

    fn cpu2disk(&self, _: &mut Self) {
        unreachable!("scalar doesn't need to be transferred to disk");
    }

    fn disk2cpu(&self, _: &mut Self) {
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
        assert_eq!(self.len, target.len);

        stream.memcpy_h2d(target.value, self.value, self.len);
    }

    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(target.device == DeviceType::CPU);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        assert_eq!(self.len, target.len);

        stream.memcpy_d2h(target.value, self.value, self.len);
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
        assert_eq!(self.len, target.len);

        stream.memcpy_d2d(target.value, self.value, self.len);
    }
}
