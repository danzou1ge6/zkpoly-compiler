use std::ptr::copy_nonoverlapping;

use group::ff::Field;
use zkpoly_cuda_api::{
    mem::{alloc_pinned, free_pinned},
    stream::CudaStream,
};

use crate::{devices::DeviceType, runtime::transfer::Transfer};

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

    pub fn as_ref(&self) -> &F {
        unsafe { &*self.value }
    }

    pub fn as_mut(&mut self) -> &mut F {
        unsafe { &mut *self.value }
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

#[derive(Debug)]
pub struct ScalarArray<F: Field> {
    pub values: *mut F,
    pub len: usize,
    pub rotate: i64,
    pub device: DeviceType,
}

unsafe impl<F: Field> Send for ScalarArray<F> {}
unsafe impl<F: Field> Sync for ScalarArray<F> {}

impl<F: Field> ScalarArray<F> {
    pub fn new(len: usize, ptr: *mut F, device: DeviceType) -> Self {
        Self {
            values: ptr,
            len,
            rotate: 0,
            device,
        }
    }

    pub fn as_ref(&self) -> &[F] {
        unsafe { std::slice::from_raw_parts(self.values, self.len) }
    }

    pub fn as_mut(&mut self) -> &mut [F] {
        unsafe { std::slice::from_raw_parts_mut(self.values, self.len) }
    }

    pub fn rotate(&mut self, shift: i64) {
        self.rotate = (self.rotate + shift) % self.len as i64;
    }
}

impl<F: Field> Transfer for ScalarArray<F> {
    fn cpu2cpu(&self, target: &mut Self) {
        assert!(self.len <= target.len);
        assert!(self.device == DeviceType::CPU);
        assert!(target.device == DeviceType::CPU);
        let shift = target.rotate - self.rotate;
        assert!(shift.abs() < self.len as i64);
        let len = self.len;
        let src = self.values;
        let dst = target.values;
        unsafe {
            if shift == 0 {
                copy_nonoverlapping(src, dst, len);
            } else if shift > 0 {
                let shift = shift as usize;
                copy_nonoverlapping(src.add(shift), dst, len - shift);
                copy_nonoverlapping(src, dst.add(len - shift), shift);
            } else {
                let shift = (-shift) as usize;
                copy_nonoverlapping(src.add(len - shift), dst, shift);
                copy_nonoverlapping(src, dst.add(shift), len - shift);
            }
        }
    }

    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.len <= target.len);
        assert!(self.device == DeviceType::CPU);
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        let shift = target.rotate - self.rotate;
        assert!(shift.abs() < self.len as i64);
        let len = self.len;
        let src = self.values;
        let dst = target.values;
        if shift == 0 {
            stream.memcpy_h2d(dst, src, len);
        } else if shift > 0 {
            let shift = shift as usize;
            stream.memcpy_h2d(dst, unsafe { src.add(shift) }, len - shift);
            stream.memcpy_h2d(unsafe { dst.add(len - shift) }, src, shift);
        } else {
            let shift = (-shift) as usize;
            stream.memcpy_h2d(dst, unsafe { src.add(len - shift) }, shift);
            stream.memcpy_h2d(unsafe { dst.add(shift) }, src, len - shift);
        }
    }

    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream) {
        assert!(self.len <= target.len);
        assert!(target.device == DeviceType::CPU);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        let shift = target.rotate - self.rotate;
        assert!(shift.abs() < self.len as i64);
        let len = self.len;
        let src = self.values;
        let dst = target.values;
        if shift == 0 {
            stream.memcpy_d2h(dst, src, len);
        } else if shift > 0 {
            let shift = shift as usize;
            stream.memcpy_d2h(dst, unsafe { src.add(shift) }, len - shift);
            stream.memcpy_d2h(unsafe { dst.add(len - shift) }, src, shift);
        } else {
            let shift = (-shift) as usize;
            stream.memcpy_d2h(dst, unsafe { src.add(len - shift) }, shift);
            stream.memcpy_d2h(unsafe { dst.add(shift) }, src, len - shift);
        }
    }

    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        // currently, we do not support copying between two different GPUs
        assert!(self.len <= target.len);
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
        let shift = target.rotate - self.rotate;
        assert!(shift.abs() < self.len as i64);
        let len = self.len;
        let src = self.values;
        let dst = target.values;
        if shift == 0 {
            stream.memcpy_d2d(dst, src, len);
        } else if shift > 0 {
            let shift = shift as usize;
            stream.memcpy_d2d(dst, unsafe { src.add(shift) }, len - shift);
            stream.memcpy_d2d(unsafe { dst.add(len - shift) }, src, shift);
        } else {
            let shift = (-shift) as usize;
            stream.memcpy_d2d(dst, unsafe { src.add(len - shift) }, shift);
            stream.memcpy_d2d(unsafe { dst.add(shift) }, src, len - shift);
        }
    }
}
