use std::ops::Index;

use crate::{devices::DeviceType, runtime::transfer::Transfer};
use halo2curves::CurveAffine;
use zkpoly_cuda_api::{
    mem::{alloc_pinned, free_pinned},
    stream::CudaStream,
};
use zkpoly_memory_pool::CpuMemoryPool;

#[derive(Clone)]
pub struct Point<P: CurveAffine> {
    pub value: *mut P,
}

impl<P: CurveAffine> std::fmt::Debug for Point<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}){:?}", self.value, unsafe { self.value.as_ref() })
    }
}

unsafe impl<P: CurveAffine> Send for Point<P> {}
unsafe impl<P: CurveAffine> Sync for Point<P> {}

impl<P: CurveAffine> Point<P> {
    pub fn new(value: P) -> Self {
        let ptr: *mut P = alloc_pinned(1);
        unsafe {
            ptr.write(value);
        }
        Self { value: ptr }
    }

    pub fn as_ref(&self) -> &P {
        unsafe { &*self.value }
    }

    pub fn as_mut(&mut self) -> &mut P {
        unsafe { &mut *self.value }
    }

    pub fn deallocate(&mut self) {
        free_pinned(self.value);
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

impl<P: CurveAffine> PartialEq for PointArray<P> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        if self.device != other.device {
            return false;
        }
        for i in 0..self.len {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}

impl<P: CurveAffine> PointArray<P> {
    pub fn new(len: usize, ptr: *mut P, device: DeviceType) -> Self {
        Self {
            values: ptr,
            len,
            device,
        }
    }

    pub fn alloc_cpu(len: usize, allocator: &mut CpuMemoryPool) -> Self {
        let ptr = allocator.allocate(len);
        Self {
            values: ptr,
            len,
            device: DeviceType::CPU,
        }
    }

    pub fn from_vec(vec: &[P], allocator: &mut CpuMemoryPool) -> Self {
        let r = Self::alloc_cpu(vec.len(), allocator);
        unsafe {
            std::ptr::copy_nonoverlapping(vec.as_ptr(), r.values, vec.len());
        }
        r
    }
}

impl<P: CurveAffine> Index<usize> for PointArray<P> {
    type Output = P;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len);
        unsafe { &*self.values.add(index) }
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

#[test]
fn test_compare_point_array() {
    use halo2curves::bn256::G1Affine as G1;
    use zkpoly_memory_pool::CpuMemoryPool;

    let mut pool = CpuMemoryPool::new(10, size_of::<G1>());
    let a = PointArray::from_vec(&[G1::generator(), G1::generator()], &mut pool);
    let b = PointArray::from_vec(&[G1::generator(), G1::generator()], &mut pool);
    assert_eq!(a, b);
}
