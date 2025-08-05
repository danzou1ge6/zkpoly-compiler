use core::panic;
use std::{
    ops::{Index, IndexMut},
    ptr::copy_nonoverlapping,
};

use group::ff::Field;
use rand_core::RngCore;
use zkpoly_common::devices::DeviceType;
use zkpoly_cuda_api::{
    mem::{alloc_pinned, free_pinned},
    stream::CudaStream,
};
use zkpoly_memory_pool::{
    buddy_disk_pool::{
        cpu_read_from_disk, cpu_write_to_disk, gpu_read_from_disk, gpu_write_to_disk, DiskAllocInfo,
    },
    BuddyDiskPool, CpuMemoryPool,
};

use crate::runtime::transfer::Transfer;

#[derive(Clone)]
pub struct Scalar<F: Field> {
    pub value: *mut F,
    pub device: DeviceType,
}

impl<F: Field> std::fmt::Debug for Scalar<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scalar")
            .field("value_ptr", &self.value)
            .field("device", &self.device)
            .field("value", unsafe { self.value.as_ref().unwrap() })
            .finish()
    }
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

    pub fn deallocate(&mut self) {
        if self.device == DeviceType::CPU {
            free_pinned(self.value);
        } else {
            panic!("deallocate is only supported for CPU");
        }
    }

    pub fn new_gpu(value: *mut F, device_id: i32) -> Self {
        Self {
            value,
            device: DeviceType::GPU { device_id },
        }
    }

    pub fn from_ff(value: &F) -> Self {
        let mut r = Self::new_cpu();
        *r.as_mut() = value.clone();
        r
    }

    pub fn to_ff(&self) -> F {
        unsafe {
            let mut r = F::ZERO;
            self.value.copy_to(&mut r, 1);
            r
        }
    }

    pub fn as_ref(&self) -> &F {
        assert!(self.device == DeviceType::CPU);
        unsafe { &*self.value }
    }

    pub fn as_mut(&mut self) -> &mut F {
        assert!(self.device == DeviceType::CPU);
        unsafe { &mut *self.value }
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

#[derive(Clone)]
pub struct ScalarArray<F: Field> {
    // when this is a slice, the pointer is pointed to the base array's start,
    // you have to visit the slice with slice_offset or by index
    pub values: *mut F,
    pub len: usize,
    pub(crate) rotate: i64, // the i64 is just to support neg during add, when getting rotate, we can safely assume it is positive
    pub device: DeviceType,
    pub slice_info: Option<ScalarSlice>,
    pub disk_pos: Vec<DiskAllocInfo>, // if the poly is stored on disk, this is the offset of each part in each file, (fd, offset)
}

impl<F: Field> std::fmt::Debug for ScalarArray<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Array<F: Field>(*mut F, usize);
        impl<F: Field> std::fmt::Debug for Array<F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                for i in 0..self.1 {
                    write!(f, "{:?},", unsafe { self.0.add(i).as_ref().unwrap() })?;
                }
                Ok(())
            }
        }
        f.debug_struct("ScalarArray")
            .field("values_ptr", &self.values)
            .field("len", &self.len)
            .field("rotate", &self.rotate)
            .field("device", &self.device)
            .field("slice_info", &self.slice_info)
            .field("values", &Array(self.values, self.len))
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct ScalarSlice {
    pub offset: usize,
    pub whole_len: usize,
}

unsafe impl<F: Field> Send for ScalarArray<F> {}
unsafe impl<F: Field> Sync for ScalarArray<F> {}

impl<F: Field> PartialEq for ScalarArray<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
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

impl<F: Field> ScalarArray<F> {
    pub fn new(len: usize, ptr: *mut F, device: DeviceType) -> Self {
        Self {
            values: ptr,
            len,
            rotate: 0,
            device,
            slice_info: None,
            disk_pos: Vec::new(),
        }
    }

    pub fn alloc_cpu(len: usize, allocator: &mut CpuMemoryPool) -> Self {
        let ptr = allocator.allocate(len);
        Self {
            values: ptr,
            len,
            rotate: 0,
            device: DeviceType::CPU,
            slice_info: None,
            disk_pos: Vec::new(),
        }
    }

    pub fn alloc_disk(len: usize, allocator: &mut Vec<BuddyDiskPool>) -> Self {
        // assert!(len % allocator.len() == 0);
        let part_len = len / allocator.len();
        let disk_pose = allocator
            .iter_mut()
            .map(|pool| {
                let pos = pool.allocate(part_len * size_of::<F>()).unwrap();
                DiskAllocInfo::new(pos, pool)
            })
            .collect::<Vec<_>>();
        Self {
            values: std::ptr::null_mut(),
            len,
            rotate: 0,
            device: DeviceType::Disk,
            slice_info: None,
            disk_pos: disk_pose,
        }
    }

    pub fn free_disk(&mut self, allocator: &mut Vec<BuddyDiskPool>) {
        allocator
            .iter_mut()
            .zip(self.disk_pos.iter())
            .for_each(|(disk_pool, dai)| {
                disk_pool
                    .deallocate(dai.offset)
                    .expect("deallocation failed");
            });
    }

    pub fn from_vec(v: &[F], allocator: &mut CpuMemoryPool) -> Self {
        let r = Self::alloc_cpu(v.len(), allocator);
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), r.values, v.len());
        }
        r
    }

    pub fn borrow_vec(v: &[F]) -> Self {
        Self {
            values: v.as_ptr() as *mut F,
            len: v.len(),
            rotate: 0,
            device: DeviceType::CPU,
            slice_info: None,
            disk_pos: Vec::new(),
        }
    }

    pub fn from_iter(
        v: impl Iterator<Item = F>,
        len: usize,
        allocator: &mut CpuMemoryPool,
    ) -> Self {
        let r = Self::alloc_cpu(len, allocator);
        for (i, x) in v.take(len).enumerate() {
            unsafe {
                *r.values.add(i) = x;
            }
        }
        r
    }

    pub fn blind(&mut self, start: usize, end: usize, rng: impl RngCore + Clone) {
        for i in start..end {
            self[i] = F::random(rng.clone());
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get_rotation(&self) -> usize {
        self.rotate as usize
    }

    pub fn set_slice_raw(&self, mut offset: usize, len: usize) -> Option<Self> {
        let whole_len = if self.slice_info.is_none() {
            self.len
        } else {
            self.slice_info.as_ref().unwrap().whole_len
        };
        offset = offset % whole_len;
        if len > whole_len || offset >= whole_len {
            eprintln!(
                "get invalid slice: offset {}, len {}, whole_len {}",
                offset, len, whole_len
            );
            return None;
        }

        let slice_info = if len == whole_len && offset == 0 {
            None
        } else {
            Some(ScalarSlice {
                offset,
                whole_len: if self.slice_info.is_none() {
                    self.len
                } else {
                    self.slice_info.as_ref().unwrap().whole_len
                },
            })
        };

        Some(Self {
            values: self.values.clone(),
            len,
            rotate: 0,
            device: self.device.clone(),
            slice_info,
            disk_pos: Vec::new(),
        })
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end);
        assert!(end <= self.len);
        if let Some(si) = &self.slice_info {
            panic!("this is already a slice {:?}, can't be sliced again", si)
        }
        let rotate = self.get_rotation();
        let actual_pos = (self.len + start - rotate) % self.len;
        Self {
            values: self.values,
            len: end - start,
            rotate: 0, // slice can't rotate
            device: self.device.clone(),
            slice_info: Some(ScalarSlice {
                offset: actual_pos,
                whole_len: self.len,
            }),
            disk_pos: self.disk_pos.clone(),
        }
    }

    pub fn as_ref(&self) -> &[F] {
        assert!(self.slice_info.is_none());
        assert!(self.rotate == 0);
        unsafe { std::slice::from_raw_parts(self.values, self.len) }
    }

    pub fn as_mut(&mut self) -> &mut [F] {
        assert!(self.slice_info.is_none());
        assert!(self.rotate == 0);
        unsafe { std::slice::from_raw_parts_mut(self.values, self.len) }
    }

    pub fn rotate(&mut self, shift: i64) {
        assert!(self.slice_info.is_none());
        self.rotate = (self.rotate + shift) % self.len as i64;
        if self.rotate < 0 {
            self.rotate += self.len as i64
        }
    }

    pub fn iter(&self) -> ScalarArrayIter<'_, F> {
        let mod_len = if self.slice_info.is_none() {
            self.len
        } else {
            self.slice_info.as_ref().unwrap().whole_len
        };
        ScalarArrayIter {
            array: self,
            pos: 0,
            mod_len,
        }
    }

    pub fn iter_mut(&mut self) -> ScalarArrayIterMut<'_, F> {
        let mod_len = if self.slice_info.is_none() {
            self.len
        } else {
            self.slice_info.as_ref().unwrap().whole_len
        };
        ScalarArrayIterMut {
            array: self,
            pos: 0,
            mod_len,
        }
    }

    // helper function for transfer check
    pub fn check_target_len(&self, target: &Self) {
        if self.len != target.len {
            if self.len >= target.len {
                panic!(
                    "source array length {} is larger than target array length {}",
                    self.len, target.len
                );
            }
            assert!(self.len < target.len);
            assert!(target.rotate == 0);
            assert!(target.slice_info.is_none());
        }
    }

    pub fn get_shift(&self, target: &Self) -> usize {
        let shift: i64 = (target.rotate + self.len as i64 - self.rotate) % self.len as i64;
        assert!(shift >= 0 && shift < self.len as i64); // normalized
        shift as usize
    }

    pub fn get_ptr(&self, index: usize) -> *mut F {
        assert!(index < self.len);
        let (offset, mod_len) = if self.slice_info.is_none() {
            (0, self.len)
        } else {
            let slice_info = self.slice_info.as_ref().unwrap();
            (slice_info.offset, slice_info.whole_len)
        };
        let rotate = self.rotate as usize;
        let actual_pos = (index + mod_len + offset - rotate) % mod_len;
        unsafe { self.values.add(actual_pos) }
    }
}

pub struct ScalarArrayIter<'a, F: Field> {
    array: &'a ScalarArray<F>,
    pos: usize,
    mod_len: usize,
}

pub struct ScalarArrayIterMut<'a, F: Field> {
    array: &'a mut ScalarArray<F>,
    pos: usize,
    mod_len: usize,
}

impl<'a, F: Field> Iterator for ScalarArrayIter<'a, F> {
    type Item = &'a F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.array.len {
            None
        } else {
            let rotate = self.array.rotate as usize;
            let offset = if self.array.slice_info.is_none() {
                0
            } else {
                self.array.slice_info.as_ref().unwrap().offset
            };
            let actual_pos = (self.pos + self.mod_len + offset - rotate) % self.mod_len;
            self.pos += 1;
            Some(unsafe { &*self.array.values.add(actual_pos) })
        }
    }
}

impl<'a, F: Field> Iterator for ScalarArrayIterMut<'a, F> {
    type Item = &'a mut F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.array.len {
            None
        } else {
            let rotate = self.array.rotate as usize;
            let offset = if self.array.slice_info.is_none() {
                0
            } else {
                self.array.slice_info.as_ref().unwrap().offset
            };
            let actual_pos = (self.pos + self.mod_len + offset - rotate) % self.mod_len;
            self.pos += 1;
            Some(unsafe { &mut *self.array.values.add(actual_pos) })
        }
    }
}

impl<F: Field> Index<usize> for ScalarArray<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.get_ptr(index) }
    }
}

impl<F: Field> IndexMut<usize> for ScalarArray<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.get_ptr(index) }
    }
}

impl<F: Field> Transfer for ScalarArray<F> {
    fn cpu2cpu(&self, target: &mut Self) {
        self.check_target_len(target);
        assert!(self.device == DeviceType::CPU);
        assert!(target.device == DeviceType::CPU);

        let shift = self.get_shift(target);
        let len = self.len;
        let src = self.values;
        let dst = target.values;

        if self.slice_info.is_none() && target.slice_info.is_none() {
            unsafe {
                copy_nonoverlapping(src.add(shift), dst, len - shift);
                copy_nonoverlapping(src, dst.add(len - shift), shift);
            }
        } else if self.slice_info.is_some() && target.slice_info.is_none() {
            let slice_info = self.slice_info.as_ref().unwrap();
            let (offset, whole_len) = (slice_info.offset, slice_info.whole_len);
            if offset + self.len <= whole_len {
                // case1: slice is continuous
                unsafe {
                    copy_nonoverlapping(src.add(shift + offset), dst, len - shift);
                    copy_nonoverlapping(src.add(offset), dst.add(len - shift), shift);
                }
            } else {
                // case2: slice is not continuous
                if shift + offset > whole_len {
                    // subcase1: start array_end (shift+start) end
                    let seg1 = whole_len - offset;
                    let seg2 = shift - seg1;
                    unsafe {
                        copy_nonoverlapping(src.add(offset), dst.add(len - shift), seg1);
                        copy_nonoverlapping(src, dst.add(len - shift + seg1), seg2);
                        copy_nonoverlapping(src.add(seg2), dst, len - shift);
                    }
                } else {
                    // subcase2: start (start+shift) array_end end
                    let seg1 = whole_len - (offset + shift);
                    let seg2 = len - shift - seg1;
                    unsafe {
                        copy_nonoverlapping(src.add(offset), dst.add(len - shift), shift);
                        copy_nonoverlapping(src.add(offset + shift), dst, seg1);
                        copy_nonoverlapping(src, dst.add(seg1), seg2);
                    }
                }
            }
        } else if self.slice_info.is_none() && target.slice_info.is_some() {
            let slice_info = target.slice_info.as_ref().unwrap();
            let (offset, whole_len) = (slice_info.offset, slice_info.whole_len);
            if offset + target.len <= whole_len {
                // case1: slice is continuous
                unsafe {
                    copy_nonoverlapping(src.add(shift), dst.add(offset), len - shift);
                    copy_nonoverlapping(src, dst.add(offset + len - shift), shift);
                }
            } else {
                // case2: slice is not continuous
                if offset + len - shift > whole_len {
                    // subcase1: start array_end (len-shift+offset) end
                    let seg1 = whole_len - offset;
                    let seg2 = len - shift - seg1;
                    unsafe {
                        copy_nonoverlapping(src.add(shift), dst.add(offset), seg1);
                        copy_nonoverlapping(src.add(shift + seg1), dst, seg2);
                        copy_nonoverlapping(src, dst.add(seg2), shift);
                    }
                } else {
                    // subcase2: start (len - shift + offset) array_end end
                    let seg1 = whole_len - (offset + len - shift);
                    let seg2 = shift - seg1;
                    unsafe {
                        copy_nonoverlapping(src.add(shift), dst.add(offset), len - offset);
                        copy_nonoverlapping(src, dst.add(offset + seg1), seg1);
                        copy_nonoverlapping(src.add(seg1), dst, seg2);
                    }
                }
            }
        } else {
            assert!(shift == 0); // slice can't rotate
            assert_eq!(self.len, target.len); // slices has to have same length
            let src_slice_info = self.slice_info.as_ref().unwrap();
            let dst_slice_info = target.slice_info.as_ref().unwrap();
            let (src_offset, src_whole_len) = (src_slice_info.offset, src_slice_info.whole_len);
            let (dst_offset, dst_whole_len) = (dst_slice_info.offset, dst_slice_info.whole_len);
            if src_offset + self.len <= src_whole_len && dst_offset + target.len <= dst_whole_len {
                // case1: both slices are continuous
                unsafe {
                    copy_nonoverlapping(src.add(src_offset), dst.add(dst_offset), len);
                }
            } else if src_offset + self.len <= src_whole_len {
                // case2: self is continuous, target is not
                let seg1 = dst_whole_len - dst_offset;
                let seg2 = len - seg1;
                unsafe {
                    copy_nonoverlapping(src.add(src_offset), dst.add(dst_offset), seg1);
                    copy_nonoverlapping(src.add(src_offset + seg1), dst, seg2);
                }
            } else if dst_offset + len <= dst_whole_len {
                // case3: target is continuous, self is not
                let seg1 = src_whole_len - src_offset;
                let seg2 = len - seg1;
                unsafe {
                    copy_nonoverlapping(src.add(src_offset), dst.add(dst_offset), seg1);
                    copy_nonoverlapping(src, dst.add(dst_offset + seg1), seg2);
                }
            } else {
                // case4: neither is continuous
                let seg1 = src_whole_len - src_offset;
                let seg2 = dst_whole_len - dst_offset;
                let seg3 = len - seg1.max(seg2);
                if seg1 < seg2 {
                    unsafe {
                        copy_nonoverlapping(src.add(src_offset), dst.add(dst_offset), seg1);
                        copy_nonoverlapping(src, dst.add(dst_offset + seg1), seg2 - seg1);
                        copy_nonoverlapping(src.add(seg2 - seg1), dst, seg3);
                    }
                } else {
                    unsafe {
                        copy_nonoverlapping(src.add(src_offset), dst.add(dst_offset), seg2);
                        copy_nonoverlapping(src.add(src_offset + seg2), dst, seg1 - seg2);
                        copy_nonoverlapping(src, dst.add(seg1 - seg2), seg3);
                    }
                }
            }
        }
    }

    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        self.check_target_len(target);
        assert!(self.device == DeviceType::CPU);
        assert!(
            target.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                },
            "target device {:?} is not GPU with stream device {}",
            target.device,
            stream.get_device()
        );

        let shift = self.get_shift(target);
        let len = self.len;
        let src = self.values;
        let dst = target.values;

        if self.slice_info.is_none() && target.slice_info.is_none() {
            stream.memcpy_h2d(dst, unsafe { src.add(shift) }, len - shift);
            stream.memcpy_h2d(unsafe { dst.add(len - shift) }, src, shift);
        } else if self.slice_info.is_some() && target.slice_info.is_none() {
            let slice_info = self.slice_info.as_ref().unwrap();
            let (offset, whole_len) = (slice_info.offset, slice_info.whole_len);
            if offset + self.len <= whole_len {
                // case1: slice is continuous
                stream.memcpy_h2d(dst, unsafe { src.add(shift + offset) }, len - shift);
                stream.memcpy_h2d(
                    unsafe { dst.add(len - shift) },
                    unsafe { src.add(offset) },
                    shift,
                );
            } else {
                // case2: slice is not continuous
                if shift + offset > whole_len {
                    // subcase1: start array_end (shift+start) end
                    let seg1 = whole_len - offset;
                    let seg2 = shift - seg1;
                    stream.memcpy_h2d(dst, unsafe { src.add(offset) }, seg1);

                    unsafe {
                        stream.memcpy_h2d(dst.add(len - shift), src.add(offset), seg1);
                        stream.memcpy_h2d(dst.add(len - shift + seg1), src, seg2);
                        stream.memcpy_h2d(dst, src.add(seg2), len - shift);
                    }
                } else {
                    // subcase2: start (start+shift) array_end end
                    let seg1 = whole_len - (offset + shift);
                    let seg2 = len - shift - seg1;
                    unsafe {
                        stream.memcpy_h2d(dst.add(len - shift), src.add(offset), shift);
                        stream.memcpy_h2d(dst, src.add(offset + shift), seg1);
                        stream.memcpy_h2d(dst.add(seg1), src, seg2);
                    }
                }
            }
        } else if self.slice_info.is_none() && target.slice_info.is_some() {
            let slice_info = target.slice_info.as_ref().unwrap();
            let (offset, whole_len) = (slice_info.offset, slice_info.whole_len);
            if offset + target.len <= whole_len {
                // case1: slice is continuous
                unsafe {
                    stream.memcpy_h2d(dst.add(offset), src.add(shift), len - shift);
                    stream.memcpy_h2d(dst.add(offset + len - shift), src, shift);
                }
            } else {
                // case2: slice is not continuous
                if offset + len - shift > whole_len {
                    // subcase1: start array_end (len-shift+offset) end
                    let seg1 = whole_len - offset;
                    let seg2 = len - shift - seg1;
                    unsafe {
                        stream.memcpy_h2d(dst.add(offset), src.add(shift), seg1);
                        stream.memcpy_h2d(dst, src.add(shift + seg1), seg2);
                        stream.memcpy_h2d(dst.add(seg2), src, shift);
                    }
                } else {
                    // subcase2: start (len - shift + offset) array_end end
                    let seg1 = whole_len - (offset + len - shift);
                    let seg2 = shift - seg1;
                    unsafe {
                        stream.memcpy_h2d(dst.add(offset), src.add(shift), len - offset);
                        stream.memcpy_h2d(dst.add(offset + seg1), src, seg1);
                        stream.memcpy_h2d(dst, src.add(seg1), seg2);
                    }
                }
            }
        } else {
            assert!(shift == 0); // slice can't rotate
            assert_eq!(self.len, target.len); // slices has to have same length
            let src_slice_info = self.slice_info.as_ref().unwrap();
            let dst_slice_info = target.slice_info.as_ref().unwrap();
            let (src_offset, src_whole_len) = (src_slice_info.offset, src_slice_info.whole_len);
            let (dst_offset, dst_whole_len) = (dst_slice_info.offset, dst_slice_info.whole_len);
            if src_offset + self.len <= src_whole_len && dst_offset + target.len <= dst_whole_len {
                // case1: both slices are continuous
                unsafe {
                    stream.memcpy_h2d(dst.add(dst_offset), src.add(src_offset), len);
                }
            } else if src_offset + self.len <= src_whole_len {
                // case2: self is continuous, target is not
                let seg1 = dst_whole_len - dst_offset;
                let seg2 = len - seg1;
                unsafe {
                    stream.memcpy_h2d(dst.add(dst_offset), src.add(src_offset), seg1);
                    stream.memcpy_h2d(dst, src.add(src_offset + seg1), seg2);
                }
            } else if dst_offset + len <= dst_whole_len {
                // case3: target is continuous, self is not
                let seg1 = src_whole_len - src_offset;
                let seg2 = len - seg1;
                unsafe {
                    stream.memcpy_h2d(dst.add(dst_offset), src.add(src_offset), seg1);
                    stream.memcpy_h2d(dst.add(dst_offset + seg1), src, seg2);
                }
            } else {
                // case4: neither is continuous
                let seg1 = src_whole_len - src_offset;
                let seg2 = dst_whole_len - dst_offset;
                let seg3 = len - seg1.max(seg2);
                if seg1 < seg2 {
                    unsafe {
                        stream.memcpy_h2d(dst.add(dst_offset), src.add(src_offset), seg1);
                        stream.memcpy_h2d(dst.add(dst_offset + seg1), src, seg2 - seg1);
                        stream.memcpy_h2d(dst, src.add(seg2 - seg1), seg3);
                    }
                } else {
                    unsafe {
                        stream.memcpy_h2d(dst.add(dst_offset), src.add(src_offset), seg2);
                        stream.memcpy_h2d(dst, src.add(src_offset + seg2), seg1 - seg2);
                        stream.memcpy_h2d(dst.add(seg1 - seg2), src, seg3);
                    }
                }
            }
        }
    }

    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream) {
        self.check_target_len(target);
        assert!(target.device == DeviceType::CPU);
        assert!(
            self.device
                == DeviceType::GPU {
                    device_id: stream.get_device()
                }
        );
        let shift = self.get_shift(target);
        let len = self.len;
        let src = self.values;
        let dst = target.values;

        if self.slice_info.is_none() && target.slice_info.is_none() {
            stream.memcpy_d2h(dst, unsafe { src.add(shift) }, len - shift);
            stream.memcpy_d2h(unsafe { dst.add(len - shift) }, src, shift);
        } else if self.slice_info.is_some() && target.slice_info.is_none() {
            let slice_info = self.slice_info.as_ref().unwrap();
            let (offset, whole_len) = (slice_info.offset, slice_info.whole_len);
            if offset + self.len <= whole_len {
                // case1: slice is continuous
                stream.memcpy_d2h(dst, unsafe { src.add(shift + offset) }, len - shift);
                stream.memcpy_d2h(
                    unsafe { dst.add(len - shift) },
                    unsafe { src.add(offset) },
                    shift,
                );
            } else {
                // case2: slice is not continuous
                if shift + offset > whole_len {
                    // subcase1: start array_end (shift+start) end
                    let seg1 = whole_len - offset;
                    let seg2 = shift - seg1;
                    stream.memcpy_d2h(dst, unsafe { src.add(offset) }, seg1);

                    unsafe {
                        stream.memcpy_d2h(dst.add(len - shift), src.add(offset), seg1);
                        stream.memcpy_d2h(dst.add(len - shift + seg1), src, seg2);
                        stream.memcpy_d2h(dst, src.add(seg2), len - shift);
                    }
                } else {
                    // subcase2: start (start+shift) array_end end
                    let seg1 = whole_len - (offset + shift);
                    let seg2 = len - shift - seg1;
                    unsafe {
                        stream.memcpy_d2h(dst.add(len - shift), src.add(offset), shift);
                        stream.memcpy_d2h(dst, src.add(offset + shift), seg1);
                        stream.memcpy_d2h(dst.add(seg1), src, seg2);
                    }
                }
            }
        } else if self.slice_info.is_none() && target.slice_info.is_some() {
            let slice_info = target.slice_info.as_ref().unwrap();
            let (offset, whole_len) = (slice_info.offset, slice_info.whole_len);
            if offset + target.len <= whole_len {
                // case1: slice is continuous
                unsafe {
                    stream.memcpy_d2h(dst.add(offset), src.add(shift), len - shift);
                    stream.memcpy_d2h(dst.add(offset + len - shift), src, shift);
                }
            } else {
                // case2: slice is not continuous
                if offset + len - shift > whole_len {
                    // subcase1: start array_end (len-shift+offset) end
                    let seg1 = whole_len - offset;
                    let seg2 = len - shift - seg1;
                    unsafe {
                        stream.memcpy_d2h(dst.add(offset), src.add(shift), seg1);
                        stream.memcpy_d2h(dst, src.add(shift + seg1), seg2);
                        stream.memcpy_d2h(dst.add(seg2), src, shift);
                    }
                } else {
                    // subcase2: start (len - shift + offset) array_end end
                    let seg1 = whole_len - (offset + len - shift);
                    let seg2 = shift - seg1;
                    unsafe {
                        stream.memcpy_d2h(dst.add(offset), src.add(shift), len - offset);
                        stream.memcpy_d2h(dst.add(offset + seg1), src, seg1);
                        stream.memcpy_d2h(dst, src.add(seg1), seg2);
                    }
                }
            }
        } else {
            assert!(shift == 0); // slice can't rotate
            assert_eq!(self.len, target.len); // slices has to have same length
            let src_slice_info = self.slice_info.as_ref().unwrap();
            let dst_slice_info = target.slice_info.as_ref().unwrap();
            let (src_offset, src_whole_len) = (src_slice_info.offset, src_slice_info.whole_len);
            let (dst_offset, dst_whole_len) = (dst_slice_info.offset, dst_slice_info.whole_len);
            if src_offset + self.len <= src_whole_len && dst_offset + target.len <= dst_whole_len {
                // case1: both slices are continuous
                unsafe {
                    stream.memcpy_d2h(dst.add(dst_offset), src.add(src_offset), len);
                }
            } else if src_offset + self.len <= src_whole_len {
                // case2: self is continuous, target is not
                let seg1 = dst_whole_len - dst_offset;
                let seg2 = len - seg1;
                unsafe {
                    stream.memcpy_d2h(dst.add(dst_offset), src.add(src_offset), seg1);
                    stream.memcpy_d2h(dst, src.add(src_offset + seg1), seg2);
                }
            } else if dst_offset + len <= dst_whole_len {
                // case3: target is continuous, self is not
                let seg1 = src_whole_len - src_offset;
                let seg2 = len - seg1;
                unsafe {
                    stream.memcpy_d2h(dst.add(dst_offset), src.add(src_offset), seg1);
                    stream.memcpy_d2h(dst.add(dst_offset + seg1), src, seg2);
                }
            } else {
                // case4: neither is continuous
                let seg1 = src_whole_len - src_offset;
                let seg2 = dst_whole_len - dst_offset;
                let seg3 = len - seg1.max(seg2);
                if seg1 < seg2 {
                    unsafe {
                        stream.memcpy_d2h(dst.add(dst_offset), src.add(src_offset), seg1);
                        stream.memcpy_d2h(dst.add(dst_offset + seg1), src, seg2 - seg1);
                        stream.memcpy_d2h(dst, src.add(seg2 - seg1), seg3);
                    }
                } else {
                    unsafe {
                        stream.memcpy_d2h(dst.add(dst_offset), src.add(src_offset), seg2);
                        stream.memcpy_d2h(dst, src.add(src_offset + seg2), seg1 - seg2);
                        stream.memcpy_d2h(dst.add(seg1 - seg2), src, seg3);
                    }
                }
            }
        }
    }

    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream) {
        // currently, we do not support copying between two different GPUs
        self.check_target_len(target);
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
        let shift = self.get_shift(target);
        let len = self.len;
        let src = self.values;
        let dst = target.values;

        if self.slice_info.is_none() && target.slice_info.is_none() {
            stream.memcpy_d2d(dst, unsafe { src.add(shift) }, len - shift);
            stream.memcpy_d2d(unsafe { dst.add(len - shift) }, src, shift);
        } else if self.slice_info.is_some() && target.slice_info.is_none() {
            let slice_info = self.slice_info.as_ref().unwrap();
            let (offset, whole_len) = (slice_info.offset, slice_info.whole_len);
            if offset + self.len <= whole_len {
                // case1: slice is continuous
                stream.memcpy_d2d(dst, unsafe { src.add(shift + offset) }, len - shift);
                stream.memcpy_d2d(
                    unsafe { dst.add(len - shift) },
                    unsafe { src.add(offset) },
                    shift,
                );
            } else {
                // case2: slice is not continuous
                if shift + offset > whole_len {
                    // subcase1: start array_end (shift+start) end
                    let seg1 = whole_len - offset;
                    let seg2 = shift - seg1;
                    stream.memcpy_d2d(dst, unsafe { src.add(offset) }, seg1);

                    unsafe {
                        stream.memcpy_d2d(dst.add(len - shift), src.add(offset), seg1);
                        stream.memcpy_d2d(dst.add(len - shift + seg1), src, seg2);
                        stream.memcpy_d2d(dst, src.add(seg2), len - shift);
                    }
                } else {
                    // subcase2: start (start+shift) array_end end
                    let seg1 = whole_len - (offset + shift);
                    let seg2 = len - shift - seg1;
                    unsafe {
                        stream.memcpy_d2d(dst.add(len - shift), src.add(offset), shift);
                        stream.memcpy_d2d(dst, src.add(offset + shift), seg1);
                        stream.memcpy_d2d(dst.add(seg1), src, seg2);
                    }
                }
            }
        } else if self.slice_info.is_none() && target.slice_info.is_some() {
            let slice_info = target.slice_info.as_ref().unwrap();
            let (offset, whole_len) = (slice_info.offset, slice_info.whole_len);
            if offset + target.len <= whole_len {
                // case1: slice is continuous
                unsafe {
                    stream.memcpy_d2d(dst.add(offset), src.add(shift), len - shift);
                    stream.memcpy_d2d(dst.add(offset + len - shift), src, shift);
                }
            } else {
                // case2: slice is not continuous
                if offset + len - shift > whole_len {
                    // subcase1: start array_end (len-shift+offset) end
                    let seg1 = whole_len - offset;
                    let seg2 = len - shift - seg1;
                    unsafe {
                        stream.memcpy_d2d(dst.add(offset), src.add(shift), seg1);
                        stream.memcpy_d2d(dst, src.add(shift + seg1), seg2);
                        stream.memcpy_d2d(dst.add(seg2), src, shift);
                    }
                } else {
                    // subcase2: start (len - shift + offset) array_end end
                    let seg1 = whole_len - (offset + len - shift);
                    let seg2 = shift - seg1;
                    unsafe {
                        stream.memcpy_d2d(dst.add(offset), src.add(shift), len - offset);
                        stream.memcpy_d2d(dst.add(offset + seg1), src, seg1);
                        stream.memcpy_d2d(dst, src.add(seg1), seg2);
                    }
                }
            }
        } else {
            assert!(shift == 0); // slice can't rotate
            assert_eq!(self.len, target.len); // slices has to have same length
            let src_slice_info = self.slice_info.as_ref().unwrap();
            let dst_slice_info = target.slice_info.as_ref().unwrap();
            let (src_offset, src_whole_len) = (src_slice_info.offset, src_slice_info.whole_len);
            let (dst_offset, dst_whole_len) = (dst_slice_info.offset, dst_slice_info.whole_len);
            if src_offset + self.len <= src_whole_len && dst_offset + target.len <= dst_whole_len {
                // case1: both slices are continuous
                unsafe {
                    stream.memcpy_d2d(dst.add(dst_offset), src.add(src_offset), len);
                }
            } else if src_offset + self.len <= src_whole_len {
                // case2: self is continuous, target is not
                let seg1 = dst_whole_len - dst_offset;
                let seg2 = len - seg1;
                unsafe {
                    stream.memcpy_d2d(dst.add(dst_offset), src.add(src_offset), seg1);
                    stream.memcpy_d2d(dst, src.add(src_offset + seg1), seg2);
                }
            } else if dst_offset + len <= dst_whole_len {
                // case3: target is continuous, self is not
                let seg1 = src_whole_len - src_offset;
                let seg2 = len - seg1;
                unsafe {
                    stream.memcpy_d2d(dst.add(dst_offset), src.add(src_offset), seg1);
                    stream.memcpy_d2d(dst.add(dst_offset + seg1), src, seg2);
                }
            } else {
                // case4: neither is continuous
                let seg1 = src_whole_len - src_offset;
                let seg2 = dst_whole_len - dst_offset;
                let seg3 = len - seg1.max(seg2);
                if seg1 < seg2 {
                    unsafe {
                        stream.memcpy_d2d(dst.add(dst_offset), src.add(src_offset), seg1);
                        stream.memcpy_d2d(dst.add(dst_offset + seg1), src, seg2 - seg1);
                        stream.memcpy_d2d(dst, src.add(seg2 - seg1), seg3);
                    }
                } else {
                    unsafe {
                        stream.memcpy_d2d(dst.add(dst_offset), src.add(src_offset), seg2);
                        stream.memcpy_d2d(dst, src.add(src_offset + seg2), seg1 - seg2);
                        stream.memcpy_d2d(dst.add(seg1 - seg2), src, seg3);
                    }
                }
            }
        }
    }

    fn cpu2disk(&self, target: &mut Self) {
        assert_eq!(self.device, DeviceType::CPU);
        assert_eq!(target.device, DeviceType::Disk);
        assert_eq!(self.rotate, target.rotate);
        assert!(self.slice_info.is_none());
        assert!(target.slice_info.is_none());
        let alligned_size = (self.len * size_of::<F>()).next_multiple_of(4096);
        cpu_write_to_disk(self.values as *const u8, &target.disk_pos, alligned_size);
    }

    fn disk2cpu(&self, target: &mut Self) {
        assert_eq!(self.device, DeviceType::Disk);
        assert_eq!(target.device, DeviceType::CPU);
        assert_eq!(self.rotate, target.rotate);
        assert!(self.slice_info.is_none());
        assert!(target.slice_info.is_none());
        let alligned_size = (self.len * size_of::<F>()).next_multiple_of(4096);
        cpu_read_from_disk(target.values as *mut u8, &self.disk_pos, alligned_size);
    }

    fn gpu2disk(&self, target: &mut Self) {
        if let DeviceType::GPU { device_id } = self.device {
            assert_eq!(target.device, DeviceType::Disk);
            assert_eq!(self.rotate, target.rotate);
            assert!(self.slice_info.is_none());
            assert!(target.slice_info.is_none());
            let alligned_size = (self.len * size_of::<F>()).next_multiple_of(4096);
            gpu_write_to_disk(
                self.values as *const u8,
                &target.disk_pos,
                alligned_size,
                device_id,
            );
        } else {
            panic!("gpu2disk is only supported for GPU arrays");
        }
    }

    fn disk2gpu(&self, target: &mut Self) {
        if let DeviceType::GPU { device_id } = target.device {
            assert_eq!(self.device, DeviceType::Disk);
            assert_eq!(self.rotate, target.rotate);
            assert!(self.slice_info.is_none());
            assert!(target.slice_info.is_none());
            let alligned_size = (self.len * size_of::<F>()).next_multiple_of(4096);
            gpu_read_from_disk(
                target.values as *mut u8,
                &self.disk_pos,
                alligned_size,
                device_id,
            );
        } else {
            panic!("disk2gpu is only supported for GPU arrays");
        }
    }
}

#[test]
fn test_tranfer_cpu_disk() {
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(10, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(28), Some("/tmp".into())).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some("/data/tmp".into())).unwrap(),
    ];
    let mut array1 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array1.iter_mut().for_each(|v| {
        *v = F::random(rand_core::OsRng);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(1024, &mut disk_pool);
    array1.cpu2disk(&mut array2);
    let mut array3 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array2.disk2cpu(&mut array3);
    for i in 0..1024 {
        assert_eq!(array1[i], array3[i]);
    }
}

#[test]
fn test_tranfer_cpu_disk_large() {
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(26, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(34), Some("/tmp".into())).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some("/data/tmp".into())).unwrap(),
    ];
    let mut array1 = ScalarArray::<F>::alloc_cpu(2usize.pow(26), &mut cpu_pool);
    array1.iter_mut().enumerate().for_each(|(id, v)| {
        *v = F::from(id as u64);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(2usize.pow(26), &mut disk_pool);
    array1.cpu2disk(&mut array2);
    let mut array3 = ScalarArray::<F>::alloc_cpu(2usize.pow(26), &mut cpu_pool);
    array2.disk2cpu(&mut array3);
    for i in 0..1024 {
        assert_eq!(array1[i], array3[i]);
    }
}

#[test]
fn test_transfer_bandwidth_2gb() {
    use halo2curves::bn256::Fr as F;
    use std::mem::size_of;
    use std::time::Instant;
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    // 2GB = 2 * 1024 * 1024 * 1024 bytes
    let bytes_2gb = 2usize * 1024 * 1024 * 1024;
    let n_elem = bytes_2gb / size_of::<F>();

    // 分配CPU池和磁盘池
    let mut cpu_pool = CpuMemoryPool::new((bytes_2gb as f64).log2().ceil() as u32, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(bytes_2gb, Some("/tmp".into())).unwrap(),
        BuddyDiskPool::new(bytes_2gb, Some("/data/tmp".into())).unwrap(),
    ];

    // 分配并初始化数据
    let mut array1 = ScalarArray::<F>::alloc_cpu(n_elem, &mut cpu_pool);
    array1.iter_mut().enumerate().for_each(|(id, v)| {
        *v = F::from(id as u64);
    });

    // 写到磁盘并计时
    let mut array2 = ScalarArray::<F>::alloc_disk(n_elem, &mut disk_pool);
    let start_write = Instant::now();
    array1.cpu2disk(&mut array2);
    let elapsed_write = start_write.elapsed().as_secs_f64();
    let write_bw = (bytes_2gb as f64) / elapsed_write / (1024.0 * 1024.0 * 1024.0);

    // 从磁盘读回并计时
    let mut array3 = ScalarArray::<F>::alloc_cpu(n_elem, &mut cpu_pool);
    let start_read = Instant::now();
    array2.disk2cpu(&mut array3);
    let elapsed_read = start_read.elapsed().as_secs_f64();
    let read_bw = (bytes_2gb as f64) / elapsed_read / (1024.0 * 1024.0 * 1024.0);

    // 验证数据正确性（只比较前1024个元素）
    for i in 0..1024 {
        assert_eq!(array1[i], array3[i]);
    }

    println!(
        "Write bandwidth: {:.2} GB/s, Read bandwidth: {:.2} GB/s",
        write_bw, read_bw
    );
}

#[test]
fn test_transfer_gpu_disk() {
    use zkpoly_cuda_api::mem::page_allocator::PageAllocator;
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(10, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(28), Some("/tmp".into())).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some("/data/tmp".into())).unwrap(),
    ];
    let mut gpu_pool = PageAllocator::new(DeviceType::GPU { device_id: 0 }, 1024 * 1024 * 2, 2);
    let mut array1 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array1.iter_mut().for_each(|v| {
        *v = F::random(rand_core::OsRng);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(1024, &mut disk_pool);
    array1.cpu2disk(&mut array2);

    // Simulate GPU transfer
    let stream = CudaStream::new(0);
    let ptr_gpu: *mut F = gpu_pool.allocate(1024 * 1024 * 2, vec![0]);
    let mut array3 = ScalarArray::<F>::new(1024, ptr_gpu, DeviceType::GPU { device_id: 0 });
    array2.disk2gpu(&mut array3);

    let mut array4 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array3.gpu2cpu(&mut array4, &stream);

    stream.sync();

    for i in 0..1024 {
        if array1[i] != array4[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array4[i]
            );
        }
    }

    // Simulate GPU transfer back to Disk
    let mut array5 = ScalarArray::<F>::alloc_disk(1024, &mut disk_pool);
    array4.cpu2disk(&mut array5);
    let mut array6 = ScalarArray::<F>::alloc_cpu(1024, &mut cpu_pool);
    array5.disk2cpu(&mut array6);
    for i in 0..1024 {
        if array1[i] != array6[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array6[i]
            );
        }
    }
}

#[test]
fn test_transfer_gpu_disk_large() {
    use zkpoly_cuda_api::mem::page_allocator::PageAllocator;
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(28, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(34), Some("/tmp".into())).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some("/data/tmp".into())).unwrap(),
    ];
    let mut gpu_pool =
        PageAllocator::new(DeviceType::GPU { device_id: 0 }, 1024 * 1024 * 1024 * 8, 1);
    let mut array1 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array1.iter_mut().enumerate().for_each(|(id, v)| {
        *v = F::from(id as u64);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(2usize.pow(28), &mut disk_pool);
    array1.cpu2disk(&mut array2);

    // Simulate GPU transfer
    let stream = CudaStream::new(0);
    let ptr_gpu: *mut F = gpu_pool.allocate(1024 * 1024 * 1024 * 16, vec![0]);
    let mut array3 =
        ScalarArray::<F>::new(2usize.pow(28), ptr_gpu, DeviceType::GPU { device_id: 0 });
    array2.disk2gpu(&mut array3);

    let mut array4 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array3.gpu2cpu(&mut array4, &stream);

    stream.sync();

    for i in 0..2usize.pow(28) {
        if array1[i] != array4[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array4[i]
            );
        }
    }

    // Simulate GPU transfer back to Disk
    let mut array5 = ScalarArray::<F>::alloc_disk(2usize.pow(28), &mut disk_pool);
    array4.cpu2disk(&mut array5);
    let mut array6 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array5.disk2cpu(&mut array6);
    for i in 0..2usize.pow(28) {
        if array1[i] != array6[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array6[i]
            );
        }
    }
}

#[test]
fn test_transfer_gpu_disk_large_no_direct() {
    use zkpoly_cuda_api::mem::page_allocator::PageAllocator;
    use zkpoly_memory_pool::BuddyDiskPool;
    use zkpoly_memory_pool::CpuMemoryPool;

    use halo2curves::bn256::Fr as F;
    let mut cpu_pool = CpuMemoryPool::new(28, size_of::<F>());
    let mut disk_pool = vec![
        BuddyDiskPool::new(2usize.pow(34), Some("/tmp".into())).unwrap(),
        // BuddyDiskPool::new(2usize.pow(28), Some("/data/tmp".into())).unwrap(),
    ];
    let mut gpu_pool =
        PageAllocator::new(DeviceType::GPU { device_id: 0 }, 1024 * 1024 * 1024 * 8, 1);
    let mut array1 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array1.iter_mut().enumerate().for_each(|(id, v)| {
        *v = F::from(id as u64);
    });
    let mut array2 = ScalarArray::<F>::alloc_disk(2usize.pow(28), &mut disk_pool);
    array1.cpu2disk(&mut array2);

    // Simulate GPU transfer
    let stream = CudaStream::new(0);
    let ptr_gpu: *mut F = gpu_pool.allocate(1024 * 1024 * 1024 * 16, vec![0]);
    let mut array3 =
        ScalarArray::<F>::new(2usize.pow(28), ptr_gpu, DeviceType::GPU { device_id: 0 });

    let temp_size: usize = 1024 * 1024 * 2; // 2MB temporary buffer size
    let temp_buffer = alloc_pinned::<u8>(temp_size); // 2MB temporary buffer
    let temp_buffers = vec![temp_buffer];

    // array2.disk2gpu(&mut array3);
    zkpoly_memory_pool::buddy_disk_pool::gpu_read_from_disk_no_direct(
        array3.values.cast(),
        &array2.disk_pos,
        2usize.pow(28) * size_of::<F>(),
        0,
        &temp_buffers,
        temp_size,
    );
    // gpu_write_to_disk_no_direct(array3.values.cast(), &array2.disk_pos, 2usize.pow(28) * size_of::<F>(), 0, &temp_buffers, temp_size);

    let mut array4 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array3.gpu2cpu(&mut array4, &stream);

    stream.sync();

    for i in 0..2usize.pow(28) {
        if array1[i] != array4[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array4[i]
            );
        }
    }

    // Simulate GPU transfer back to Disk
    let mut array5 = ScalarArray::<F>::alloc_disk(2usize.pow(28), &mut disk_pool);

    // array3.gpu2disk(&mut array5);
    zkpoly_memory_pool::buddy_disk_pool::gpu_write_to_disk_no_direct(
        array3.values.cast(),
        &array5.disk_pos,
        2usize.pow(28) * size_of::<F>(),
        0,
        &temp_buffers,
        temp_size,
    );
    let mut array6 = ScalarArray::<F>::alloc_cpu(2usize.pow(28), &mut cpu_pool);
    array5.disk2cpu(&mut array6);
    for i in 0..2usize.pow(28) {
        if array1[i] != array6[i] {
            println!(
                "Mismatch at index {}: {:?} != {:?}",
                i, array1[i], array6[i]
            );
        }
    }
}
