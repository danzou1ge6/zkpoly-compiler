use std::{
    ops::{Index, IndexMut},
    ptr::copy_nonoverlapping,
};

use group::ff::Field;
use zkpoly_cuda_api::{
    mem::{alloc_pinned, free_pinned},
    stream::CudaStream,
};

use crate::{devices::DeviceType, runtime::transfer::Transfer};

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct ScalarArray<F: Field> {
    // when this is a slice, the pointer is pointed to the base array's start,
    // you have to visit the slice with slice_offset or by index
    pub values: *mut F,
    pub len: usize,
    rotate: i64, // the i64 is just to support neg during add, when getting rotate, we can safely assume it is positive
    pub device: DeviceType,
    pub slice_info: Option<ScalarSlice>,
}

#[derive(Debug, Clone)]
pub struct ScalarSlice {
    pub offset: usize,
    pub whole_len: usize,
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
            slice_info: None,
        }
    }

    pub fn get_rotation(&self) -> usize {
        self.rotate as usize
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end);
        assert!(end <= self.len);
        assert!(self.slice_info.is_none(), "slice can't be sliced again");
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
        assert!(index < self.len);
        let (offset, mod_len) = if self.slice_info.is_none() {
            (0, self.len)
        } else {
            let slice_info = self.slice_info.as_ref().unwrap();
            (slice_info.offset, slice_info.whole_len)
        };
        let rotate = self.rotate as usize;
        let actual_pos = (index + mod_len + offset - rotate) % mod_len;
        unsafe { &*self.values.add(actual_pos) }
    }
}

impl<F: Field> IndexMut<usize> for ScalarArray<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len);
        let (offset, mod_len) = if self.slice_info.is_none() {
            (0, self.len)
        } else {
            let slice_info = self.slice_info.as_ref().unwrap();
            (slice_info.offset, slice_info.whole_len)
        };
        let rotate = self.rotate as usize;
        let actual_pos = (index + mod_len + offset - rotate) % mod_len;
        unsafe { &mut *self.values.add(actual_pos) }
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
                }
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
}
