use group::ff::Field;
use zkpoly_runtime::scalar::ScalarArray;

#[repr(C)]
pub struct PolyPtr {
    pub ptr: *mut u32,
    pub len: usize,
    pub rotate: i64,
    pub offset: usize,
    pub whole_len: usize,
}

#[repr(C)]
pub struct  ConstPolyPtr {
    pub ptr: *const u32,
    pub len: usize,
    pub rotate: i64,
    pub offset: usize,
    pub whole_len: usize,
}

impl <F: Field> From<ScalarArray<F>> for PolyPtr {
    fn from(poly: ScalarArray<F>) -> Self {
        let len = poly.len;
        let ptr = poly.values;
        let rotate = poly.get_rotation();
        let (offset, whole_len) = if poly.slice_info.is_none() {
            (0, len)
        } else {
            let slice_info = poly.slice_info.unwrap();
            (slice_info.offset, slice_info.whole_len)
        };
        PolyPtr {
            ptr: ptr as *mut u32,
            len,
            rotate: rotate as i64,
            offset,
            whole_len,
        }
    }
}

impl <F: Field> From<ScalarArray<F>> for ConstPolyPtr {
    fn from(poly: ScalarArray<F>) -> Self {
        let len = poly.len;
        let ptr = poly.values;
        let rotate = poly.get_rotation();
        let (offset, whole_len) = if poly.slice_info.is_none() {
            (0, len)
        } else {
            let slice_info = poly.slice_info.unwrap();
            (slice_info.offset, slice_info.whole_len)
        };
        ConstPolyPtr {
            ptr: ptr as *const u32,
            len,
            rotate: rotate as i64,
            offset,
            whole_len,
        }
    }
}