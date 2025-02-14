// this file is used to show how to insert a new function into the runtime

use std::{
    any::type_name,
    ffi::c_ulong,
    marker::PhantomData,
    os::raw::{c_longlong, c_uint, c_ulonglong, c_void},
    ptr::{null, null_mut},
};

use libloading::Symbol;
use zkpoly_cuda_api::{
    bindings::{
        cudaError_cudaSuccess, cudaError_t, cudaGetErrorString, cudaSetDevice, cudaStream_t,
    },
    cuda_check,
};

use zkpoly_runtime::{
    args::{RuntimeType, Variable},
    error::RuntimeError,
};

use super::build_func::{resolve_type, xmake_config, xmake_run};

use zkpoly_common::load_dynamic::Libs;

use zkpoly_runtime::functions::{Function, FunctionValue, RegisteredFunction};

pub struct PolyAdd<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            res: *mut c_uint,
            r_rotate: c_longlong,
            a: *const c_uint,
            a_rotate: c_longlong,
            b: *const c_uint,
            b_rotate: c_longlong,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolySub<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            res: *mut c_uint,
            r_rotate: c_longlong,
            a: *const c_uint,
            a_rotate: c_longlong,
            b: *const c_uint,
            b_rotate: c_longlong,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolyMul<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            res: *mut c_uint,
            r_rotate: c_longlong,
            a: *const c_uint,
            a_rotate: c_longlong,
            b: *const c_uint,
            b_rotate: c_longlong,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolyZero<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            target: *mut c_uint,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolyOne<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            target: *mut c_uint,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolyEval<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            temp_buf: *mut c_void,
            temp_buf_size: *mut c_ulong,
            poly: *const c_uint,
            res: *mut c_uint,
            x: *const c_uint,
            len: c_ulonglong,
            rotate: c_longlong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct KateDivision<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            temp_buf: *mut c_void,
            temp_buf_size: *mut c_ulong,
            log_p: c_uint,
            p: *const c_uint,
            p_rotate: c_longlong,
            b: *const c_uint,
            q: *mut c_uint,
            q_rotate: c_longlong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolyScan<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            temp_buffer: *mut c_void,
            buffer_size: *mut c_ulong,
            poly: *const c_uint,
            p_rotate: c_longlong,
            target: *mut c_uint,
            t_rotate: c_longlong,
            x0: *const c_uint,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolyInvert<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            temp_buffer: *mut c_void,
            buffer_size: *mut c_ulong,
            poly: *mut c_uint,
            inv: *mut c_uint,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

macro_rules! impl_poly_new {
    ($struct_name:ident, $symbol_name:literal) => {
        impl<T: RuntimeType> $struct_name<T> {
            pub fn new(libs: &mut Libs) -> Self {
                let field_type = resolve_type(type_name::<T::Field>());
                xmake_config("POLY_FIELD", field_type);
                xmake_run("poly");
                let lib = libs.load("../lib/libpoly.so");
                let c_func = unsafe { lib.get(concat!($symbol_name, "\0").as_bytes()) }.unwrap();
                Self {
                    _marker: PhantomData,
                    c_func,
                }
            }
        }
    };
}

impl_poly_new!(PolyInvert, "batched_invert");
impl_poly_new!(PolyScan, "scan_mul");
impl_poly_new!(PolyOne, "poly_one");
impl_poly_new!(PolyZero, "poly_zero");
impl_poly_new!(KateDivision, "kate_division");
impl_poly_new!(PolyEval, "poly_eval");
impl_poly_new!(PolyAdd, "poly_add");
impl_poly_new!(PolySub, "poly_sub");
impl_poly_new!(PolyMul, "poly_mul");

impl<T: RuntimeType> PolyInvert<T> {
    pub fn get_buffer_size(&self, len: usize) -> usize {
        let mut buf_size: usize = 0;
        unsafe {
            cuda_check!((self.c_func)(
                std::ptr::null_mut(),
                &mut buf_size as *mut usize as *mut c_ulong,
                null_mut(),
                null_mut(),
                len.try_into().unwrap(),
                null_mut(),
            ));
        }
        buf_size
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyInvert<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 3);
            assert_eq!(var.len(), 1);
            let [temp_buf_var, target, inv] = &mut mut_var[..] else {
                panic!("Expected 3 elements in mut_var");
            };
            let temp_buf = temp_buf_var.unwrap_gpu_buffer_mut();
            let target = target.unwrap_scalar_array_mut();
            let inv = inv.unwrap_scalar_mut();
            let stream = var[0].unwrap_stream();

            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut(),
                    target.values as *mut c_uint,
                    inv.value as *mut c_uint,
                    target.len.try_into().unwrap(),
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            name: "batched_invert".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> PolyScan<T> {
    pub fn get_buffer_size(&self, len: usize) -> usize {
        let mut buf_size: usize = 0;
        unsafe {
            cuda_check!((self.c_func)(
                std::ptr::null_mut(),
                &mut buf_size as *mut usize as *mut c_ulong,
                null(),
                0,
                null_mut(),
                0,
                null(),
                len.try_into().unwrap(),
                null_mut(),
            ));
        }
        buf_size
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyScan<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 2);
            assert_eq!(var.len(), 3);
            let (temp_buf_var, target) = mut_var.split_at_mut(1);
            let temp_buf = temp_buf_var[0].unwrap_gpu_buffer_mut();
            let target = target[0].unwrap_scalar_array_mut();

            let p = var[0].unwrap_scalar_array();
            assert_eq!(target.len, p.len);
            let x0 = var[1].unwrap_scalar();
            let stream = var[2].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut(),
                    p.values as *const u32,
                    p.get_rotation() as i64,
                    target.values as *mut u32,
                    target.get_rotation() as i64,
                    x0.value as *const u32,
                    p.len.try_into().unwrap(),
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            name: "PolyScan".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyOne<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            assert_eq!(var.len(), 1);
            let target = mut_var[0].unwrap_scalar_array_mut();
            let stream = var[0].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    target.values as *mut c_uint,
                    target.len.try_into().unwrap(),
                    stream.raw()
                ));
            }
            Ok(())
        };

        Function {
            name: "poly_one".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyZero<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            assert_eq!(var.len(), 1);
            let target = mut_var[0].unwrap_scalar_array_mut();
            let stream = var[0].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    target.values as *mut c_uint,
                    target.len.try_into().unwrap(),
                    stream.raw()
                ));
            }
            Ok(())
        };

        Function {
            name: "poly_zero".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> KateDivision<T> {
    pub fn get_buffer_size(&self, log_p: u32) -> usize {
        let mut buf_size: usize = 0;
        unsafe {
            cuda_check!((self.c_func)(
                std::ptr::null_mut(),
                &mut buf_size as *mut usize as *mut c_ulong,
                log_p,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            ));
        }
        buf_size
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for KateDivision<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert!(mut_var.len() == 2);
            assert!(var.len() == 3);
            let (temp_buf_var, res_var) = mut_var.split_at_mut(1);
            let temp_buf = temp_buf_var[0].unwrap_gpu_buffer_mut();
            let res = res_var[0].unwrap_scalar_array_mut();

            let p = var[0].unwrap_scalar_array();
            assert_eq!(res.len, p.len);
            assert!(p.len.is_power_of_two());
            let log_p = p.len.trailing_zeros();

            let b = var[1].unwrap_scalar();
            let stream = var[2].unwrap_stream();

            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut() as *mut c_ulong,
                    log_p,
                    p.values as *const c_uint,
                    p.get_rotation() as i64,
                    b.value as *const c_uint,
                    res.values as *mut c_uint,
                    res.get_rotation() as i64,
                    stream.raw(),
                ));
            }
            Ok(())
        };

        Function {
            name: "kate_division".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> PolyEval<T> {
    pub fn get_buffer_size(&self, len: usize) -> usize {
        let mut buf_size: usize = 0;
        unsafe {
            cuda_check!((self.c_func)(
                std::ptr::null_mut(),
                &mut buf_size as *mut usize as *mut c_ulong,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                len.try_into().unwrap(),
                0,
                std::ptr::null_mut(),
            ));
        }
        buf_size
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyEval<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert!(mut_var.len() == 2);
            assert!(var.len() == 3);
            let (temp_buf_var, res_var) = mut_var.split_at_mut(1);
            let temp_buf = temp_buf_var[0].unwrap_gpu_buffer_mut();
            let res = res_var[0].unwrap_scalar_mut();
            let poly = var[0].unwrap_scalar_array();
            let x = var[1].unwrap_scalar();
            let len = poly.len;
            let stream = var[2].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut() as *mut c_ulong,
                    poly.values as *const c_uint,
                    res.value as *mut c_uint,
                    x.value as *const c_uint,
                    len.try_into().unwrap(),
                    poly.get_rotation() as i64,
                    stream.raw(),
                ))
            }

            Ok(())
        };
        Function {
            name: "poly_eval".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyAdd<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();

        // define the rust side wrapper function
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert!(mut_var.len() == 1);
            assert!(var.len() == 3);
            let res = mut_var[0].unwrap_scalar_array_mut();
            let a = var[0].unwrap_scalar_array();
            let b = var[1].unwrap_scalar_array();
            let len = a.len;
            let stream = var[2].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    res.values as *mut c_uint,
                    res.get_rotation() as i64,
                    a.values as *const c_uint,
                    a.get_rotation() as i64,
                    b.values as *const c_uint,
                    b.get_rotation() as i64,
                    len as c_ulonglong,
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            name: "poly_add".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolySub<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();

        // define the rust side wrapper function
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert!(mut_var.len() == 1);
            assert!(var.len() == 3);
            let res = mut_var[0].unwrap_scalar_array_mut();
            let a = var[0].unwrap_scalar_array();
            let b = var[1].unwrap_scalar_array();
            let len = a.len;
            let stream = var[2].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    res.values as *mut c_uint,
                    res.get_rotation() as i64,
                    a.values as *const c_uint,
                    a.get_rotation() as i64,
                    b.values as *const c_uint,
                    b.get_rotation() as i64,
                    len as c_ulonglong,
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            name: "poly_sub".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyMul<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();

        // define the rust side wrapper function
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert!(mut_var.len() == 1);
            assert!(var.len() == 3);
            let res = mut_var[0].unwrap_scalar_array_mut();
            let a = var[0].unwrap_scalar_array();
            let b = var[1].unwrap_scalar_array();
            let len = a.len;
            let stream = var[2].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    res.values as *mut c_uint,
                    res.get_rotation() as i64,
                    a.values as *const c_uint,
                    a.get_rotation() as i64,
                    b.values as *const c_uint,
                    b.get_rotation() as i64,
                    len as c_ulonglong,
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            name: "poly_mul".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}
