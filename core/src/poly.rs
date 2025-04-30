// this file is used to show how to insert a new function into the runtime

use std::{
    any::type_name,
    ffi::{c_ulong, c_ulonglong},
    marker::PhantomData,
    os::raw::{c_uint, c_void},
    ptr::null_mut,
};

use group::ff::Field;

use libloading::Symbol;
use zkpoly_cuda_api::{
    bindings::{
        cudaError_cudaSuccess, cudaError_t, cudaGetErrorString, cudaSetDevice, cudaStream_t,
    },
    cuda_check,
};

use zkpoly_runtime::{
    args::{RuntimeType, Variable},
    devices::DeviceType,
    error::RuntimeError,
    functions::{FuncMeta, KernelType},
    runtime::transfer::Transfer,
};

use super::build_func::{resolve_type, xmake_config, xmake_run};

use zkpoly_common::load_dynamic::Libs;

use zkpoly_runtime::functions::{Function, FunctionValue, RegisteredFunction};

use crate::poly_ptr::{ConstPolyPtr, PolyPtr};

static LIB_NAME: &str = "libpoly.so";

pub struct PolyAdd<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            res: PolyPtr,
            a: ConstPolyPtr,
            b: ConstPolyPtr,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolySub<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            res: PolyPtr,
            a: ConstPolyPtr,
            b: ConstPolyPtr,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolyMul<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            res: PolyPtr,
            a: ConstPolyPtr,
            b: ConstPolyPtr,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolyZero<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func:
        Symbol<'static, unsafe extern "C" fn(target: PolyPtr, stream: cudaStream_t) -> cudaError_t>,
}

pub struct PolyOneLagrange<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func:
        Symbol<'static, unsafe extern "C" fn(target: PolyPtr, stream: cudaStream_t) -> cudaError_t>,
}

pub struct PolyOneCoef<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func:
        Symbol<'static, unsafe extern "C" fn(target: PolyPtr, stream: cudaStream_t) -> cudaError_t>,
}

pub struct PolyEval<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            temp_buf: *mut c_void,
            temp_buf_size: *mut c_ulong,
            poly: ConstPolyPtr,
            res: *mut c_uint,
            x: *const c_uint,
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
            p: ConstPolyPtr,
            b: *const c_uint,
            q: PolyPtr,
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
            target: PolyPtr,
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
            poly: PolyPtr,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct PolyPermute<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            temp_buffer: *mut c_void,
            buffer_size: *mut c_ulong,
            usable: c_ulong,
            input: ConstPolyPtr,
            table: ConstPolyPtr,
            res_input: PolyPtr,
            res_table: PolyPtr,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub struct ScalarInv<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(target: *mut c_uint, stream: cudaStream_t) -> cudaError_t,
    >,
}

pub struct ScalarPow<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            target: *mut c_uint,
            exp: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
    exp: u64,
}

macro_rules! impl_poly_new {
    ($struct_name:ident, $symbol_name:literal) => {
        impl<T: RuntimeType> $struct_name<T> {
            pub fn new(libs: &mut Libs) -> Self {
                if !libs.contains(LIB_NAME) {
                    let field_type = resolve_type(type_name::<T::Field>());
                    xmake_config("POLY_FIELD", field_type);
                    xmake_run("poly");
                }
                let lib = libs.load("libpoly.so");
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
impl_poly_new!(PolyOneLagrange, "poly_one_lagrange");
impl_poly_new!(PolyOneCoef, "poly_one_coef");
impl_poly_new!(PolyZero, "poly_zero");
impl_poly_new!(KateDivision, "kate_division");
impl_poly_new!(PolyEval, "poly_eval");
impl_poly_new!(PolyAdd, "poly_add");
impl_poly_new!(PolySub, "poly_sub");
impl_poly_new!(PolyMul, "poly_mul");
impl_poly_new!(ScalarInv, "inv_scalar");
impl_poly_new!(PolyPermute, "permute");

impl<T: RuntimeType> RegisteredFunction<T> for PolyPermute<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 3);
            assert_eq!(var.len(), 3);
            let [res_input_var, res_table_var, temp_buf_var] = &mut mut_var[..] else {
                panic!("Expected 3 elements in mut_var");
            };
            let res_input = res_input_var.unwrap_scalar_array_mut();
            let res_table = res_table_var.unwrap_scalar_array_mut();
            let temp_buf = temp_buf_var.unwrap_gpu_buffer_mut();

            let input = var[0].unwrap_scalar_array();
            let table = var[1].unwrap_scalar_array();
            let stream = var[2].unwrap_stream();

            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut(),
                    res_input.len() as u64,
                    ConstPolyPtr::from(input),
                    ConstPolyPtr::from(table),
                    PolyPtr::from(res_input),
                    PolyPtr::from(res_table),
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new("poly_permute".to_string(), KernelType::PolyPermute),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> PolyPermute<T> {
    pub fn get_buffer_size(&self, usable_len: usize) -> usize {
        let mut buf_size: usize = 0;
        unsafe {
            cuda_check!((self.c_func)(
                std::ptr::null_mut(),
                &mut buf_size as *mut usize as *mut c_ulong,
                usable_len as c_ulong,
                ConstPolyPtr::null(usable_len),
                ConstPolyPtr::null(usable_len),
                PolyPtr::null(usable_len),
                PolyPtr::null(usable_len),
                null_mut(),
            ));
        }
        buf_size
    }
}

impl<T: RuntimeType> ScalarPow<T> {
    pub fn new(libs: &mut Libs, exp: u64) -> Self {
        if !libs.contains(LIB_NAME) {
            let field_type = resolve_type(type_name::<T::Field>());
            xmake_config("POLY_FIELD", field_type);
            xmake_run("poly");
        }
        let lib = libs.load("libpoly.so");
        let c_func = unsafe { lib.get(concat!("scalar_pow", "\0").as_bytes()) }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
            exp,
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for ScalarPow<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let exp = self.exp.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            if var.len() == 0 {
                let scalar = mut_var[0].unwrap_scalar_mut();
                *scalar.as_mut() = scalar.as_ref().pow(vec![exp]);
            } else if var.len() == 1 {
                let scalar = mut_var[0].unwrap_scalar_mut();
                let stream = var[0].unwrap_stream();
                unsafe {
                    cuda_check!(cudaSetDevice(stream.get_device()));
                    cuda_check!(c_func(scalar.value as *mut c_uint, exp, stream.raw()));
                }
            } else {
                unreachable!("var len should be 1 or 2 for ScalarPow");
            }

            Ok(())
        };
        Function {
            meta: FuncMeta::new("scalar_pow".to_string(), KernelType::ScalarPow(exp)),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for ScalarInv<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            let scalar = mut_var[0].unwrap_scalar_mut();

            if var.len() == 1 {
                // gpu
                let stream = var[0].unwrap_stream();
                unsafe {
                    cuda_check!(cudaSetDevice(stream.get_device()));
                    cuda_check!(c_func(scalar.value as *mut c_uint, stream.raw(),));
                }
            } else if var.len() == 0 {
                *scalar.as_mut() = scalar.as_ref().invert().unwrap();
            } else {
                unreachable!("var len should be 1 or 0 for ScalarInv");
            }

            Ok(())
        };
        Function {
            meta: FuncMeta::new("scalar_inv".to_string(), KernelType::ScalarInvert),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> PolyInvert<T> {
    pub fn get_buffer_size(&self, len: usize) -> usize {
        let mut buf_size: usize = 0;
        unsafe {
            cuda_check!((self.c_func)(
                std::ptr::null_mut(),
                &mut buf_size as *mut usize as *mut c_ulong,
                PolyPtr::null(len),
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
            assert_eq!(mut_var.len(), 2);
            assert_eq!(var.len(), 1);
            let [temp_buf_var, target] = &mut mut_var[..] else {
                panic!("Expected 3 elements in mut_var");
            };
            let temp_buf = temp_buf_var.unwrap_gpu_buffer_mut();
            let target = target.unwrap_scalar_array_mut();
            let stream = var[0].unwrap_stream();

            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut(),
                    PolyPtr::from(target),
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new("batched_invert".to_string(), KernelType::BatchedInvert),
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
                PolyPtr::null(len),
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

            stream.memcpy_d2d(target.get_ptr(0), x0.value, 1);

            let mut target_slice = target.slice(1, target.len);
            let p_slice = p.slice(0, p.len - 1);
            p_slice.gpu2gpu(&mut target_slice, stream);

            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut(),
                    PolyPtr::from(target),
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new("scan_mul".to_string(), KernelType::ScanMul),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyOneLagrange<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            let target = mut_var[0].unwrap_scalar_array_mut();
            if target.device == DeviceType::CPU {
                assert_eq!(var.len(), 0);
                for iter in target.iter_mut() {
                    *iter = T::Field::ONE;
                }
            } else {
                assert_eq!(var.len(), 1);
                let stream = var[0].unwrap_stream();
                unsafe {
                    cuda_check!(cudaSetDevice(stream.get_device()));
                    cuda_check!(c_func(PolyPtr::from(target), stream.raw()));
                }
            }
            Ok(())
        };

        Function {
            meta: FuncMeta::new("poly_one_lagrange".to_string(), KernelType::NewOneLagrange),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyOneCoef<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert_eq!(mut_var.len(), 1);
            let target = mut_var[0].unwrap_scalar_array_mut();
            if target.device == DeviceType::CPU {
                assert_eq!(var.len(), 0);
                for i in 0..target.len {
                    if i == 0 {
                        target[i] = T::Field::ONE;
                    } else {
                        target[i] = T::Field::ZERO;
                    }
                }
            } else {
                assert_eq!(var.len(), 1);
                let stream = var[0].unwrap_stream();
                unsafe {
                    cuda_check!(cudaSetDevice(stream.get_device()));
                    cuda_check!(c_func(PolyPtr::from(target), stream.raw()));
                }
            }
            Ok(())
        };

        Function {
            meta: FuncMeta::new("poly_one_coef".to_string(), KernelType::NewOneCoef),
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
            let target = mut_var[0].unwrap_scalar_array_mut();
            if target.device == DeviceType::CPU {
                assert_eq!(var.len(), 0);
                for num in target.iter_mut() {
                    *num = T::Field::ZERO;
                }
            } else {
                assert_eq!(var.len(), 1);
                let stream = var[0].unwrap_stream();
                unsafe {
                    cuda_check!(cudaSetDevice(stream.get_device()));
                    cuda_check!(c_func(PolyPtr::from(target), stream.raw()));
                }
            }
            Ok(())
        };

        Function {
            meta: FuncMeta::new("poly_zero".to_string(), KernelType::NewZero),
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
                ConstPolyPtr::null(1 << log_p),
                std::ptr::null(),
                PolyPtr::null(1 << log_p),
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
                    ConstPolyPtr::from(p),
                    b.value as *const c_uint,
                    PolyPtr::from(res),
                    stream.raw(),
                ));
            }
            Ok(())
        };

        Function {
            meta: FuncMeta::new("kate_division".to_string(), KernelType::KateDivision),
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
                ConstPolyPtr::null(len),
                std::ptr::null_mut(),
                std::ptr::null(),
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
            let stream = var[2].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut() as *mut c_ulong,
                    ConstPolyPtr::from(poly),
                    res.value as *mut c_uint,
                    x.value as *const c_uint,
                    stream.raw(),
                ))
            }

            Ok(())
        };
        Function {
            meta: FuncMeta::new("poly_eval".to_string(), KernelType::EvaluatePoly),
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
            let stream = var[2].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    PolyPtr::from(res),
                    ConstPolyPtr::from(a),
                    ConstPolyPtr::from(b),
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new("poly_add".to_string(), KernelType::PolyAdd),
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
            let stream = var[2].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    PolyPtr::from(res),
                    ConstPolyPtr::from(a),
                    ConstPolyPtr::from(b),
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new("poly_sub".to_string(), KernelType::PolySub),
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
            let stream = var[2].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!(c_func(
                    PolyPtr::from(res),
                    ConstPolyPtr::from(a),
                    ConstPolyPtr::from(b),
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new(
                "poly_mul".to_string(),
                KernelType::Other, // not used in practice
            ),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}
