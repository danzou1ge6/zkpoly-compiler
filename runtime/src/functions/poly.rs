// this file is used to show how to insert a new function into the runtime

use std::{
    any::type_name,
    ffi::c_ulong,
    marker::PhantomData,
    os::raw::{c_uint, c_ulonglong, c_void},
    ptr::{null, null_mut},
};

use libloading::Symbol;
use zkpoly_cuda_api::{
    bindings::{cudaError_cudaSuccess, cudaError_t, cudaGetErrorString, cudaStream_t},
    cuda_check,
};

use crate::{
    args::{RuntimeType, Variable},
    error::RuntimeError,
};

use super::{
    build_func::{resolve_type, xmake_config, xmake_run},
    load_dynamic::Libs,
    Function, FunctionValue, RegisteredFunction,
};

pub struct PolyAdd<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            res: *mut c_uint,
            a: *const c_uint,
            b: *const c_uint,
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
            a: *const c_uint,
            b: *const c_uint,
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
            a: *const c_uint,
            b: *const c_uint,
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
            b: *const c_uint,
            q: *mut c_uint,
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
            target: *mut c_uint,
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

impl<T: RuntimeType> PolyInvert<T> {
    pub fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("POLY_FIELD", field_type);
        xmake_run("poly");

        // load the dynamic library
        let lib = libs.load("../lib/libpoly.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"batched_invert\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }

    pub fn get_buffer_size(&self, len: u64) -> usize {
        let mut buf_size: usize = 0;
        unsafe {
            cuda_check!((self.c_func)(
                std::ptr::null_mut(),
                &mut buf_size as *mut usize as *mut c_ulong,
                null_mut(),
                null_mut(),
                len,
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
    pub fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("POLY_FIELD", field_type);
        xmake_run("poly");

        // load the dynamic library
        let lib = libs.load("../lib/libpoly.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"scan_mul\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }

    pub fn get_buffer_size(&self, len: u64) -> usize {
        let mut buf_size: usize = 0;
        unsafe {
            cuda_check!((self.c_func)(
                std::ptr::null_mut(),
                &mut buf_size as *mut usize as *mut c_ulong,
                null(),
                null_mut(),
                null(),
                len,
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
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut(),
                    p.values as *const u32,
                    target.values as *mut u32,
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

impl<T: RuntimeType> PolyOne<T> {
    pub fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("POLY_FIELD", field_type);
        xmake_run("poly");

        // load the dynamic library
        let lib = libs.load("../lib/libpoly.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"poly_one\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
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

impl<T: RuntimeType> PolyZero<T> {
    pub fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("POLY_FIELD", field_type);
        xmake_run("poly");

        // load the dynamic library
        let lib = libs.load("../lib/libpoly.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"poly_zero\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
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
    pub fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("POLY_FIELD", field_type);
        xmake_run("poly");

        // load the dynamic library
        let lib = libs.load("../lib/libpoly.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"kate_division\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }

    pub fn get_buffer_size(&self, log_p: u32) -> usize {
        let mut buf_size: usize = 0;
        unsafe {
            cuda_check!((self.c_func)(
                std::ptr::null_mut(),
                &mut buf_size as *mut usize as *mut c_ulong,
                log_p,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
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
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut() as *mut c_ulong,
                    log_p,
                    p.values as *const c_uint,
                    b.value as *const c_uint,
                    res.values as *mut c_uint,
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
    pub fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("POLY_FIELD", field_type);
        xmake_run("poly");

        // load the dynamic library
        let lib = libs.load("../lib/libpoly.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"poly_eval\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }
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
                cuda_check!(c_func(
                    temp_buf.ptr as *mut c_void,
                    null_mut() as *mut c_ulong,
                    poly.values as *const c_uint,
                    res.value as *mut c_uint,
                    x.value as *const c_uint,
                    len.try_into().unwrap(),
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

impl<T: RuntimeType> PolyAdd<T> {
    fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("POLY_FIELD", field_type);
        xmake_run("poly");

        // load the dynamic library
        let lib = libs.load("../lib/libpoly.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"poly_add\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolyAdd<T> {
    fn get_fn(&self) -> super::Function<T> {
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
                cuda_check!(c_func(
                    res.values as *mut c_uint,
                    a.values as *const c_uint,
                    b.values as *const c_uint,
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

impl<T: RuntimeType> PolySub<T> {
    fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("POLY_FIELD", field_type);
        xmake_run("poly");

        // load the dynamic library
        let lib = libs.load("../lib/libpoly.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"poly_sub\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PolySub<T> {
    fn get_fn(&self) -> super::Function<T> {
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
                cuda_check!(c_func(
                    res.values as *mut c_uint,
                    a.values as *const c_uint,
                    b.values as *const c_uint,
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

impl<T: RuntimeType> PolyMul<T> {
    fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("POLY_FIELD", field_type);
        xmake_run("poly");

        // load the dynamic library
        let lib = libs.load("../lib/libpoly.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"poly_mul\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }
}
impl<T: RuntimeType> RegisteredFunction<T> for PolyMul<T> {
    fn get_fn(&self) -> super::Function<T> {
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
                cuda_check!(c_func(
                    res.values as *mut c_uint,
                    a.values as *const c_uint,
                    b.values as *const c_uint,
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

#[cfg(test)]
mod test {
    static K: u32 = 20;

    use crate::args::RuntimeType;
    use crate::args::Variable;
    use crate::devices::DeviceType;
    use crate::functions::load_dynamic::Libs;
    use crate::functions::RegisteredFunction;
    use crate::gpu_buffer::GpuBuffer;
    use crate::runtime::transfer::Transfer;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarArray;
    use crate::transcript::{Blake2bWrite, Challenge255};
    use group::ff::BatchInvert;
    use group::ff::Field;
    use halo2_proofs::arithmetic::kate_division;
    use halo2curves::bn256;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use zkpoly_cuda_api::stream::CudaStream;
    use zkpoly_memory_pool::PinnedMemoryPool;

    use super::*;

    #[derive(Debug, Clone)]
    struct MyRuntimeType;

    impl RuntimeType for MyRuntimeType {
        type Field = bn256::Fr;
        type PointAffine = bn256::G1Affine;
        type Challenge = Challenge255<bn256::G1Affine>;
        type Trans = Blake2bWrite<Vec<u8>, bn256::G1Affine, Challenge255<bn256::G1Affine>>;
    }

    type MyField = <MyRuntimeType as RuntimeType>::Field;

    fn test_binary(
        new: fn(&mut Libs) -> Box<dyn RegisteredFunction<MyRuntimeType>>,
        truth_func: fn(MyField, MyField) -> MyField,
    ) {
        let mut libs = Libs::new();
        let poly_add = new(&mut libs);
        let func = poly_add.get_fn();
        let f = match func {
            Function {
                f: FunctionValue::Fn(func),
                ..
            } => func,
            _ => panic!("expected Fn"),
        };
        let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
        let stream = Variable::Stream(CudaStream::new(0));
        let len = 1 << K;
        let mut a = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut b = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut res = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut rng = XorShiftRng::from_seed([0; 16]);
        for i in 0..len {
            a.as_mut()[i] = MyField::random(&mut rng);
            b.as_mut()[i] = MyField::random(&mut rng);
        }
        let mut a_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));
        let mut b_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));
        let mut res_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));
        a.cpu2gpu(a_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
        b.cpu2gpu(b_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
        f(vec![&mut res_d], vec![&a_d, &b_d, &stream]).unwrap();
        res_d
            .unwrap_scalar_array_mut()
            .gpu2cpu(&mut res, stream.unwrap_stream());
        stream
            .unwrap_stream()
            .free(a_d.unwrap_scalar_array().values);
        stream
            .unwrap_stream()
            .free(b_d.unwrap_scalar_array().values);
        stream
            .unwrap_stream()
            .free(res_d.unwrap_scalar_array().values);
        stream.unwrap_stream().sync();
        for ((a, b), r) in a
            .as_ref()
            .iter()
            .zip(b.as_ref().iter())
            .zip(res.as_ref().iter())
        {
            assert_eq!(*r, truth_func(*a, *b));
        }
    }

    #[test]
    fn test_add() {
        test_binary(
            |libs| -> Box<dyn RegisteredFunction<MyRuntimeType>> {
                Box::new(PolyAdd::<MyRuntimeType>::new(libs))
            },
            |a, b| a + b,
        );
    }

    #[test]
    fn test_sub() {
        test_binary(
            |libs| -> Box<dyn RegisteredFunction<MyRuntimeType>> {
                Box::new(PolySub::<MyRuntimeType>::new(libs))
            },
            |a, b| a - b,
        );
    }

    #[test]
    fn test_mul() {
        test_binary(
            |libs| -> Box<dyn RegisteredFunction<MyRuntimeType>> {
                Box::new(PolyMul::<MyRuntimeType>::new(libs))
            },
            |a, b| a * b,
        );
    }

    #[test]
    fn test_eval() {
        let len = 1 << K;
        let mut libs = Libs::new();
        let poly_eval = PolyEval::<MyRuntimeType>::new(&mut libs);
        let func = match poly_eval.get_fn() {
            Function {
                f: FunctionValue::Fn(func),
                ..
            } => func,
            _ => unreachable!(),
        };
        let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
        let stream = Variable::Stream(CudaStream::new(0));
        let mut poly = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut x = Scalar::new_cpu();
        let mut res = Scalar::new_cpu();
        let mut rng = XorShiftRng::from_seed([0; 16]);
        for i in 0..len {
            poly.as_mut()[i] = MyField::random(&mut rng);
        }
        *x.as_mut() = MyField::random(&mut rng);
        let mut poly_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));
        let mut x_d = Variable::Scalar(Scalar::new_gpu(stream.unwrap_stream().allocate(1), 0));
        let mut res_d = Variable::Scalar(Scalar::new_gpu(stream.unwrap_stream().allocate(1), 0));
        let buf_size = poly_eval.get_buffer_size(len);
        let mut temp_buf = Variable::GpuBuffer(GpuBuffer::new(
            stream.unwrap_stream().allocate(buf_size),
            buf_size,
        ));

        poly.cpu2gpu(poly_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
        x.cpu2gpu(x_d.unwrap_scalar_mut(), stream.unwrap_stream());
        func(
            vec![&mut temp_buf, &mut res_d],
            vec![&poly_d, &x_d, &stream],
        )
        .unwrap();
        res_d
            .unwrap_scalar_mut()
            .gpu2cpu(&mut res, stream.unwrap_stream());
        stream
            .unwrap_stream()
            .free(poly_d.unwrap_scalar_array().values);
        stream.unwrap_stream().free(x_d.unwrap_scalar().value);
        stream.unwrap_stream().free(res_d.unwrap_scalar().value);
        stream
            .unwrap_stream()
            .free(temp_buf.unwrap_gpu_buffer().ptr);
        stream.unwrap_stream().sync();
        let mut truth = MyField::zero();
        for i in (0..len).rev() {
            truth = truth * *x.as_ref() + poly.as_ref()[i];
        }
        assert_eq!(*res.as_ref(), truth);
    }

    #[test]
    fn test_kate() {
        let log_len = K;
        let len = 1 << K;
        let mut libs = Libs::new();
        let kate = KateDivision::<MyRuntimeType>::new(&mut libs);
        let func = match kate.get_fn() {
            Function {
                f: FunctionValue::Fn(func),
                ..
            } => func,
            _ => unreachable!(),
        };
        let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
        let stream = Variable::Stream(CudaStream::new(0));
        let mut poly = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut b = Scalar::new_cpu();
        let mut res = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut rng = XorShiftRng::from_seed([0; 16]);
        for i in 0..len {
            poly.as_mut()[i] = MyField::random(&mut rng);
        }
        *b.as_mut() = MyField::random(&mut rng);
        let mut poly_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));
        let mut b_d = Variable::Scalar(Scalar::new_gpu(stream.unwrap_stream().allocate(1), 0));
        let mut res_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));
        let buf_size = kate.get_buffer_size(log_len);
        let mut temp_buf = Variable::GpuBuffer(GpuBuffer::new(
            stream.unwrap_stream().allocate(buf_size),
            buf_size,
        ));

        poly.cpu2gpu(poly_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
        b.cpu2gpu(b_d.unwrap_scalar_mut(), stream.unwrap_stream());
        func(
            vec![&mut temp_buf, &mut res_d],
            vec![&poly_d, &b_d, &stream],
        )
        .unwrap();
        res_d
            .unwrap_scalar_array()
            .gpu2cpu(&mut res, stream.unwrap_stream());
        stream
            .unwrap_stream()
            .free(poly_d.unwrap_scalar_array().values);
        stream.unwrap_stream().free(b_d.unwrap_scalar().value);
        stream
            .unwrap_stream()
            .free(res_d.unwrap_scalar_array().values);
        stream
            .unwrap_stream()
            .free(temp_buf.unwrap_gpu_buffer().ptr);
        stream.unwrap_stream().sync();

        let truth = kate_division(poly.as_ref(), *b.as_ref());
        for i in 0..len - 1 {
            assert_eq!(res.as_ref()[i], truth[i]);
        }
    }

    #[test]
    fn test_zero_one() {
        let len = 1 << K;
        let mut libs = Libs::new();
        let zero = PolyZero::<MyRuntimeType>::new(&mut libs);
        let func = match zero.get_fn() {
            Function {
                f: FunctionValue::Fn(func),
                ..
            } => func,
            _ => unreachable!(),
        };
        let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
        let stream = Variable::Stream(CudaStream::new(0));
        let mut poly = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);

        let mut poly_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));

        func(vec![&mut poly_d], vec![&stream]).unwrap();
        poly_d
            .unwrap_scalar_array()
            .gpu2cpu(&mut poly, stream.unwrap_stream());

        stream.unwrap_stream().sync();

        let truth: Vec<_> = (0..len).into_iter().map(|_| MyField::ZERO).collect();
        for i in 0..len {
            assert_eq!(poly.as_ref()[i], truth[i]);
        }

        let one = PolyOne::<MyRuntimeType>::new(&mut libs);
        let func = match one.get_fn() {
            Function {
                f: FunctionValue::Fn(func),
                ..
            } => func,
            _ => unreachable!(),
        };

        func(vec![&mut poly_d], vec![&stream]).unwrap();
        poly_d
            .unwrap_scalar_array()
            .gpu2cpu(&mut poly, stream.unwrap_stream());

        stream.unwrap_stream().sync();

        let truth: Vec<_> = (0..len).into_iter().map(|_| MyField::ONE).collect();
        for i in 0..len {
            assert_eq!(poly.as_ref()[i], truth[i]);
        }

        stream
            .unwrap_stream()
            .free(poly_d.unwrap_scalar_array().values);
    }

    #[test]
    fn test_scan() {
        let len = 1 << K;
        let mut libs = Libs::new();
        let scan = PolyScan::<MyRuntimeType>::new(&mut libs);
        let func = match scan.get_fn() {
            Function {
                f: FunctionValue::Fn(func),
                ..
            } => func,
            _ => unreachable!(),
        };
        let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
        let stream = Variable::Stream(CudaStream::new(0));
        let mut poly = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut x0 = Scalar::new_cpu();
        let mut res = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut rng = XorShiftRng::from_seed([0; 16]);
        for i in 0..len {
            poly.as_mut()[i] = MyField::random(&mut rng);
        }
        *x0.as_mut() = MyField::random(&mut rng);
        let mut poly_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));
        let mut x0_d = Variable::Scalar(Scalar::new_gpu(stream.unwrap_stream().allocate(1), 0));
        let mut res_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));
        let buf_size = scan.get_buffer_size(len.try_into().unwrap());
        let mut temp_buf = Variable::GpuBuffer(GpuBuffer::new(
            stream.unwrap_stream().allocate(buf_size),
            buf_size,
        ));

        poly.cpu2gpu(poly_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
        x0.cpu2gpu(x0_d.unwrap_scalar_mut(), stream.unwrap_stream());
        func(
            vec![&mut temp_buf, &mut res_d],
            vec![&poly_d, &x0_d, &stream],
        )
        .unwrap();
        res_d
            .unwrap_scalar_array()
            .gpu2cpu(&mut res, stream.unwrap_stream());
        stream
            .unwrap_stream()
            .free(poly_d.unwrap_scalar_array().values);
        stream.unwrap_stream().free(x0_d.unwrap_scalar().value);
        stream
            .unwrap_stream()
            .free(res_d.unwrap_scalar_array().values);
        stream
            .unwrap_stream()
            .free(temp_buf.unwrap_gpu_buffer().ptr);
        stream.unwrap_stream().sync();

        assert_eq!(res.as_ref()[0], *x0.as_ref());
        for i in 1..len {
            assert_eq!(res.as_ref()[i], res.as_ref()[i - 1] * poly.as_ref()[i - 1]);
        }
    }

    #[test]
    fn test_invert() {
        let len = 1 << K;
        let mut libs = Libs::new();
        let invert = PolyInvert::<MyRuntimeType>::new(&mut libs);
        let func = match invert.get_fn() {
            Function {
                f: FunctionValue::Fn(func),
                ..
            } => func,
            _ => unreachable!(),
        };
        let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
        let stream = Variable::Stream(CudaStream::new(0));
        let mut poly = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut inv = Scalar::<MyField>::new_cpu();
        let mut res = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
        let mut rng = XorShiftRng::from_seed([0; 16]);
        for i in 0..len {
            poly.as_mut()[i] = if i % 17 == 0 {
                MyField::ZERO
            } else {
                MyField::random(&mut rng)
            };
        }
        let mut poly_d = Variable::ScalarArray(ScalarArray::new(
            len,
            stream.unwrap_stream().allocate(len),
            DeviceType::GPU { device_id: 0 },
        ));
        let mut inv_d = Variable::Scalar(Scalar::new_gpu(stream.unwrap_stream().allocate(1), 0));

        let buf_size = invert.get_buffer_size(len.try_into().unwrap());
        let mut temp_buf = Variable::GpuBuffer(GpuBuffer::new(
            stream.unwrap_stream().allocate(buf_size),
            buf_size,
        ));

        poly.cpu2gpu(poly_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
        func(vec![&mut temp_buf, &mut poly_d, &mut inv_d], vec![&stream]).unwrap();
        poly_d
            .unwrap_scalar_array()
            .gpu2cpu(&mut res, stream.unwrap_stream());
        inv_d
            .unwrap_scalar()
            .gpu2cpu(&mut inv, stream.unwrap_stream());
        stream
            .unwrap_stream()
            .free(poly_d.unwrap_scalar_array().values);
        stream
            .unwrap_stream()
            .free(temp_buf.unwrap_gpu_buffer().ptr);
        stream.unwrap_stream().free(inv_d.unwrap_scalar().value);
        stream.unwrap_stream().sync();

        let truth = poly.as_mut().batch_invert();
        assert_eq!(*inv.as_ref(), truth);

        for i in 0..len - 1 {
            assert_eq!(res.as_ref()[i], poly.as_ref()[i]);
        }
    }
}
