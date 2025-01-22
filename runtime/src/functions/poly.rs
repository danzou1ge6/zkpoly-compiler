// this file is used to show how to insert a new function into the runtime

use std::{
    any::type_name,
    marker::PhantomData,
    os::raw::{c_uint, c_ulonglong},
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
    use crate::runtime::transfer::Transfer;
    use crate::scalar::ScalarArray;
    use crate::transcript::{Blake2bWrite, Challenge255};
    use group::ff::Field;
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
}
