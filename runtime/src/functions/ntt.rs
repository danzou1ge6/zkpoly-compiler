use std::any::type_name;
use std::marker::PhantomData;
use std::os::raw::c_uint;
use std::ptr::null;

use libloading::Symbol;
use zkpoly_cuda_api::cuda_check;

use crate::args::{RuntimeType, Variable};
use crate::error::RuntimeError;
use crate::scalar::ScalarArray;

use super::build_func::{resolve_type, xmake_config, xmake_run};
use super::load_dynamic::Libs;
use super::{Function, FunctionValue, RegisteredFunction};

use zkpoly_cuda_api::bindings::{
    cudaError_cudaSuccess, cudaError_t, cudaGetErrorString, cudaSetDevice, cudaStream_t,
};

pub struct SsipNtt<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            x: *mut c_uint,
            twiddle: *const c_uint,
            log_len: c_uint,
            stream: cudaStream_t,
            max_threads_stage1_log: c_uint,
            max_threads_stage2_log: c_uint,
        ) -> cudaError_t,
    >,
}

// this is used by the compiler, so we don't need to implement the RegisteredFunction trait
pub struct SsipPrecompute<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            twiddle: *mut c_uint,
            log_len: c_uint,
            unit: *const c_uint,
        ) -> cudaError_t,
    >,
}

impl<T: RuntimeType> RegisteredFunction<T> for SsipNtt<T> {
    fn new(libs: &mut Libs) -> Self {
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("NTT_FIELD", field_type);
        xmake_run("ntt");

        // load the dynamic library
        let lib = libs.load("../lib/libntt.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"ssip_ntt\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }

    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();

        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert!(mut_var.len() == 1);
            assert!(var.len() == 2);
            let x = mut_var[0].unwrap_scalar_array_mut();
            assert!(x.len.is_power_of_two());
            let log_len = x.len.trailing_zeros();
            let twiddle = var[0].unwrap_scalar_array();
            assert_eq!(twiddle.len * 2, x.len);
            let stream = var[1].unwrap_stream();
            let max_threads_stage1_log = match size_of::<T::Field>() {
                32 => 8,
                _ => unimplemented!(),
            };
            let max_threads_stage2_log = match size_of::<T::Field>() {
                32 => 8,
                _ => unimplemented!(),
            };
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!((c_func)(
                    x.values as *mut c_uint,
                    twiddle.values as *const c_uint,
                    log_len,
                    stream.raw(),
                    max_threads_stage1_log,
                    max_threads_stage2_log,
                ));
            }
            Ok(())
        };
        Function {
            name: "ntt".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> SsipPrecompute<T> {
    pub fn new(libs: &mut Libs) -> Self {
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("NTT_FIELD", field_type);
        xmake_run("ntt");

        // load the dynamic library
        let lib = libs.load("../lib/libntt.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"ssip_precompute\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }

    pub fn get_fn(&self) -> Box<dyn Fn(&mut ScalarArray<T::Field>, &T::Field) -> Result<(), RuntimeError>> {
        let c_func = self.c_func.clone();

        Box::new(move |twiddle: &mut ScalarArray<T::Field>, unit: &T::Field| -> Result<(), RuntimeError> {
            assert!(twiddle.len.is_power_of_two());
            unsafe {
                cuda_check!((c_func)(
                    twiddle.values as *mut c_uint,
                    twiddle.len as c_uint,
                    unit as *const T::Field as *const c_uint,
                ));
            }
            Ok(())
        })
        
    }
}
#[test]
fn test_ntt() {
    
    let max_k = 20;

    use crate::args::Variable;
    use crate::transcript::{Blake2bWrite, Challenge255};
    use group::ff::Field;
    use halo2curves::bn256;
    use rand_core::OsRng;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use halo2_proofs::arithmetic;
    use zkpoly_memory_pool::PinnedMemoryPool;
    use zkpoly_cuda_api::stream::CudaStream;
    use crate::devices::DeviceType;
    use crate::runtime::transfer::Transfer;

    #[derive(Debug, Clone)]
    struct MyRuntimeType;

    impl RuntimeType for MyRuntimeType {
        type Field = bn256::Fr;
        type PointAffine = bn256::G1Affine;
        type Challenge = Challenge255<bn256::G1Affine>;
        type Trans = Blake2bWrite<Vec<u8>, bn256::G1Affine, Challenge255<bn256::G1Affine>>;
    }

    type MyField = <MyRuntimeType as RuntimeType>::Field;

    let mut libs = Libs::new();
    let ntt = SsipNtt::<MyRuntimeType>::new(&mut libs);
    let precompute = SsipPrecompute::<MyRuntimeType>::new(&mut libs);
    let ntt_fn = match ntt.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => panic!("expected Fn"),
    };

    let precompute_fn = precompute.get_fn();
    let cpu_alloc = PinnedMemoryPool::new(max_k, size_of::<MyField>());
    let stream = Variable::<MyRuntimeType>::Stream(CudaStream::new(0));

    for k in 1..=max_k {
        println!("generating data for k = {k}...");
        let mut data_rust: Vec<_> = (0..(1 << k))
            .into_iter()
            .map(|_| MyField::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();

        let mut poly_cpu = ScalarArray::<MyField>::new(1 << k, cpu_alloc.allocate(1 << k), DeviceType::CPU);

        unsafe {
            std::ptr::copy_nonoverlapping(data_rust.as_ptr(), poly_cpu.values, 1 << k);
        }

        let omega = MyField::random(XorShiftRng::from_rng(OsRng).unwrap()); // would be weird if this mattered

        println!("testing on cpu for k = {k}...");
        arithmetic::best_fft(&mut data_rust, omega, k as u32);

        println!("precomputing twiddle factors for k = {k}...");

        let mut twiddle = ScalarArray::<MyField>::new(1 << (k-1), cpu_alloc.allocate(1 << (k-1)), DeviceType::CPU);
        precompute_fn(&mut twiddle, &omega).unwrap();

        println!("transferring data to gpu for k = {k}...");

        let ptr_data: *mut MyField = stream.unwrap_stream().allocate(1 << k);
        let ptr_twiddle: *mut MyField = stream.unwrap_stream().allocate(1 << (k-1));

        let mut poly_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(1 << k, ptr_data, DeviceType::GPU{device_id: 0}));
        let mut twiddle_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(1 << (k-1), ptr_twiddle, DeviceType::GPU{device_id: 0}));

        poly_cpu.cpu2gpu(poly_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());
        twiddle.cpu2gpu(twiddle_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());

        stream.unwrap_stream().sync();

        println!("testing on gpu for k = {k}...");

        ntt_fn(vec![&mut poly_gpu], vec![&twiddle_gpu, &stream]).unwrap();

        stream.unwrap_stream().sync();

        println!("transferring data back to cpu for k = {k}...");

        poly_gpu.unwrap_scalar_array().gpu2cpu(&mut poly_cpu, stream.unwrap_stream());

        stream.unwrap_stream().sync();

        println!("comparing results for k = {k}...");

        let data_cuda = unsafe{ std::slice::from_raw_parts(poly_cpu.values, 1 << k) };

        data_cuda
            .iter()
            .zip(data_rust.iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });
    }
}
