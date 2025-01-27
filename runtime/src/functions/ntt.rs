use std::any::type_name;
use std::marker::PhantomData;
use std::os::raw::c_uint;

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

// recompute ntt with memory consumption sizeof(data) + constant(at most (2^11 + 32) * sizeof(Field))
pub struct RecomputeNtt<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            x: *mut c_uint,
            pq: *const c_uint,
            pq_deg: c_uint,
            omegas: *const c_uint,
            log_len: c_uint,
            stream: cudaStream_t,
            max_threads_stage1_log: c_uint,
            max_threads_stage2_log: c_uint,
        ) -> cudaError_t,
    >,
}

pub struct GenPqOmegas<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            pq: *mut c_uint,
            omegas: *mut c_uint,
            pq_deg: c_uint,
            len: c_uint,
            unit: *const c_uint,
        ),
    >,
}

impl<T: RuntimeType> SsipNtt<T> {
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
}

impl<T: RuntimeType> RegisteredFunction<T> for SsipNtt<T> {
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
            let (max_threads_stage1_log, max_threads_stage2_log) = get_stage_threads::<T>();
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

    pub fn get_fn(
        &self,
    ) -> Box<dyn Fn(&mut ScalarArray<T::Field>, &T::Field) -> Result<(), RuntimeError>> {
        let c_func = self.c_func.clone();

        Box::new(
            move |twiddle: &mut ScalarArray<T::Field>,
                  unit: &T::Field|
                  -> Result<(), RuntimeError> {
                assert!(twiddle.len.is_power_of_two());
                unsafe {
                    cuda_check!((c_func)(
                        twiddle.values as *mut c_uint,
                        twiddle.len as c_uint,
                        unit as *const T::Field as *const c_uint,
                    ));
                }
                Ok(())
            },
        )
    }
}

impl<T: RuntimeType> RecomputeNtt<T> {
    fn new(libs: &mut Libs) -> Self {
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("NTT_FIELD", field_type);
        xmake_run("ntt");

        // load the dynamic library
        let lib = libs.load("../lib/libntt.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"recompute_ntt\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for RecomputeNtt<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();

        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert!(mut_var.len() == 1);
            assert!(var.len() == 3);
            let x = mut_var[0].unwrap_scalar_array_mut();
            assert!(x.len.is_power_of_two());
            let log_len = x.len.trailing_zeros();
            let pq = var[0].unwrap_scalar_array();
            assert!(pq.len.is_power_of_two());
            let pq_deg = pq.len.trailing_zeros() + 1;
            let omegas = var[1].unwrap_scalar_array();
            assert!(omegas.len == 32);
            let stream = var[2].unwrap_stream();
            let (max_threads_stage1_log, max_threads_stage2_log) = get_stage_threads::<T>();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!((c_func)(
                    x.values as *mut c_uint,
                    pq.values as *const c_uint,
                    pq_deg,
                    omegas.values as *const c_uint,
                    log_len,
                    stream.raw(),
                    max_threads_stage1_log,
                    max_threads_stage2_log,
                ));
            }
            Ok(())
        };
        Function {
            name: "recompute_ntt".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

impl<T: RuntimeType> GenPqOmegas<T> {
    pub fn new(libs: &mut Libs) -> Self {
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("NTT_FIELD", field_type);
        xmake_run("ntt");

        // load the dynamic library
        let lib = libs.load("../lib/libntt.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"gen_pq_omegas\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }

    // plan partition for NTT stages
    fn get_deg(deg_stage: u32, max_deg_stage: u32) -> u32 {
        let mut deg_per_round: u32;
        let mut rounds = 1;
        loop {
            deg_per_round = if rounds == 1 {
                deg_stage
            } else {
                (deg_stage - 1) / rounds + 1
            };
            if deg_per_round <= max_deg_stage {
                break;
            }
            rounds += 1;
        }
        deg_per_round
    }

    pub fn get_pq_len(&self, log_len: u32) -> usize {
        let (max_threads_stage1_log, max_threads_stage2_log) = get_stage_threads::<T>();

        let total_deg_stage1 = (log_len + 1) / 2;
        let total_deg_stage2 = log_len / 2;

        let max_deg_stage1 = max_threads_stage1_log + 1;
        let max_deg_stage2 = (max_threads_stage2_log + 2) / 2; // 4 elements per thread

        let deg_stage1 = Self::get_deg(total_deg_stage1, max_deg_stage1);
        let deg_stage2 = Self::get_deg(total_deg_stage2, max_deg_stage2);

        1 << (std::cmp::max(deg_stage1, deg_stage2) - 1)
    }

    pub fn get_fn(
        &self,
    ) -> Box<
        dyn Fn(
            &mut ScalarArray<T::Field>,
            &mut ScalarArray<T::Field>,
            usize,
            &T::Field,
        ) -> Result<(), RuntimeError>,
    > {
        let c_func = self.c_func.clone();

        Box::new(
            move |pq: &mut ScalarArray<T::Field>,
                  omegas: &mut ScalarArray<T::Field>,
                  len: usize,
                  unit: &T::Field|
                  -> Result<(), RuntimeError> {
                assert!(omegas.len == 32);
                let pq_deg = pq.len.trailing_zeros() + 1;

                unsafe {
                    ((c_func)(
                        pq.values as *mut c_uint,
                        omegas.values as *mut c_uint,
                        pq_deg,
                        len.try_into().unwrap(),
                        unit as *const T::Field as *const c_uint,
                    ));
                }
                Ok(())
            },
        )
    }
}

fn get_stage_threads<T: RuntimeType>() -> (u32, u32) {
    match size_of::<T::Field>() {
        32 => (8, 8),
        _ => unimplemented!(),
    }
}

#[cfg(test)]
mod tests {
    static MAX_K: u32 = 20;

    use crate::args::RuntimeType;
    use crate::args::Variable;
    use crate::devices::DeviceType;
    use crate::functions::load_dynamic::Libs;
    use crate::functions::RegisteredFunction;
    use crate::runtime::transfer::Transfer;
    use crate::transcript::{Blake2bWrite, Challenge255};
    use group::ff::Field;
    use halo2_proofs::arithmetic;
    use halo2curves::bn256;
    use rand_core::OsRng;
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

    #[test]
    fn test_ssip_ntt() {
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
        let cpu_alloc = PinnedMemoryPool::new(MAX_K, size_of::<MyField>());
        let stream = Variable::<MyRuntimeType>::Stream(CudaStream::new(0));

        for k in 1..=MAX_K {
            println!("generating data for k = {k}...");
            let mut data_rust: Vec<_> = (0..(1 << k))
                .into_iter()
                .map(|_| MyField::random(XorShiftRng::from_rng(OsRng).unwrap()))
                .collect();

            let mut poly_cpu =
                ScalarArray::<MyField>::new(1 << k, cpu_alloc.allocate(1 << k), DeviceType::CPU);

            unsafe {
                std::ptr::copy_nonoverlapping(data_rust.as_ptr(), poly_cpu.values, 1 << k);
            }

            let omega = MyField::random(XorShiftRng::from_rng(OsRng).unwrap()); // would be weird if this mattered

            println!("testing on cpu for k = {k}...");
            arithmetic::best_fft(&mut data_rust, omega, k as u32);

            println!("precomputing twiddle factors for k = {k}...");

            let mut twiddle = ScalarArray::<MyField>::new(
                1 << (k - 1),
                cpu_alloc.allocate(1 << (k - 1)),
                DeviceType::CPU,
            );
            precompute_fn(&mut twiddle, &omega).unwrap();

            println!("testing on gpu for k = {k}...");

            let start = std::time::Instant::now();

            let ptr_data: *mut MyField = stream.unwrap_stream().allocate(1 << k);
            let ptr_twiddle: *mut MyField = stream.unwrap_stream().allocate(1 << (k - 1));

            let mut poly_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
                1 << k,
                ptr_data,
                DeviceType::GPU { device_id: 0 },
            ));
            let mut twiddle_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
                1 << (k - 1),
                ptr_twiddle,
                DeviceType::GPU { device_id: 0 },
            ));

            poly_cpu.cpu2gpu(poly_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());
            twiddle.cpu2gpu(
                twiddle_gpu.unwrap_scalar_array_mut(),
                stream.unwrap_stream(),
            );

            ntt_fn(vec![&mut poly_gpu], vec![&twiddle_gpu, &stream]).unwrap();

            poly_gpu
                .unwrap_scalar_array()
                .gpu2cpu(&mut poly_cpu, stream.unwrap_stream());

            let end1 = std::time::Instant::now();

            stream.unwrap_stream().sync();

            let end2 = std::time::Instant::now();
            let dur1 = end1 - start;
            let dur2 = end2 - start;

            println!(
                "time for k = {k}: {:?} (launch) + {:?} (actual)",
                dur1, dur2
            );

            stream.unwrap_stream().free(ptr_data);
            stream.unwrap_stream().free(ptr_twiddle);

            println!("comparing results for k = {k}...");

            let data_cuda = unsafe { std::slice::from_raw_parts(poly_cpu.values, 1 << k) };

            data_cuda.iter().zip(data_rust.iter()).for_each(|(a, b)| {
                assert_eq!(a, b);
            });

            cpu_alloc.free(poly_cpu.values);
            cpu_alloc.free(twiddle.values);
        }
    }

    #[test]
    fn test_recompute_ntt() {
        let mut libs = Libs::new();
        let ntt = RecomputeNtt::<MyRuntimeType>::new(&mut libs);
        let precompute = GenPqOmegas::<MyRuntimeType>::new(&mut libs);
        let ntt_fn = match ntt.get_fn() {
            Function {
                f: FunctionValue::Fn(func),
                ..
            } => func,
            _ => panic!("expected Fn"),
        };

        let precompute_fn = precompute.get_fn();
        let cpu_alloc = PinnedMemoryPool::new(MAX_K, size_of::<MyField>());
        let stream = Variable::<MyRuntimeType>::Stream(CudaStream::new(0));

        for k in 1..=MAX_K {
            println!("generating data for k = {k}...");
            let mut data_rust: Vec<_> = (0..(1 << k))
                .into_iter()
                .map(|_| MyField::random(XorShiftRng::from_rng(OsRng).unwrap()))
                .collect();

            let mut poly_cpu =
                ScalarArray::<MyField>::new(1 << k, cpu_alloc.allocate(1 << k), DeviceType::CPU);

            unsafe {
                std::ptr::copy_nonoverlapping(data_rust.as_ptr(), poly_cpu.values, 1 << k);
            }

            let omega = MyField::random(XorShiftRng::from_rng(OsRng).unwrap()); // would be weird if this mattered

            println!("testing on cpu for k = {k}...");
            arithmetic::best_fft(&mut data_rust, omega, k as u32);

            println!("precomputing twiddle factors for k = {k}...");

            let mut pq = ScalarArray::<MyField>::new(
                precompute.get_pq_len(k),
                cpu_alloc.allocate(precompute.get_pq_len(k)),
                DeviceType::CPU,
            );
            let mut omegas =
                ScalarArray::<MyField>::new(32, cpu_alloc.allocate(32), DeviceType::CPU);
            precompute_fn(&mut pq, &mut omegas, 1 << k, &omega).unwrap();

            println!("transferring data to gpu for k = {k}...");

            let ptr_data: *mut MyField = stream.unwrap_stream().allocate(1 << k);
            let ptr_pq: *mut MyField = stream.unwrap_stream().allocate(precompute.get_pq_len(k));
            let ptr_omegas: *mut MyField = stream.unwrap_stream().allocate(32);

            let mut poly_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
                1 << k,
                ptr_data,
                DeviceType::GPU { device_id: 0 },
            ));
            let mut pq_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
                precompute.get_pq_len(k),
                ptr_pq,
                DeviceType::GPU { device_id: 0 },
            ));
            let mut omegas_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
                32,
                ptr_omegas,
                DeviceType::GPU { device_id: 0 },
            ));

            println!("testing on gpu for k = {k}...");

            poly_cpu.cpu2gpu(poly_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());
            pq.cpu2gpu(pq_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());
            omegas.cpu2gpu(omegas_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());

            ntt_fn(vec![&mut poly_gpu], vec![&pq_gpu, &omegas_gpu, &stream]).unwrap();

            poly_gpu
                .unwrap_scalar_array()
                .gpu2cpu(&mut poly_cpu, stream.unwrap_stream());

            stream.unwrap_stream().sync();
            stream.unwrap_stream().free(ptr_data);
            stream.unwrap_stream().free(ptr_pq);
            stream.unwrap_stream().free(ptr_omegas);

            println!("comparing results for k = {k}...");

            let data_cuda = unsafe { std::slice::from_raw_parts(poly_cpu.values, 1 << k) };

            data_cuda.iter().zip(data_rust.iter()).for_each(|(a, b)| {
                assert_eq!(a, b);
            });
            cpu_alloc.free(poly_cpu.values);
            cpu_alloc.free(pq.values);
            cpu_alloc.free(omegas.values);
        }
    }
}
