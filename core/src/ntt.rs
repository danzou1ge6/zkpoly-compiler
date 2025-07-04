use std::any::type_name;
use std::ffi::c_ulonglong;
use std::marker::PhantomData;
use std::os::raw::c_uint;
use std::sync::Arc;

use libloading::Symbol;
use zkpoly_cuda_api::cuda_check;

use zkpoly_runtime::args::{RuntimeType, Variable};
use zkpoly_runtime::error::RuntimeError;
use zkpoly_runtime::scalar::ScalarArray;

use crate::poly_ptr::PolyPtr;

use super::build_func::{resolve_type, xmake_config, xmake_run};
use zkpoly_common::load_dynamic::Libs;
use zkpoly_runtime::functions::{FuncMeta, Function, KernelType, RegisteredFunction};

use zkpoly_cuda_api::bindings::{cudaError_t, cudaSetDevice, cudaStream_t};

static LIB_NAME: &str = "libntt.so";

pub struct SsipNtt<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            x: PolyPtr,
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
            x: PolyPtr,
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

pub struct DistributePowers<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            x: PolyPtr,
            powers: *const c_uint,
            power_num: c_ulonglong,
            stream: cudaStream_t,
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

fn load_lib<T: RuntimeType>(libs: &mut Libs) -> &'static libloading::Library {
    if !libs.contains(LIB_NAME) {
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("NTT_FIELD", field_type);
        xmake_run("ntt");
    }
    libs.load(LIB_NAME)
}

impl<T: RuntimeType> DistributePowers<T> {
    pub fn new(libs: &mut Libs) -> Self {
        // load the dynamic library
        let lib = load_lib::<T>(libs);
        // get the function pointer
        let c_func = unsafe { lib.get(b"distribute_powers\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for DistributePowers<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>,
                              _: Arc<dyn Fn(i32) -> i32 + Send + Sync>|
              -> Result<(), RuntimeError> {
            assert!(mut_var.len() == 1);
            assert!(var.len() == 2);
            let x = mut_var[0].unwrap_scalar_array_mut();
            let powers = var[0].unwrap_scalar_array();
            let stream = var[1].unwrap_stream();
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!((c_func)(
                    PolyPtr::from(x),
                    powers.values as *const c_uint,
                    powers.len as u64,
                    stream.raw(),
                ));
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new(
                "distribute_powers".to_string(),
                KernelType::DistributePowers,
            ),
            f: Arc::new(rust_func),
        }
    }
}

impl<T: RuntimeType> SsipNtt<T> {
    pub fn new(libs: &mut Libs) -> Self {
        // load the dynamic library
        let lib = load_lib::<T>(libs);
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
                              var: Vec<&Variable<T>>,
                              _: Arc<dyn Fn(i32) -> i32 + Send + Sync>|
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
                    PolyPtr::from(x),
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
            meta: FuncMeta::new("ssip_ntt".to_string(), KernelType::NttPrcompute),
            f: Arc::new(rust_func),
        }
    }
}

impl<T: RuntimeType> SsipPrecompute<T> {
    pub fn new(libs: &mut Libs) -> Self {
        // load the dynamic library
        let lib = load_lib::<T>(libs);
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
    pub fn new(libs: &mut Libs) -> Self {
        // load the dynamic library
        let lib = load_lib::<T>(libs);
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
                              var: Vec<&Variable<T>>,
                              _: Arc<dyn Fn(i32) -> i32 + Send + Sync>|
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
                    PolyPtr::from(x),
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
            meta: FuncMeta::new("recompute_ntt".to_string(), KernelType::NttRecompute),
            f: Arc::new(rust_func),
        }
    }
}

impl<T: RuntimeType> GenPqOmegas<T> {
    pub fn new(libs: &mut Libs) -> Self {
        // load the dynamic library
        let lib = load_lib::<T>(libs);
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
