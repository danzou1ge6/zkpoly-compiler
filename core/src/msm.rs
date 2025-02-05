use std::any::type_name;
use std::marker::PhantomData;
use std::os::raw::{c_long, c_uint, c_ulonglong, c_void};
use std::ptr::{null, null_mut};

use libloading::Symbol;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_common::msm_config::MSMConfig;
use zkpoly_cuda_api::bindings::{cudaError_cudaSuccess, cudaError_t, cudaGetErrorString};
use zkpoly_cuda_api::cuda_check;
use zkpoly_runtime::args::{RuntimeType, Variable};
use zkpoly_runtime::error::RuntimeError;
use zkpoly_runtime::functions::{Function, FunctionValue, RegisteredFunction};
use zkpoly_runtime::point::PointArray;

use crate::build_func::{resolve_curve, xmake_config, xmake_run};

pub struct MSM<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            buffers: *const *mut c_void,
            buffer_sizes: *mut c_long,
            len: c_ulonglong,
            batch_per_run: c_uint,
            parts: c_uint,
            stage_scalers: c_uint,
            stage_points: c_uint,
            num_cards: c_uint,
            cards: *const c_uint,
            h_points: *const *const c_uint,
            batches: c_uint,
            h_scaler_batch: *const *const c_uint,
            h_result: *const *mut c_uint,
        ) -> cudaError_t,
    >,
    config: MSMConfig<T::PointAffine>,
}

pub struct MSMPrecompute<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            len: c_ulonglong,
            h_points: *const *mut c_uint,
            max_cards: c_uint,
        ) -> cudaError_t,
    >,
}

impl<T: RuntimeType> MSMPrecompute<T> {
    pub fn new(libs: &mut Libs, config: MSMConfig<T::PointAffine>) -> Self {
        let (curve, bits) = resolve_curve(type_name::<T::PointAffine>());
        xmake_config("MSM_BITS", bits.to_string().as_str());
        xmake_config("MSM_CURVE", curve);
        xmake_config("MSM_WINDOW_SIZE", config.window_size.to_string().as_str());
        xmake_config(
            "MSM_TARGET_WINDOWS",
            config.target_window.to_string().as_str(),
        );
        xmake_config("MSM_DEBUG", (config.debug as u32).to_string().as_str());
        xmake_run("msm");

        // load the dynamic library
        let lib = libs.load("../lib/libmsm.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"msm_precompute\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }
    pub fn get_fn(
        &self,
    ) -> Box<dyn Fn(Vec<&mut PointArray<T::PointAffine>>, usize, u32) -> Result<(), RuntimeError>>
    {
        let c_func = self.c_func.clone();
        let rust_func = move |target: Vec<&mut PointArray<T::PointAffine>>,
                              len: usize,
                              max_cards: u32|
              -> Result<(), RuntimeError> {
            let h_points: Vec<*mut c_uint> =
                target.iter().map(|p| p.values as *mut c_uint).collect();
            unsafe {
                cuda_check!(c_func(
                    len.try_into().unwrap(),
                    h_points.as_ptr(),
                    max_cards,
                ))
            }
            Ok(())
        };
        Box::new(rust_func)
    }
}

impl<T: RuntimeType> MSM<T> {
    pub fn new(libs: &mut Libs, config: MSMConfig<T::PointAffine>) -> Self {
        let (curve, bits) = resolve_curve(type_name::<T::PointAffine>());
        xmake_config("MSM_BITS", bits.to_string().as_str());
        xmake_config("MSM_CURVE", curve);
        xmake_config("MSM_WINDOW_SIZE", config.window_size.to_string().as_str());
        xmake_config(
            "MSM_TARGET_WINDOWS",
            config.target_window.to_string().as_str(),
        );
        xmake_config("MSM_DEBUG", (config.debug as u32).to_string().as_str());
        xmake_run("msm");

        // load the dynamic library
        let lib = libs.load("../lib/libmsm.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"msm\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
            config,
        }
    }

    pub fn get_buffer_size(&self, len: usize) -> Vec<usize> {
        let mut buffer_sizes = vec![0; self.config.cards.len()];
        unsafe {
            cuda_check!((self.c_func)(
                null(),
                buffer_sizes.as_mut_ptr() as *mut c_long,
                len.try_into().unwrap(),
                self.config.batch_per_run,
                self.config.parts,
                self.config.stage_scalers,
                self.config.stage_points,
                self.config.cards.len().try_into().unwrap(),
                self.config.cards.as_ptr(),
                null_mut(),
                0,
                null_mut(),
                null(),
            ))
        }
        buffer_sizes
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for MSM<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let config = self.config.clone();
        let n_precompute = config.get_precompute();

        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            let batches: u32 = var.len() as u32 - n_precompute;
            assert_eq!(mut_var.len(), batches as usize + config.cards.len());
            let len = var[0].unwrap_point_array().len;
            let h_points: Vec<*const c_uint> = (0..n_precompute.try_into().unwrap())
                .map(|i: usize| var[i].unwrap_point_array().values as *const c_uint)
                .collect();
            let h_scaler_batch: Vec<*const c_uint> = (n_precompute.try_into().unwrap()..var.len())
                .map(|i: usize| var[i].unwrap_scalar_array().values as *const c_uint)
                .collect();
            let (buffers, answers) = mut_var.split_at_mut(config.cards.len());
            assert_eq!(buffers.len(), config.cards.len());
            let buffers = buffers
                .iter_mut()
                .map(|v| v.unwrap_gpu_buffer_mut().ptr as *mut c_void)
                .collect::<Vec<*mut c_void>>();
            let h_result = answers
                .iter_mut()
                .map(|v| &mut v.unwrap_point_mut().value as *mut T::PointAffine as *mut c_uint)
                .collect::<Vec<*mut c_uint>>();

            unsafe {
                cuda_check!(c_func(
                    buffers.as_ptr() as *const *mut c_void,
                    null_mut(),
                    len.try_into().unwrap(),
                    config.batch_per_run,
                    config.parts,
                    config.stage_scalers,
                    config.stage_points,
                    config.cards.len().try_into().unwrap(),
                    config.cards.as_ptr(),
                    h_points.as_ptr(),
                    batches,
                    h_scaler_batch.as_ptr(),
                    h_result.as_ptr(),
                ))
            }
            Ok(())
        };

        Function {
            name: "msm".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}
