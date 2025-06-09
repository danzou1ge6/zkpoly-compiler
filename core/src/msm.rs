use std::any::type_name;
use std::io::{BufRead, BufReader};
use std::marker::PhantomData;
use std::os::raw::{c_long, c_uint, c_ulonglong, c_void};
use std::process::{Command, Stdio};
use std::ptr::{null, null_mut};

use libloading::Symbol;
use zkpoly_common::get_project_root::get_project_root;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_common::msm_config::MsmConfig;
use zkpoly_cuda_api::bindings::cudaError_t;
use zkpoly_cuda_api::cuda_check;
use zkpoly_runtime::args::{RuntimeType, Variable};
use zkpoly_runtime::error::RuntimeError;
use zkpoly_runtime::functions::{
    FuncMeta, Function, FunctionValue, KernelType, RegisteredFunction,
};
use zkpoly_runtime::point::PointArray;

use crate::build_func::{make_run, resolve_curve};

fn get_curve_bits<T: RuntimeType>() -> (usize, usize) {
    // get the number of bits of the point and scalar
    let name = resolve_curve(type_name::<T::PointAffine>());
    if name == "bls12_381" {
        return (381, 255);
    } else if name == "bn254" {
        return (254, 254);
    } else {
        unimplemented!("curve not supported")
    }
}

pub fn get_best_config<T: RuntimeType>(len: usize, batches: u32, mem_limit: usize) -> MsmConfig {
    let (point_bits, scalar_bits) = get_curve_bits::<T>();
    let degree = 64 - (len as u64 - 1).leading_zeros(); // log2_ceil(len)

    // call python script to get the best config
    let python_script = "core/src/msm/tuning/cost_model.py";
    let output = Command::new("python3")
        .current_dir(get_project_root())
        .arg(python_script)
        .arg(format!("--k={}", degree))
        .arg(format!("--n={}", batches))
        .arg(format!("--l={}", scalar_bits))
        .arg(format!("--p={}", point_bits))
        .arg(format!("--mem={}", mem_limit))
        .stdout(Stdio::piped()) // pipe the output
        .spawn()
        .expect("Failed to start Python script");

    let mut results = Vec::new();
    // read the output
    if let Some(stdout) = output.stdout {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            // if empty line, continue
            if line.as_ref().unwrap().is_empty() {
                continue;
            }
            match line {
                Ok(line) => {
                    // each line is a number
                    match line.trim().parse::<u64>() {
                        Ok(number) => results.push(number),
                        Err(err) => panic!("Failed to parse number: {}", err),
                    }
                }
                Err(err) => panic!("Failed to read line: {}", err),
            }
        }
    } else {
        panic!("Failed to get stdout");
    }

    let (precompute, window, _, parts, batch_per_run) = (
        results[0] as u32,
        results[1] as u32,
        results[2] as u32,
        results[3] as u32,
        results[4] as u32,
    );

    let (stage_scalars, stage_points) = if parts == 1 { (1, 1) } else { (2, 2) };

    MsmConfig::new(
        window,
        precompute,
        vec![0],
        false,
        batch_per_run,
        parts,
        stage_scalars,
        stage_points,
        scalar_bits as u32,
    )
}

fn get_msm_lib_name<T: RuntimeType>(config: MsmConfig) -> String {
    let mut lib_name = "msm".to_string();
    lib_name += "_";
    lib_name += resolve_curve(type_name::<T::PointAffine>());
    lib_name += "_";
    lib_name += &config.window_size.to_string();
    lib_name += "_";
    lib_name += &config.target_window.to_string();
    lib_name += "_";
    lib_name += &config.debug.to_string();
    lib_name
}

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
    config: MsmConfig,
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
    pub fn new(libs: &mut Libs, config: MsmConfig) -> Self {
        let lib_name = get_msm_lib_name::<T>(config.clone());
        let lib_path = "lib".to_string() + &lib_name + ".so";
        let target_name = "lib/".to_string() + &lib_path;
        if !libs.contains(&lib_path) {
            make_run(&target_name, "Makefile.msm");
        }

        // load the dynamic library
        let lib = libs.load(&lib_path);
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
    pub fn new(libs: &mut Libs, config: MsmConfig) -> Self {
        let lib_name = get_msm_lib_name::<T>(config.clone());
        let lib_path = "lib".to_string() + &lib_name + ".so";
        let target_name = "lib/".to_string() + &lib_path;
        if !libs.contains(&lib_path) {
            make_run(&target_name, "Makefile.msm");
        }

        // load the dynamic library
        let lib = libs.load(&lib_path);
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
                self.config.stage_scalars,
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
                .map(|i: usize| {
                    let array = var[i].unwrap_scalar_array();
                    assert!(
                        array.get_rotation() == 0,
                        "currently, we don't support rotate in msm"
                    );
                    array.values as *const c_uint
                })
                .collect();
            let (buffers, answers) = mut_var.split_at_mut(config.cards.len());
            assert_eq!(buffers.len(), config.cards.len());
            let buffers = buffers
                .iter_mut()
                .map(|v| v.unwrap_gpu_buffer_mut().ptr as *mut c_void)
                .collect::<Vec<*mut c_void>>();
            let h_result = answers
                .iter_mut()
                .map(|v| v.unwrap_point_mut().as_mut() as *mut T::PointAffine as *mut c_uint)
                .collect::<Vec<*mut c_uint>>();

            unsafe {
                cuda_check!(c_func(
                    buffers.as_ptr() as *const *mut c_void,
                    null_mut(),
                    len.try_into().unwrap(),
                    config.batch_per_run,
                    config.parts,
                    config.stage_scalars,
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
            meta: FuncMeta::new("msm".to_string(), KernelType::Msm(self.config.clone())),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}
