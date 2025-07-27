use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;
use std::{
    any::type_name,
    marker::PhantomData,
    os::raw::{c_uint, c_ulonglong},
};

use libloading::Symbol;
use zkpoly_common::{
    arith::{ArithGraph, FusedType},
    load_dynamic::Libs,
};
use zkpoly_cuda_api::stream::{CudaEventRaw, CudaStream};
use zkpoly_cuda_api::{
    bindings::{cudaError_t, cudaSetDevice, cudaStream_t},
    cuda_check,
};
use zkpoly_runtime::functions::FusedKernelMeta;
use zkpoly_runtime::runtime::transfer::Transfer;
use zkpoly_runtime::scalar::{Scalar, ScalarArray};
use zkpoly_runtime::{
    args::{RuntimeType, Variable},
    error::RuntimeError,
    functions::{FuncMeta, Function, KernelType, RegisteredFunction},
};

use crate::build_func::{xmake_config_absolute, xmake_run_absolute};
use crate::{
    build_func::{resolve_type, xmake_config, xmake_run},
    poly_ptr::{ConstPolyPtr, PolyPtr},
};

pub mod kernel_gen;

static LIB_NAME: &str = "libfused_kernels.so";

pub struct FusedKernel<T: RuntimeType> {
    _marker: PhantomData<T>,
    pub meta: FusedKernelMeta,
    pub c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            vars: *const ConstPolyPtr,
            mut_vars: *const PolyPtr,
            len: c_ulonglong,
            is_first: bool,
            local_buffer: *mut c_void,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

// all input/output are on cpu
pub struct PipelinedFusedKernel<T: RuntimeType> {
    kernel: FusedKernel<T>,
}

pub struct FusedOp<OuterId, InnerId> {
    graph: ArithGraph<OuterId, InnerId>,
    name: String,
    pub vars: Vec<(FusedType, InnerId)>,
    pub mut_vars: Vec<(FusedType, InnerId)>,
    schedule: Vec<InnerId>,
    partition: Vec<usize>,
    reg_limit: u32,
    field_regs: u32,
    var_mapping: BTreeMap<usize, usize>,
    mut_var_mapping: BTreeMap<usize, usize>,
    limbs: usize,
}

const FIELD_NAME: &str = "FUSED_FIELD";
impl<T: RuntimeType> FusedKernel<T> {
    pub fn new(libs: &mut Libs, meta: FusedKernelMeta) -> Self {
        let lib = if meta.lib_path.is_none() {
            if !libs.contains(LIB_NAME) {
                let field_type = resolve_type(type_name::<T::Field>());
                xmake_config(FIELD_NAME, field_type);
                xmake_run("fused_kernels");
            }
            libs.load_relative(LIB_NAME)
        } else {
            if !libs.contains_absolute(meta.lib_path.as_ref().unwrap()) {
                let field_type = resolve_type(type_name::<T::Field>());
                xmake_config_absolute(FIELD_NAME, field_type, meta.lib_path.as_ref().unwrap());
                xmake_run_absolute("fused_kernels", meta.lib_path.as_ref().unwrap());
            }
            libs.load_absolute(meta.lib_path.as_ref().unwrap())
        };
        // get the function pointer with the provided name (with null terminator)
        let c_func = unsafe { lib.get(format!("{}\0", meta.name).as_bytes()) }
            .expect("Failed to load function pointer");
        Self {
            _marker: PhantomData,
            meta,
            c_func,
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for FusedKernel<T> {
    fn get_fn(&self) -> Function<T> {
        assert!(self.meta.pipelined_meta.is_none());
        let c_func = self.c_func.clone();
        let meta = self.meta.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>,
                              _: Arc<dyn Fn(i32) -> i32 + Send + Sync>|
              -> Result<(), RuntimeError> {
            let mut len = 0;
            assert_eq!(meta.num_vars, var.len() - 1);
            assert_eq!(meta.num_mut_vars, mut_var.len() - 2);
            let stream = var[0].unwrap_stream();
            let (tmp_buffers, mut_vars) = mut_var.split_at_mut(2);
            let arg_buffer = tmp_buffers[0].unwrap_gpu_buffer();
            let local_buffer = tmp_buffers[1].unwrap_gpu_buffer();
            let vars = var
                .iter()
                .skip(1)
                .map(|v| match v {
                    Variable::ScalarArray(poly) => {
                        if len == 0 || len == 1 {
                            len = poly.len;
                        } else {
                            assert_eq!(len, poly.len);
                        }
                        ConstPolyPtr::from(poly)
                    }
                    Variable::Scalar(scalar) => {
                        if len == 0 {
                            len = 1;
                        }
                        ConstPolyPtr {
                            ptr: scalar.value as *const c_uint,
                            len: 1,
                            rotate: 0,
                            offset: 0,
                            whole_len: 1,
                        }
                    }
                    _ => unreachable!("Only scalars and scalar arrays are supported"),
                })
                .collect::<Vec<_>>();
            let mut_vars = mut_vars
                .iter_mut()
                .map(|v| match v {
                    Variable::ScalarArray(poly) => {
                        if len == 0 || len == 1 {
                            len = poly.len;
                        } else {
                            assert_eq!(len, poly.len);
                        }
                        PolyPtr::from(poly)
                    }
                    Variable::Scalar(scalar) => {
                        if len == 0 {
                            len = 1;
                        }
                        PolyPtr {
                            ptr: scalar.value as *mut c_uint,
                            len: 1,
                            rotate: 0,
                            offset: 0,
                            whole_len: 1,
                        }
                    }
                    _ => unreachable!("Only scalars and scalar arrays are supported"),
                })
                .collect::<Vec<_>>();
            assert!(len > 0);
            // copy the arguments to the device
            unsafe {
                let d_vars: *mut ConstPolyPtr = arg_buffer.ptr as *mut ConstPolyPtr;
                let d_mut_vars: *mut PolyPtr = (arg_buffer.ptr as *mut PolyPtr).add(meta.num_vars);

                // do the transfer
                cuda_check!(cudaSetDevice(stream.get_device()));
                stream.memcpy_h2d(d_vars, vars.as_ptr(), meta.num_vars);
                stream.memcpy_h2d(d_mut_vars, mut_vars.as_ptr(), meta.num_mut_vars);

                cuda_check!((c_func)(
                    d_vars,
                    d_mut_vars,
                    len.try_into().unwrap(),
                    true,
                    local_buffer.ptr as *mut c_void,
                    stream.raw()
                ));
            }
            Ok(())
        };
        Function {
            meta: FuncMeta::new(
                self.meta.name.clone(),
                KernelType::FusedArith(self.meta.clone()),
            ),
            f: Arc::new(rust_func),
        }
    }
}

impl<T: RuntimeType> PipelinedFusedKernel<T> {
    pub fn new(libs: &mut Libs, meta: FusedKernelMeta) -> Self {
        assert!(meta.pipelined_meta.is_some());
        let pipelined_meta = meta.pipelined_meta.clone().unwrap();
        assert!(pipelined_meta.divide_parts > 3);
        Self {
            kernel: FusedKernel::new(libs, meta),
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PipelinedFusedKernel<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.kernel.c_func.clone();
        let pipelined_meta = self.kernel.meta.pipelined_meta.clone().unwrap();
        let num_scalars = pipelined_meta.num_scalars;
        let num_mut_scalars = pipelined_meta.num_mut_scalars;
        let divide_parts = pipelined_meta.divide_parts;
        let num_of_vars = self.kernel.meta.num_vars;
        let num_of_mut_vars = self.kernel.meta.num_mut_vars;
        /*
        args:
        assume there are n mut polys, m polys, p mut scalars, q scalrs to compute the fused kernel
        mut_var will have 4n + 2p elements, the first p are mut scalars, next n are the mut polys,
        then next p is gpu buffer for mut scalars,
        the next 3n are the gpu polys (triple buffer, one load, one compute, one store)
        var will have 3m + 2q elements
        the first q are scalars, next m are polys, next q are scalar buffers, the next 2m are the gpu polys (double buffer, one load, one compute)
         */
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>,
                              _: Arc<dyn Fn(i32) -> i32 + Send + Sync>|
              -> Result<(), RuntimeError> {
            // the first of mut_var is the buffer for the arguments
            let (tmp_buffers, mut_var) = mut_var.split_at_mut(2);
            let arg_buffer = tmp_buffers[0].unwrap_gpu_buffer();
            let local_buffer = tmp_buffers[1].unwrap_gpu_buffer();
            // check the buffer size
            assert!(
                arg_buffer.size
                    >= (2 * num_of_vars + 3 * num_of_mut_vars) * std::mem::size_of::<PolyPtr>()
            );

            assert!((mut_var.len() - 2 * num_mut_scalars) % 4 == 0);
            assert!((var.len() - 2 * num_scalars) % 3 == 0);

            let num_mut_poly = (mut_var.len() - 2 * num_mut_scalars) / 4;
            let num_mut_var = num_mut_poly + num_mut_scalars;
            assert_eq!(num_mut_var, num_of_mut_vars);
            let num_poly = (var.len() - 2 * num_scalars) / 3;
            let num_var = num_poly + num_scalars;
            assert_eq!(num_var, num_of_vars);

            // get the length
            let len = if num_mut_poly > 0 {
                mut_var[num_mut_scalars].unwrap_scalar_array().len()
            } else if num_poly > 0 {
                var[num_scalars].unwrap_scalar_array().len()
            } else {
                1
            };
            assert!(len % divide_parts == 0);

            // get streams
            let ref h2d_stream = CudaStream::new(0); // TODO: select the device
            let ref compute_stream = CudaStream::new(0); // TODO: select the device
            let ref d2h_stream = CudaStream::new(0); // TODO: select the device

            // get scalars
            let mut mut_scalars = Vec::new();
            for i in 0..num_mut_scalars {
                mut_scalars.push(mut_var[i].unwrap_scalar().clone());
            }
            let mut scalars = Vec::new();
            for i in 0..num_scalars {
                scalars.push(var[i].unwrap_scalar().clone());
            }

            // get scalar buffers
            let mut mut_gpu_scalars = Vec::new();
            for i in 0..num_mut_scalars {
                let buffer = mut_var[i + num_mut_var].unwrap_gpu_buffer();
                // check the buffer size
                assert_eq!(buffer.size, std::mem::size_of::<T::Field>());
                let gpu_scalar =
                    Scalar::new_gpu(buffer.ptr as *mut T::Field, buffer.device.unwrap_gpu());
                mut_gpu_scalars.push(gpu_scalar);
            }
            let mut gpu_scalars = Vec::new();
            for i in 0..num_scalars {
                let buffer = var[i + num_var].unwrap_gpu_buffer();
                // check the buffer size
                assert_eq!(buffer.size, std::mem::size_of::<T::Field>());
                let gpu_scalar =
                    Scalar::new_gpu(buffer.ptr as *mut T::Field, buffer.device.unwrap_gpu());
                gpu_scalars.push(gpu_scalar);
            }

            // transfer scalars to gpu
            for (host_scalar, gpu_scalar) in mut_scalars.iter().zip(mut_gpu_scalars.iter_mut()) {
                host_scalar.cpu2gpu(gpu_scalar, h2d_stream);
            }
            for (host_scalar, gpu_scalar) in scalars.iter().zip(gpu_scalars.iter_mut()) {
                host_scalar.cpu2gpu(gpu_scalar, h2d_stream);
            }

            // get polys
            let mut mut_polys = Vec::new();
            for i in 0..num_mut_poly {
                mut_polys.push(mut_var[i + num_mut_scalars].unwrap_scalar_array().clone());
                assert!(
                    mut_polys[i].slice_info.is_none(),
                    "pipelined fused kernel doesn't support slice"
                );
            }
            let mut polys = Vec::new();
            for i in 0..num_poly {
                polys.push(var[i + num_scalars].unwrap_scalar_array().clone());
                assert!(
                    polys[i].slice_info.is_none(),
                    "pipelined fused kernel doesn't support slice"
                );
            }

            let chunk_len = len / divide_parts;

            // get poly buffers
            let mut mut_gpu_polys = vec![Vec::new(); 3];
            let base_index = num_mut_var + num_mut_scalars;
            for i in 0..num_mut_poly {
                for j in 0..3 {
                    let buffer = mut_var[i + base_index + j * num_mut_poly].unwrap_gpu_buffer();
                    // check the buffer size
                    assert_eq!(buffer.size, chunk_len * std::mem::size_of::<T::Field>());
                    let gpu_poly = ScalarArray::new(
                        chunk_len,
                        buffer.ptr as *mut T::Field,
                        buffer.device.clone(),
                    );
                    mut_gpu_polys[j].push(gpu_poly);
                }
            }
            let mut gpu_polys = vec![Vec::new(); 2];
            let base_index = num_var + num_scalars;
            for i in 0..num_poly {
                for j in 0..2 {
                    let buffer = var[i + base_index + j * num_poly].unwrap_gpu_buffer();
                    // check the buffer size
                    assert_eq!(buffer.size, chunk_len * std::mem::size_of::<T::Field>());
                    let gpu_poly = ScalarArray::new(
                        chunk_len,
                        buffer.ptr as *mut T::Field,
                        buffer.device.clone(),
                    );
                    gpu_polys[j].push(gpu_poly);
                }
            }

            // prepare the args
            let mut mut_vars = [Vec::new(), Vec::new(), Vec::new()];
            for mut_buffer_id in 0..3 {
                for scalar in mut_gpu_scalars.iter() {
                    mut_vars[mut_buffer_id].push(PolyPtr {
                        ptr: scalar.value as *mut c_uint,
                        len: 1,
                        rotate: 0,
                        offset: 0,
                        whole_len: 1,
                    })
                }
                for poly in mut_gpu_polys[mut_buffer_id].iter_mut() {
                    mut_vars[mut_buffer_id].push(PolyPtr::from(poly))
                }
            }
            let mut vars = [Vec::new(), Vec::new()];
            for buffer_id in 0..2 {
                for scalar in gpu_scalars.iter() {
                    vars[buffer_id].push(ConstPolyPtr {
                        ptr: scalar.value as *mut c_uint,
                        len: 1,
                        rotate: 0,
                        offset: 0,
                        whole_len: 1,
                    })
                }
                for poly in gpu_polys[buffer_id].iter() {
                    vars[buffer_id].push(ConstPolyPtr::from(poly))
                }
            }

            // transfer args to gpu
            let mut d_vars = [null_mut(); 2];
            let mut d_mut_vars = [null_mut(); 3];
            unsafe {
                d_vars[0] = arg_buffer.ptr as *mut ConstPolyPtr;
                d_vars[1] = (d_vars[0] as *mut ConstPolyPtr).add(num_of_vars);
                d_mut_vars[0] = (d_vars[1] as *mut PolyPtr).add(num_of_vars);
                d_mut_vars[1] = (d_mut_vars[0] as *mut PolyPtr).add(num_of_mut_vars);
                d_mut_vars[2] = (d_mut_vars[1] as *mut PolyPtr).add(num_of_mut_vars);
            }

            for buffer_id in 0..2 {
                h2d_stream.memcpy_h2d(d_vars[buffer_id], vars[buffer_id].as_ptr(), num_of_vars);
            }
            for mut_buffer_id in 0..3 {
                h2d_stream.memcpy_h2d(
                    d_mut_vars[mut_buffer_id],
                    mut_vars[mut_buffer_id].as_ptr(),
                    num_of_mut_vars,
                );
            }

            // create events
            let mut_h2d_complete = [
                CudaEventRaw::new(0),
                CudaEventRaw::new(0),
                CudaEventRaw::new(0),
            ];
            let mut_compute_complete = [
                CudaEventRaw::new(0),
                CudaEventRaw::new(0),
                CudaEventRaw::new(0),
            ];
            let mut_d2h_complete = [
                CudaEventRaw::new(0),
                CudaEventRaw::new(0),
                CudaEventRaw::new(0),
            ];

            let h2d_complete = [CudaEventRaw::new(0), CudaEventRaw::new(0)];
            let compute_complete = [CudaEventRaw::new(0), CudaEventRaw::new(0)];

            let mut mut_buffer_id = 0;
            let mut buffer_id = 0;

            // start computing
            for chunk_id in 0..divide_parts {
                // load mutable data to gpu
                h2d_stream.wait_raw(&mut_d2h_complete[mut_buffer_id]);

                let compute_start = chunk_id * chunk_len;
                let compute_end = (chunk_id + 1) * chunk_len;

                // mut polys
                for i in 0..num_mut_poly {
                    let mut_poly = mut_polys[i].slice(compute_start, compute_end);
                    mut_poly.cpu2gpu(&mut mut_gpu_polys[mut_buffer_id][i], h2d_stream);
                }
                mut_h2d_complete[mut_buffer_id].record(h2d_stream);

                h2d_stream.wait_raw(&compute_complete[buffer_id]);
                // polys
                for i in 0..num_poly {
                    let poly = polys[i].slice(compute_start, compute_end);
                    poly.cpu2gpu(&mut gpu_polys[buffer_id][i], h2d_stream);
                }
                h2d_complete[buffer_id].record(h2d_stream);

                // wait for the previous transfer to finish
                compute_stream.wait_raw(&mut_h2d_complete[mut_buffer_id]);
                compute_stream.wait_raw(&h2d_complete[buffer_id]);

                // compute

                unsafe {
                    cuda_check!(cudaSetDevice(compute_stream.get_device()));
                    cuda_check!((c_func)(
                        d_vars[buffer_id],
                        d_mut_vars[mut_buffer_id],
                        chunk_len.try_into().unwrap(),
                        chunk_id == 0,
                        local_buffer.ptr as *mut c_void,
                        compute_stream.raw()
                    ));
                }

                compute_complete[buffer_id].record(compute_stream);
                mut_compute_complete[mut_buffer_id].record(compute_stream);

                // wait for the previous compute to finish
                d2h_stream.wait_raw(&mut_compute_complete[mut_buffer_id]);

                // transfer back mutable data
                for i in 0..num_mut_poly {
                    let mut mut_poly = mut_polys[i].slice(compute_start, compute_end);
                    mut_gpu_polys[mut_buffer_id][i].gpu2cpu(&mut mut_poly, d2h_stream);
                }
                mut_d2h_complete[mut_buffer_id].record(d2h_stream);

                mut_buffer_id = (mut_buffer_id + 1) % 3;
                buffer_id = (buffer_id + 1) % 2;
            }

            // transfer back scalars
            for (host_scalar, gpu_scalar) in mut_scalars.iter_mut().zip(mut_gpu_scalars.iter()) {
                gpu_scalar.gpu2cpu(host_scalar, d2h_stream);
            }

            h2d_stream.sync();
            compute_stream.sync();
            d2h_stream.sync();

            h2d_stream.destroy();
            compute_stream.destroy();
            d2h_stream.destroy();

            Ok(())
        };
        Function {
            meta: FuncMeta::new(
                self.kernel.meta.name.clone(),
                KernelType::FusedArith(self.kernel.meta.clone()),
            ),
            f: Arc::new(rust_func),
        }
    }
}
