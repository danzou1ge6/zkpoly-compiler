use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::thread;
use std::{
    any::type_name,
    marker::PhantomData,
    os::raw::{c_uint, c_ulonglong},
};

use libloading::Symbol;
use serde::de;
use zkpoly_common::{
    arith::{ArithGraph, FusedType},
    load_dynamic::Libs,
};
use zkpoly_cuda_api::stream::{CudaEventRaw, CudaStream};
use zkpoly_cuda_api::{
    bindings::{cudaError_t, cudaSetDevice, cudaStream_t},
    cuda_check,
};
use zkpoly_memory_pool::buddy_disk_pool::{gpu_read_from_disk, gpu_write_to_disk};
use zkpoly_runtime::functions::FusedKernelMeta;
use zkpoly_runtime::runtime::transfer::Transfer;
use zkpoly_runtime::scalar::{Scalar, ScalarArray};
use zkpoly_runtime::{
    args::{RuntimeType, Variable},
    error::RuntimeError,
    functions::{FuncMeta, Function, KernelType, RegisteredFunction},
};

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
        if !libs.contains(LIB_NAME) {
            let field_type = resolve_type(type_name::<T::Field>());
            xmake_config(FIELD_NAME, field_type);
            xmake_run("fused_kernels");
        }
        let lib = libs.load(LIB_NAME);
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
        let device_id = self.kernel.meta.device.unwrap_gpu();
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
                              gpu_mapping: Arc<dyn Fn(i32) -> i32 + Send + Sync>|
              -> Result<(), RuntimeError> {
            // the first of mut_var is the buffer for the arguments
            let (tmp_buffers, mut_var) = mut_var.split_at_mut(2);
            let arg_buffer = tmp_buffers[0].unwrap_gpu_buffer(); // buffer for pointers
            let local_buffer = tmp_buffers[1].unwrap_gpu_buffer(); // buffer for local spills
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

            let device_id = gpu_mapping(device_id);

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
            let h2d_stream = CudaStream::new(device_id);
            for (host_scalar, gpu_scalar) in mut_scalars.iter().zip(mut_gpu_scalars.iter_mut()) {
                match host_scalar.device {
                    zkpoly_common::devices::DeviceType::CPU => {
                        host_scalar.cpu2gpu(gpu_scalar, &h2d_stream);
                    }
                    zkpoly_common::devices::DeviceType::GPU { .. } => {
                        host_scalar.gpu2gpu(gpu_scalar, &h2d_stream);
                    }
                    _ => unreachable!("Only CPU and GPU scalars are supported"),
                }
            }
            for (host_scalar, gpu_scalar) in scalars.iter().zip(gpu_scalars.iter_mut()) {
                match host_scalar.device {
                    zkpoly_common::devices::DeviceType::CPU => {
                        host_scalar.cpu2gpu(gpu_scalar, &h2d_stream);
                    }
                    zkpoly_common::devices::DeviceType::GPU { .. } => {
                        host_scalar.gpu2gpu(gpu_scalar, &h2d_stream);
                    }
                    _ => unreachable!("Only CPU and GPU scalars are supported"),
                }
            }

            let mut chunks = 1; // if there is no disk, we can use one chunk

            fn update_chunk_num(disks: usize, chunks: &mut usize) {
                if disks > 0 {
                    if *chunks == 1 {
                        *chunks = disks;
                    } else {
                        assert_eq!(disks, *chunks);
                    }
                }
            }

            // the following variables are used to track the number of mutable and immutable polys on GPU, CPU and disk
            let mut num_mut_poly_gpu = 0;
            let mut num_mut_poly_cpu = 0;
            let mut num_mut_poly_disk = 0;
            let mut num_poly_gpu = 0;
            let mut num_poly_cpu = 0;
            let mut num_poly_disk = 0;

            // get polys
            let mut mut_polys = Vec::new();
            for i in 0..num_mut_poly {
                mut_polys.push(mut_var[i + num_mut_scalars].unwrap_scalar_array().clone());
                assert!(
                    mut_polys[i].slice_info.is_none(),
                    "pipelined fused kernel doesn't support slice"
                );
                match mut_polys[i].device {
                    zkpoly_common::devices::DeviceType::CPU => num_mut_poly_cpu += 1,
                    zkpoly_common::devices::DeviceType::GPU { device_id } => num_mut_poly_gpu += 1,
                    zkpoly_common::devices::DeviceType::Disk => {
                        num_mut_poly_disk += 1;
                        update_chunk_num(mut_polys[i].disk_pos.len(), &mut chunks);
                    }
                }
            }
            let mut polys = Vec::new();
            for i in 0..num_poly {
                polys.push(var[i + num_scalars].unwrap_scalar_array().clone());
                assert!(
                    polys[i].slice_info.is_none(),
                    "pipelined fused kernel doesn't support slice"
                );
                match polys[i].device {
                    zkpoly_common::devices::DeviceType::CPU => num_poly_cpu += 1,
                    zkpoly_common::devices::DeviceType::GPU { device_id } => num_poly_gpu += 1,
                    zkpoly_common::devices::DeviceType::Disk => {
                        num_poly_disk += 1;
                        update_chunk_num(polys[i].disk_pos.len(), &mut chunks);
                    }
                }
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
            #[derive(Clone)]
            struct DevicePtrs {
                d_vars: [*mut ConstPolyPtr; 2],
                d_mut_vars: [*mut PolyPtr; 3],
            }
            unsafe impl Send for DevicePtrs {}

            let mut d_ptrs = DevicePtrs {
                d_vars: [null_mut(), null_mut()],
                d_mut_vars: [null_mut(), null_mut(), null_mut()],
            };
            unsafe {
                d_ptrs.d_vars[0] = arg_buffer.ptr as *mut ConstPolyPtr;
                d_ptrs.d_vars[1] = (d_ptrs.d_vars[0] as *mut ConstPolyPtr).add(num_of_vars);
                d_ptrs.d_mut_vars[0] = (d_ptrs.d_vars[1] as *mut PolyPtr).add(num_of_vars);
                d_ptrs.d_mut_vars[1] = (d_ptrs.d_mut_vars[0] as *mut PolyPtr).add(num_of_mut_vars);
                d_ptrs.d_mut_vars[2] = (d_ptrs.d_mut_vars[1] as *mut PolyPtr).add(num_of_mut_vars);
            }

            for buffer_id in 0..2 {
                h2d_stream.memcpy_h2d(
                    d_ptrs.d_vars[buffer_id],
                    vars[buffer_id].as_ptr(),
                    num_of_vars,
                );
            }
            for mut_buffer_id in 0..3 {
                h2d_stream.memcpy_h2d(
                    d_ptrs.d_mut_vars[mut_buffer_id],
                    mut_vars[mut_buffer_id].as_ptr(),
                    num_of_mut_vars,
                );
            }

            h2d_stream.sync();
            h2d_stream.destroy();

            #[derive(Debug, Clone)]
            struct TransferInfo<T: RuntimeType> {
                src: ScalarArray<T::Field>,
                dst: ScalarArray<T::Field>,
                offset: usize, // offset from the users's view, should divide chunks to get the true offset
                /// number of chunks to transfer, equal to the number of disks (use all disks)
                /// [0, 31] placed on two disks, then [0, 15] on the first disk, [16, 31] on the second disk
                /// assume we use 8 as a size for GPU computation, then we need to transfer [0, 3] and [16, 19]
                /// first to fully utilize the disks
                /// and we have to do the same thing even when data is on cpu to maintain consistency
                chunks: usize,
                len: usize,
            }

            // create channels
            let (cpu2gpu_sender, cpu2gpu_receiver) = channel::<TransferInfo<T>>(); // for transferring data from CPU to GPU
            let (disk2gpu_sender, disk2gpu_receiver) = channel::<TransferInfo<T>>(); // for transferring data from disk to GPU
            let (gpu2cpu_sender, gpu2cpu_receiver) = channel::<TransferInfo<T>>(); // for transferring data from GPU to CPU
            let (gpu2disk_sender, gpu2disk_receiver) = channel::<TransferInfo<T>>(); // for transferring data from GPU to disk
            let (gpu_poly2slice_sender, gpu_poly2slice_receiver) = channel::<TransferInfo<T>>(); // for transferring data from GPU to slice
            let (gpu_slice2poly_sender, gpu_slice2poly_receiver) = channel::<TransferInfo<T>>(); // for transferring data from slice to GPU
            let (cpu2gpu_ok_sender, cpu2gpu_ok_receiver) = channel::<()>(); // for confirming the data transfer from CPU to GPU
            let (disk2gpu_ok_sender, disk2gpu_ok_receiver) = channel::<()>(); // for confirming the data transfer from disk to GPU
            let (gpu2cpu_ok_sender, gpu2cpu_ok_receiver) = channel::<()>(); // for confirming the data transfer from GPU to CPU
            let (gpu2disk_ok_sender, gpu2disk_ok_receiver) = channel::<()>(); // for confirming the data transfer from GPU to disk
            let (gpu_poly2slice_ok_sender, gpu_poly2slice_ok_receiver) = channel::<()>();
            let (gpu_slice2poly_ok_sender, gpu_slice2poly_ok_receiver) = channel::<()>();
            let (compute_ok_sender, compute_ok_receiver) = channel::<()>();

            // the thread for cpu to gpu transfer
            thread::spawn({
                move || {
                    let h2d_stream = CudaStream::new(device_id);
                    while let Ok(info) = cpu2gpu_receiver.recv() {
                        // transfer data from CPU to GPU
                        assert!(info.src.len() % info.chunks == 0);
                        let part_len = info.src.len() / info.chunks;
                        let part_offset = info.offset / info.chunks;
                        let part_transfer_len = info.len / info.chunks;
                        println!(
                            "cpu2gpu: part_len = {}, part_offset = {}, part_transfer_len = {}, chunks = {}",
                            part_len, part_offset, part_transfer_len, info.chunks
                        );
                        for i in 0..info.chunks {
                            let src_offset = i * part_len + part_offset;
                            let dst_offset = part_transfer_len * i;
                            let src_sliced =
                                info.src.slice(src_offset, src_offset + part_transfer_len);
                            println!("i = {i} cpu2gpu from [{}, {}] to [{}, {}]", src_offset, src_offset + part_transfer_len, dst_offset, dst_offset + part_transfer_len);
                            let mut dst_sliced =
                                info.dst.slice(dst_offset, dst_offset + part_transfer_len);
                            src_sliced.cpu2gpu(&mut dst_sliced, &h2d_stream);
                        }
                        h2d_stream.sync();
                        // confirm the transfer
                        cpu2gpu_ok_sender.send(()).unwrap();
                        println!("cpu2gpu: transfer ok sent");
                    }
                    h2d_stream.destroy();
                }
            });

            // the thread for gpu to cpu transfer
            thread::spawn({
                move || {
                    let d2h_stream = CudaStream::new(device_id);
                    while let Ok(info) = gpu2cpu_receiver.recv() {
                        // transfer data from GPU to CPU
                        assert!(info.src.len() % info.chunks == 0);
                        let part_len = info.dst.len() / info.chunks;
                        let part_offset = info.offset / info.chunks;
                        let part_transfer_len = info.len / info.chunks;
                        for i in 0..info.chunks {
                            let src_offset = part_transfer_len * i;
                            let dst_offset = i * part_len + part_offset;
                            let src_sliced =
                                info.src.slice(src_offset, src_offset + part_transfer_len);
                            println!("i = {i} gpu2cpu: [{}, {}] to [{}, {}]", src_offset, src_offset + part_transfer_len, dst_offset, dst_offset + part_transfer_len);
                            let mut dst_sliced =
                                info.dst.slice(dst_offset, dst_offset + part_transfer_len);
                            src_sliced.gpu2cpu(&mut dst_sliced, &d2h_stream);
                        }
                        d2h_stream.sync();
                        // confirm the transfer
                        println!("sending gpu2cpu ok");
                        gpu2cpu_ok_sender.send(()).unwrap();
                    }
                    d2h_stream.destroy();
                }
            });

            // the thread for disk to gpu transfer
            thread::spawn(move || {
                while let Ok(info) = disk2gpu_receiver.recv() {
                    // transfer data from disk to GPU
                    assert!(info.src.len() % info.chunks == 0);
                    gpu_read_from_disk(
                        info.dst.values.cast(),
                        &info.src.disk_pos,
                        info.src.len * size_of::<T::Field>(),
                        device_id,
                        info.len * size_of::<T::Field>(),
                        info.offset * size_of::<T::Field>(),
                    );
                    // ok
                    disk2gpu_ok_sender.send(()).unwrap();
                }
            });

            // the thread for gpu to disk transfer
            thread::spawn(move || {
                while let Ok(info) = gpu2disk_receiver.recv() {
                    // transfer data from GPU to disk
                    assert!(info.dst.len() % info.chunks == 0);
                    gpu_write_to_disk(
                        info.src.values.cast(),
                        &info.dst.disk_pos,
                        info.dst.len * size_of::<T::Field>(),
                        device_id,
                        info.len * size_of::<T::Field>(),
                        info.offset * size_of::<T::Field>(),
                    );
                    // ok
                    gpu2disk_ok_sender.send(()).unwrap();
                }
            });

            // the thread for gpu poly to slice transfer
            thread::spawn(move || {
                let stream = CudaStream::new(device_id);
                while let Ok(info) = gpu_poly2slice_receiver.recv() {
                    // transfer data from GPU poly to slice
                    assert!(info.src.len() % info.chunks == 0);
                    let part_len = info.src.len() / info.chunks;
                    let part_offset = info.offset / info.chunks;
                    let part_transfer_len = info.len / info.chunks;
                    for i in 0..info.chunks {
                        let src_offset = i * part_len + part_offset;
                        let dst_offset = part_transfer_len * i;
                        let src_sliced = info.src.slice(src_offset, src_offset + part_transfer_len);
                        let mut dst_sliced =
                            info.dst.slice(dst_offset, dst_offset + part_transfer_len);
                        src_sliced.gpu2gpu(&mut dst_sliced, &stream);
                    }
                    stream.sync();
                    // confirm the transfer
                    gpu_poly2slice_ok_sender.send(()).unwrap();
                }
                stream.destroy();
            });

            // the thread for gpu slice to poly transfer
            thread::spawn(move || {
                let stream = CudaStream::new(device_id);
                while let Ok(info) = gpu_slice2poly_receiver.recv() {
                    // transfer data from GPU slice to poly
                    assert!(info.src.len() % info.chunks == 0);
                    let part_len = info.dst.len() / info.chunks;
                    let part_offset = info.offset / info.chunks;
                    let part_transfer_len = info.len / info.chunks;
                    for i in 0..info.chunks {
                        let src_offset = part_transfer_len * i;
                        let dst_offset = i * part_len + part_offset;
                        let src_sliced = info.src.slice(src_offset, src_offset + part_transfer_len);
                        let mut dst_sliced =
                            info.dst.slice(dst_offset, dst_offset + part_transfer_len);
                        src_sliced.gpu2gpu(&mut dst_sliced, &stream);
                    }
                    stream.sync();
                    // confirm the transfer
                    gpu_slice2poly_ok_sender.send(()).unwrap();
                }
                stream.destroy();
            });

            fn wait_ok(receiver: &std::sync::mpsc::Receiver<()>, num: usize) {
                println!("waiting for {} ok", num);
                for i in 0..num {
                    receiver.recv().unwrap();
                    println!("received ok {}/{}", i + 1, num);
                }
            }

            let mut_polys_clone = mut_polys.clone();

            let c_func_clone = c_func.clone();
            // the thread for compute
            thread::spawn({
                struct SafePtr(*mut c_void);
                unsafe impl Send for SafePtr {}

                let local_buffer_ptr = SafePtr(local_buffer.ptr as *mut c_void);
                let d_ptrs = d_ptrs.clone();
                let mut_gpu_polys = mut_gpu_polys.clone();
                let gpu_slice2poly_sender = gpu_slice2poly_sender.clone();

                move || {
                    let mut mut_buffer_id = 0;
                    let mut buffer_id = 0;
                    let compute_stream = CudaStream::new(device_id);

                    // start computing
                    for chunk_id in 0..divide_parts {
                        println!("compute chunk {}", chunk_id);
                        // wait for the signal to start compute
                        println!("waiting cpu2gpu mut");
                        wait_ok(&cpu2gpu_ok_receiver, num_mut_poly_cpu);
                        println!("waiting disk2gpu mut");
                        wait_ok(&disk2gpu_ok_receiver, num_mut_poly_disk);
                        println!("waiting gpu_poly2slice mut");
                        wait_ok(&gpu_poly2slice_ok_receiver, num_mut_poly_gpu);
                        println!("waiting cpu2gpu poly");
                        wait_ok(&cpu2gpu_ok_receiver, num_poly_cpu);
                        println!("waiting disk2gpu poly");
                        wait_ok(&disk2gpu_ok_receiver, num_poly_disk);
                        println!("waiting gpu_poly2slice poly");
                        wait_ok(&gpu_poly2slice_ok_receiver, num_poly_gpu);

                        println!("start compute chunk {}", chunk_id);
                        // compute
                        let _ = &local_buffer_ptr;
                        let _ = &d_ptrs;
                        unsafe {
                            cuda_check!(cudaSetDevice(compute_stream.get_device()));
                            cuda_check!((c_func_clone)(
                                d_ptrs.d_vars[buffer_id],
                                d_ptrs.d_mut_vars[mut_buffer_id],
                                chunk_len.try_into().unwrap(),
                                chunk_id == 0,
                                local_buffer_ptr.0,
                                compute_stream.raw()
                            ));
                        }
                        compute_stream.sync();
                        compute_ok_sender.send(()).unwrap();

                        println!("finish compute chunk {}", chunk_id);

                        println!("num to transfer back: {}", num_mut_poly);

                        // trigger the transfer back
                        for i in 0..num_mut_poly {
                            let transfer_info = TransferInfo {
                                dst: mut_polys_clone[i].clone(),
                                src: mut_gpu_polys[mut_buffer_id][i].clone(),
                                chunks: chunks,
                                offset: chunk_id * chunk_len,
                                len: chunk_len,
                            };

                            match mut_polys_clone[i].device {
                                zkpoly_common::devices::DeviceType::CPU => {
                                    gpu2cpu_sender.send(transfer_info).unwrap();
                                }
                                zkpoly_common::devices::DeviceType::GPU { .. } => {
                                    gpu_slice2poly_sender.send(transfer_info).unwrap();
                                }
                                zkpoly_common::devices::DeviceType::Disk => {
                                    gpu2disk_sender.send(transfer_info).unwrap();
                                }
                            }
                        }

                        mut_buffer_id = (mut_buffer_id + 1) % 3;
                        buffer_id = (buffer_id + 1) % 2;
                    }
                    compute_stream.destroy();
                }
            });

            // start computing
            for chunk_id in 0..divide_parts {
                let mut_buffer_id = chunk_id % 3;
                let buffer_id = chunk_id % 2;
                println!(
                    "transfer chunk {}: mut_buffer_id = {}, buffer_id = {}",
                    chunk_id, mut_buffer_id, buffer_id
                );

                if chunk_id >= 2 {
                    // start to reuse the buffer, so we need to wait for the previous compute to finish
                    println!("wait for compute to finish");
                    compute_ok_receiver.recv().unwrap();
                    println!("received compute ok for chunk {}", chunk_id);
                }

                if chunk_id >= 3 {
                    println!("wait for transfer back to finish");
                    // start to reuse the buffer, so we need to wait for the previous transfer back to finish
                    for i in 0..num_mut_poly {
                        match mut_polys[i].device {
                            zkpoly_common::devices::DeviceType::CPU => {
                                gpu2cpu_ok_receiver.recv().unwrap();
                            }
                            zkpoly_common::devices::DeviceType::GPU { .. } => {
                                gpu_slice2poly_ok_receiver.recv().unwrap();
                            }
                            zkpoly_common::devices::DeviceType::Disk => {
                                gpu2disk_ok_receiver.recv().unwrap();
                            }
                        }
                    }
                }

                println!(
                    "transfer chunk {}: mut_buffer_id = {}, buffer_id = {}",
                    chunk_id, mut_buffer_id, buffer_id
                );
                // now trigger the transfer to GPU
                for i in 0..num_mut_poly {
                    let transfer_info = TransferInfo {
                        src: mut_polys[i].clone(),
                        dst: mut_gpu_polys[mut_buffer_id][i].clone(),
                        chunks: chunks,
                        offset: chunk_id * chunk_len,
                        len: chunk_len,
                    };
                    match mut_polys[i].device {
                        zkpoly_common::devices::DeviceType::CPU => {
                            cpu2gpu_sender.send(transfer_info).unwrap();
                        }
                        zkpoly_common::devices::DeviceType::GPU { .. } => {
                            gpu_poly2slice_sender.send(transfer_info).unwrap();
                        }
                        zkpoly_common::devices::DeviceType::Disk => {
                            disk2gpu_sender.send(transfer_info).unwrap();
                        }
                    }
                }

                println!(
                    "transfer chunk {}: mut_buffer_id = {}, buffer_id = {}",
                    chunk_id, mut_buffer_id, buffer_id
                );
                // now trigger the transfer to GPU for polys
                for i in 0..num_poly {
                    let transfer_info = TransferInfo {
                        src: polys[i].clone(),
                        dst: gpu_polys[buffer_id][i].clone(),
                        chunks: chunks,
                        offset: chunk_id * chunk_len,
                        len: chunk_len,
                    };
                    match polys[i].device {
                        zkpoly_common::devices::DeviceType::CPU => {
                            cpu2gpu_sender.send(transfer_info).unwrap();
                        }
                        zkpoly_common::devices::DeviceType::GPU { .. } => {
                            gpu_poly2slice_sender.send(transfer_info).unwrap();
                        }
                        zkpoly_common::devices::DeviceType::Disk => {
                            disk2gpu_sender.send(transfer_info).unwrap();
                        }
                    }
                }
            }

            println!("waiting for compute to finish");
            compute_ok_receiver.recv().unwrap();
            println!("waiting for compute to finish");
            compute_ok_receiver.recv().unwrap();

            for _ in 0..3 {
                println!("waiting for transfer back to finish");
                for i in 0..num_mut_poly {
                    match mut_polys[i].device {
                        zkpoly_common::devices::DeviceType::CPU => {
                            gpu2cpu_ok_receiver.recv().unwrap();
                        }
                        zkpoly_common::devices::DeviceType::GPU { .. } => {
                            gpu_slice2poly_ok_receiver.recv().unwrap();
                        }
                        zkpoly_common::devices::DeviceType::Disk => {
                            gpu2disk_ok_receiver.recv().unwrap();
                        }
                    }
                    println!("got tranfer back")
                }
            }

            let d2h_stream = CudaStream::new(device_id);
            // transfer back scalars
            for (host_scalar, gpu_scalar) in mut_scalars.iter_mut().zip(mut_gpu_scalars.iter()) {
                match host_scalar.device {
                    zkpoly_common::devices::DeviceType::CPU => {
                        gpu_scalar.gpu2cpu(host_scalar, &d2h_stream);
                    }
                    zkpoly_common::devices::DeviceType::GPU { .. } => {
                        gpu_scalar.gpu2gpu(host_scalar, &d2h_stream);
                    }
                    _ => unreachable!("Only CPU and GPU scalars are supported"),
                }
            }

            d2h_stream.sync();
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
