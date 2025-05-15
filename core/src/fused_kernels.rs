use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::ptr::null_mut;
use std::{
    any::type_name,
    borrow::Borrow,
    fs,
    marker::PhantomData,
    os::raw::{c_uint, c_ulonglong},
};

use libloading::Symbol;
use zkpoly_common::arith::Mutability;
use zkpoly_common::{
    arith::{Arith, ArithBinOp, ArithGraph, ArithUnrOp, BinOp, FusedType, Operation, SpOp, UnrOp},
    get_project_root::get_project_root,
    heap::UsizeId,
    load_dynamic::Libs,
    typ::PolyType,
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
    functions::{FuncMeta, Function, FunctionValue, KernelType, RegisteredFunction},
};

use crate::{
    build_func::{resolve_type, xmake_config, xmake_run},
    poly_ptr::{ConstPolyPtr, PolyPtr},
};

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
    limbs: usize,
    schedule: Vec<InnerId>,
    regs: usize,
    partition: Vec<usize>,
    inputs: Vec<BTreeSet<InnerId>>, // inputs of each partitioned sub function
    outputs: Vec<BTreeSet<InnerId>>, // outputs of each partitioned sub function
}

const TMP_PREFIX: &str = "tmp";
const SUB_FUNC_NAME: &str = "part";
const HEADER_SUFFIX: &str = "_header.cuh";
const TABS: &str = "    ";
const WRAPPER_SUFFIX: &str = "_wrapper.cu";
const FIELD_NAME: &str = "FUSED_FIELD";

impl<OuterId: UsizeId, InnerId: UsizeId + 'static> FusedOp<OuterId, InnerId> {
    pub fn num_scalars(&self) -> (usize, usize) {
        let mut num_scalars = 0;
        let mut num_mut_scalars = 0;
        for (typ, _) in self.vars.iter() {
            if *typ == FusedType::Scalar {
                num_scalars += 1;
            }
        }
        for (typ, _) in self.mut_vars.iter() {
            if *typ == FusedType::Scalar {
                num_mut_scalars += 1;
            }
        }
        (num_scalars, num_mut_scalars)
    }

    pub fn new(graph: ArithGraph<OuterId, InnerId>, name: String, limbs: usize) -> Self {
        let (vars, mut_vars) = graph.gen_var_lists();
        let (schedule, live_ts, regs) = graph.schedule();
        let partition = graph.partition(&schedule, 1024); // magic number for compile time

        // now we have to generate the inputs and outputs for each partition
        let mut inputs = vec![BTreeSet::new(); partition.len()];
        let mut outputs = vec![BTreeSet::new(); partition.len()];

        let mut past_vars = BTreeSet::new();
        for i in 0..(partition.len() - 1) {
            let start = partition[i];
            let end = partition[i + 1];
            let mut cur_vars = BTreeSet::new();
            for j in start..end {
                let cur_node = schedule[j];
                let vertex = graph.g.vertex(cur_node);
                cur_vars.insert(cur_node);
                match &vertex.op {
                    Operation::Arith(arith) => {
                        for depend in arith.uses() {
                            if past_vars.contains(&depend) {
                                // this is an input
                                inputs[i].insert(depend);
                            }
                        }
                        if live_ts[cur_node.clone().into()] as usize >= end {
                            // this is an output
                            outputs[i].insert(cur_node);
                        }
                    }
                    Operation::Output { store_node, .. } => {
                        if past_vars.contains(store_node) {
                            // this is an input
                            inputs[i].insert(*store_node);
                        }
                    }
                    Operation::Input { .. } => {
                        if live_ts[cur_node.clone().into()] as usize >= end {
                            // this is an output
                            outputs[i].insert(cur_node);
                        }
                    }
                    Operation::Todo => unreachable!("todo should be handled earlier"),
                }
            }
            past_vars.extend(cur_vars);
        }

        Self {
            graph,
            name,
            vars,
            mut_vars,
            limbs,
            schedule,
            regs,
            partition,
            inputs,
            outputs,
        }
    }

    fn compare_or_write(&self, path: &str, s: &str) {
        if let Ok(mut f) = fs::File::open(path) {
            let mut old = String::new();
            f.read_to_string(&mut old).unwrap();
            if s == old {
                return;
            }
        }

        let mut f = fs::File::create(path).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    pub fn gen(&self, head_annotation: impl Borrow<str>) {
        let header = format!("{}\n{}", head_annotation.borrow(), self.gen_header());
        let kernels = self.gen_kernel();
        let wrapper = self.gen_wrapper();

        let project_root = get_project_root();
        let base_path = project_root + "/core/src/fused_kernels/src/" + self.name.as_str();

        // header file
        let header_path = base_path.clone() + HEADER_SUFFIX;
        self.compare_or_write(&header_path, &header);

        // wrapper file
        let wrapper_path = base_path.clone() + WRAPPER_SUFFIX;
        self.compare_or_write(&wrapper_path, &wrapper);

        for (id, kernel) in kernels.iter().enumerate() {
            let kernel_path = base_path.clone() + &format!("_{SUB_FUNC_NAME}{id}.cu");
            self.compare_or_write(&kernel_path, kernel);
        }
    }

    fn gen_device_func_args(&self, id: usize, typename: &str) -> String {
        let mut func_sig = String::new();
        func_sig.push_str("(\n");
        func_sig.push_str(&format!(
            "{TABS}ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long idx, bool is_first",
        ));

        // inputs
        for input_id in self.inputs[id].iter() {
            let input_id: usize = input_id.clone().into();
            func_sig += &format!(",\n{TABS}const {typename} {TMP_PREFIX}{input_id}");
        }

        // outputs
        for output_id in self.outputs[id].iter() {
            let output_id: usize = output_id.clone().into();
            func_sig += &format!(",\n{TABS}{typename} &{TMP_PREFIX}{output_id}");
        }
        func_sig.push_str("\n)");
        func_sig
    }

    fn gen_header(&self) -> String {
        let mut header = String::new();
        header.push_str("#pragma once\n");
        header.push_str("#include \"../../common/mont/src/field_impls.cuh\"\n");
        header.push_str("#include \"../../common/iter/src/iter.cuh\"\n");
        header.push_str("#include \"../../common/error/src/check.cuh\"\n");
        header.push_str("#include <cuda_runtime.h>\n");
        header.push_str("using iter::SliceIterator;\n");
        header.push_str("using mont::u32;\n");
        header.push_str("using iter::make_slice_iter;\n");

        header.push_str("namespace detail {\n");

        // generate the kernel signatures
        for id in 0..(self.partition.len() - 1) {
            header.push_str(&format!(
                "template <typename Field>\n__device__ void {}_{SUB_FUNC_NAME}{id}",
                self.name
            ));
            header.push_str(&self.gen_device_func_args(id, "Field"));
            header.push_str(";\n\n");
        }
        header.push_str("}\n");

        header
    }

    fn gen_wrapper(&self) -> String {
        let mut wrapper = String::new();

        let u32_regs = self.regs * self.limbs;

        let launch_bounds = 256;
        // if u32_regs < 128 {
        //     1024
        // } else if u32_regs < 256 {
        //     512
        // } else {
        //     256
        // };

        wrapper += &format!("#include \"{}{HEADER_SUFFIX}\"\n\n", self.name);

        wrapper.push_str("namespace detail {\n");

        // generate kernel signature
        wrapper.push_str("template <typename Field>\n");
        wrapper += &format!("__launch_bounds__({launch_bounds}) __global__ void ");
        wrapper.push_str(&self.name);
        wrapper.push_str(
            "(ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long len, bool is_first) {\n",
        );

        wrapper +=
            &format!("{TABS}unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
        wrapper += &format!("{TABS}if (idx >= len) return;\n");

        // call the partitioned kernels
        for id in 0..(self.partition.len() - 1) {
            // create output variables
            for (output_rank, output_id) in self.outputs[id].iter().enumerate() {
                let output_id: usize = output_id.clone().into();
                if output_rank == 0 {
                    wrapper += &format!("{TABS}Field ");
                }
                if output_rank + 1 == self.outputs[id].len() {
                    wrapper += &format!("{TMP_PREFIX}{output_id};\n");
                } else {
                    wrapper += &format!("{TMP_PREFIX}{output_id}, ");
                }
            }

            wrapper += &format!(
                "{TABS}{}_{SUB_FUNC_NAME}{id}<Field>(\n{TABS}{TABS}vars, mut_vars, idx, is_first",
                self.name
            );
            // inputs
            for input_id in self.inputs[id].iter() {
                let input_id: usize = input_id.clone().into();
                wrapper += &format!(",\n{TABS}{TABS}{TMP_PREFIX}{input_id}");
            }
            // outputs
            for output_id in self.outputs[id].iter() {
                let output_id: usize = output_id.clone().into();
                wrapper += &format!(",\n{TABS}{TABS}{TMP_PREFIX}{output_id}");
            }
            wrapper.push_str(&format!("\n{TABS});\n"));
        }
        wrapper.push_str("}\n");
        wrapper.push_str("}\n");

        // outer C signature
        wrapper.push_str("extern \"C\" cudaError_t ");
        wrapper.push_str(&self.name);
        wrapper.push_str(
            "(ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long len, bool is_first, cudaStream_t stream) {\n",
        );

        wrapper.push_str("uint block_size = 256;\n");
        wrapper.push_str("uint grid_size = (len + block_size - 1) / block_size;\n");
        wrapper.push_str("detail::");
        wrapper.push_str(&self.name);
        wrapper.push_str(&format!("<{FIELD_NAME}>"));
        wrapper.push_str(
            " <<< grid_size, block_size, 0, stream >>> (vars, mut_vars, len, is_first);\n",
        );

        wrapper.push_str("return cudaGetLastError();\n");
        wrapper.push_str("}\n");
        wrapper
    }

    fn gen_kernel(&self) -> Vec<String> {
        let mut kernels = Vec::new();

        // prepare the mapping from the input/output to the kernel arguments
        let var_mapping = self
            .vars
            .iter()
            .enumerate()
            .map(|(i, (_, id))| {
                let id: usize = id.clone().into();
                (id, i)
            })
            .collect::<BTreeMap<_, _>>();
        let mut_var_mapping = {
            let mut x = self
                .mut_vars
                .iter()
                .enumerate()
                .map(|(i, (_, id))| {
                    let id: usize = id.clone().into();
                    (id, i)
                })
                .collect::<BTreeMap<_, _>>();

            self.graph.outputs.iter().for_each(|oi| {
                let (_, _, _, in_nodes) = self.graph.g.vertex(*oi).op.unwrap_output();
                for in_node in in_nodes {
                    if self.graph.g.vertex(*in_node).op.unwrap_input_mutability() == Mutability::Mut
                    {
                        let in_node_usize: usize = in_node.clone().into();
                        let oi_usize: usize = oi.clone().into();
                        x.insert(in_node_usize, x[&oi_usize]);
                    }
                }
            });

            x
        };

        for id in 0..(self.partition.len() - 1) {
            let mut kernel = String::new();
            kernel.push_str(&format!("#include \"{}{HEADER_SUFFIX}\"\n", self.name));
            kernel.push_str("namespace detail {\n");

            // generate the kernel signature
            kernel.push_str(&format!(
                "template <typename Field>\n__device__ void {}_{SUB_FUNC_NAME}{id}",
                self.name
            ));
            kernel.push_str(&self.gen_device_func_args(id, "Field"));
            kernel.push_str(" {\n");

            let start = self.partition[id];
            let end = self.partition[id + 1];

            for step in start..end {
                let head = self.schedule[step];
                let vertex = self.graph.g.vertex(head.clone());
                let store_target = if self.outputs[id].contains(&head) {
                    format!("{TMP_PREFIX}{}", head.clone().into())
                } else {
                    format!("auto {TMP_PREFIX}{}", head.clone().into())
                };
                match &vertex.op {
                    Operation::Output {
                        typ, store_node, ..
                    } => {
                        let src: usize = store_node.clone().into();
                        let id: usize = head.clone().into(); // self.outputs_i2o[&head].clone().into();
                        match typ {
                            FusedType::Scalar => {
                                kernel += &format!("*reinterpret_cast<Field*>(mut_vars[{}].ptr) = {TMP_PREFIX}{src};\n", mut_var_mapping[&id]);
                            }
                            FusedType::ScalarArray => {
                                kernel += &format!("make_slice_iter<Field>(mut_vars[{}])[idx] = {TMP_PREFIX}{src};\n", mut_var_mapping[&id]);
                            }
                        }
                    }
                    Operation::Input { typ, .. } => {
                        let id = head.clone().into(); // outer_id.clone().into();
                        let (map_id, mutability) = if var_mapping.contains_key(&id) {
                            (var_mapping[&id], "vars")
                        } else {
                            (mut_var_mapping[&id], "mut_vars")
                        };
                        match typ {
                            FusedType::Scalar => {
                                kernel += &format!("{store_target} = *reinterpret_cast<const Field*>({}[{}].ptr);\n", mutability, map_id);
                            }
                            FusedType::ScalarArray => {
                                kernel += &format!(
                                    "{store_target} = make_slice_iter<Field>({}[{}])[idx];\n",
                                    mutability, map_id
                                );
                            }
                        }
                    }
                    Operation::Arith(arith) => match arith {
                        Arith::Bin(op, lhs, rhs) => {
                            let lhs: usize = lhs.clone().into();
                            let rhs: usize = rhs.clone().into();
                            match op {
                                BinOp::Pp(op) => match op {
                                    ArithBinOp::Add => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Sub => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Mul => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} * {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Div => {
                                        unreachable!("division should be handled in batched invert")
                                    }
                                },
                                BinOp::Ss(op) => match op {
                                    ArithBinOp::Add => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Sub => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Mul => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} * {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    ArithBinOp::Div => {
                                        unreachable!("division should be handled in scalar invert")
                                    }
                                },
                                BinOp::Sp(op) => match op {
                                    SpOp::Add => {
                                        let stmt = match self.graph.poly_repr {
                                            PolyType::Coef => {
                                                format!("{store_target} = (idx == 0 && is_first) ? {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs};\n")
                                            }
                                            PolyType::Lagrange => {
                                                format!("{store_target} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n")
                                            }
                                        };
                                        kernel += &stmt;
                                    }
                                    SpOp::Sub => {
                                        let stmt = match self.graph.poly_repr {
                                            PolyType::Coef => {
                                                format!("{store_target} = (idx == 0 && is_first) ? {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs}.neg();\n")
                                            }
                                            PolyType::Lagrange => {
                                                format!("{store_target} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};\n")
                                            }
                                        };
                                        kernel += &stmt;
                                    }
                                    SpOp::SubBy => {
                                        let stmt = match self.graph.poly_repr {
                                            PolyType::Coef => {
                                                format!("{store_target} = (idx == 0 && is_first) ? {TMP_PREFIX}{rhs} - {TMP_PREFIX}{lhs} : {TMP_PREFIX}{rhs};\n")
                                            }
                                            PolyType::Lagrange => {
                                                format!("{store_target} = {TMP_PREFIX}{rhs} - {TMP_PREFIX}{lhs};\n")
                                            }
                                        };
                                        kernel += &stmt;
                                    }
                                    SpOp::Mul => {
                                        kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{lhs} * {TMP_PREFIX}{rhs};\n"
                                    )
                                    }
                                    SpOp::Div => {
                                        unreachable!("division should be handled in type2")
                                    }
                                    SpOp::DivBy => {
                                        unreachable!("division should be handled in type2")
                                    }
                                },
                            }
                        }
                        Arith::Unr(op, arg) => {
                            let arg: usize = arg.clone().into();
                            match op {
                                UnrOp::P(ArithUnrOp::Neg) => {
                                    kernel +=
                                        &format!("{store_target} = {TMP_PREFIX}{arg}.neg();\n");
                                }
                                UnrOp::P(ArithUnrOp::Inv) => {
                                    unreachable!("invert poly should be handled in batche invert")
                                }
                                UnrOp::P(ArithUnrOp::Pow(power)) => {
                                    kernel += &format!(
                                        "{store_target} = {TMP_PREFIX}{arg}.pow({power});\n"
                                    );
                                }
                                UnrOp::S(ArithUnrOp::Neg) => {
                                    kernel +=
                                        &format!("{store_target} = {TMP_PREFIX}{arg}.neg();\n");
                                }
                                UnrOp::S(ArithUnrOp::Inv) => {
                                    unreachable!("invert scalar should be handled in scalar invert")
                                }
                                UnrOp::S(ArithUnrOp::Pow(_)) => {
                                    unreachable!("power scalar should be handled in scalar power")
                                }
                            }
                        }
                    },
                    Operation::Todo => unreachable!("todo should be handled earlier"),
                }
            }

            kernel.push_str("}\n");

            // explicitly instantiate the template
            kernel += &format!(
                "template __device__ void {}_{SUB_FUNC_NAME}{id}<{FIELD_NAME}>",
                self.name
            );
            kernel.push_str(&self.gen_device_func_args(id, FIELD_NAME));
            kernel.push_str(";\n");

            kernel.push_str("}\n");
            kernels.push(kernel);
        }
        kernels
    }
}

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
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            let mut len = 0;
            assert_eq!(meta.num_vars, var.len() - 1);
            assert_eq!(meta.num_mut_vars, mut_var.len() - 1);
            let stream = var[0].unwrap_stream();
            let (arg_buffer, mut_vars) = mut_var.split_at_mut(1);
            let arg_buffer = arg_buffer[0].unwrap_gpu_buffer_mut();
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
            f: FunctionValue::Fn(Box::new(rust_func)),
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
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            // the first of mut_var is the buffer for the arguments
            let (arg_buffer, mut_var) = mut_var.split_at_mut(1);
            let arg_buffer = arg_buffer[0].unwrap_gpu_buffer_mut();
            // check the buffer size
            assert_eq!(arg_buffer.size, (2 * num_of_vars + 3 * num_of_mut_vars) * std::mem::size_of::<PolyPtr>());

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
                CudaEventRaw::new(),
                CudaEventRaw::new(),
                CudaEventRaw::new(),
            ];
            let mut_compute_complete = [
                CudaEventRaw::new(),
                CudaEventRaw::new(),
                CudaEventRaw::new(),
            ];
            let mut_d2h_complete = [
                CudaEventRaw::new(),
                CudaEventRaw::new(),
                CudaEventRaw::new(),
            ];

            let h2d_complete = [CudaEventRaw::new(), CudaEventRaw::new()];
            let compute_complete = [CudaEventRaw::new(), CudaEventRaw::new()];

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

            Ok(())
        };
        Function {
            meta: FuncMeta::new(
                self.kernel.meta.name.clone(),
                KernelType::FusedArith(self.kernel.meta.clone()),
            ),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}
