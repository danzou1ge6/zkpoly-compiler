use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
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
use zkpoly_cuda_api::stream::CudaEvent;
use zkpoly_cuda_api::{
    bindings::{
        cudaError_cudaSuccess, cudaError_t, cudaGetErrorString, cudaSetDevice, cudaStream_t,
    },
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

pub mod scheduler;

pub struct FusedKernel<T: RuntimeType> {
    _marker: PhantomData<T>,
    pub meta: FusedKernelMeta,
    pub c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            vars: *const ConstPolyPtr,
            mut_vars: *const PolyPtr,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

// all input/output are on cpu
pub struct PipelinedFusedKernel<T: RuntimeType> {
    kernel: FusedKernel<T>,
    divide_parts: usize, // how many parts to divide the poly into, must > 3 and later calls must have len which is a multiple of this
    num_scalars: usize,
    num_mut_scalars: usize,
}

pub struct FusedOp<OuterId, InnerId> {
    graph: ArithGraph<OuterId, InnerId>,
    name: String,
    pub vars: Vec<(FusedType, OuterId)>,
    pub mut_vars: Vec<(FusedType, OuterId)>,
    outputs_i2o: BTreeMap<InnerId, OuterId>,
    limbs: usize,
}

const TMP_PREFIX: &str = "tmp";
const SCALAR_PREFIX: &str = "var";
const ITER_PREFIX: &str = "iter";

pub fn gen_var_lists<OuterId: Ord + Copy, InnerId: UsizeId>(
    outputs: impl Iterator<Item = OuterId>,
    graph: &ArithGraph<OuterId, InnerId>,
) -> (Vec<(FusedType, OuterId)>, Vec<(FusedType, OuterId)>) {
    let mut vars = Vec::new();
    let mut mut_vars = Vec::new();
    let mut var_set = BTreeSet::new();
    for inner_id in graph.inputs.iter() {
        if let Operation::Input {
            outer_id,
            typ,
            mutability,
        } = &graph.g.vertex(*inner_id).op
        {
            if var_set.get(outer_id).is_none() {
                var_set.insert(*outer_id);
                let outer_id = (*outer_id).clone();
                match (typ, mutability) {
                    (FusedType::Scalar, Mutability::Const) => {
                        vars.push((FusedType::Scalar, outer_id))
                    }
                    (FusedType::Scalar, Mutability::Mut) => {
                        mut_vars.push((FusedType::Scalar, outer_id))
                    }
                    (FusedType::ScalarArray, Mutability::Const) => {
                        vars.push((FusedType::ScalarArray, outer_id))
                    }
                    (FusedType::ScalarArray, Mutability::Mut) => {
                        mut_vars.push((FusedType::ScalarArray, outer_id))
                    }
                }
            }
        } else {
            unreachable!("input should be an Operation::Input");
        }
    }
    for (inner_id, outer_id) in graph.outputs.iter().zip(outputs) {
        if let Operation::Output { typ, .. } = &graph.g.vertex(*inner_id).op {
            if var_set.get(&outer_id).is_none() {
                var_set.insert(outer_id);
                let outer_id = (outer_id).clone();
                match typ {
                    FusedType::Scalar => mut_vars.push((FusedType::Scalar, outer_id)),
                    FusedType::ScalarArray => mut_vars.push((FusedType::ScalarArray, outer_id)),
                }
            }
        } else {
            unreachable!("output should be an Operation::Output");
        }
    }
    (vars, mut_vars)
}

impl<OuterId: UsizeId, InnerId: UsizeId + 'static> FusedOp<OuterId, InnerId> {
    pub fn new(
        graph: ArithGraph<OuterId, InnerId>,
        name: String,
        outputs_i2o: BTreeMap<InnerId, OuterId>,
        limbs: usize,
    ) -> Self {
        let (vars, mut_vars) = gen_var_lists(graph.outputs.iter().map(|i| outputs_i2o[i]), &graph);
        Self {
            graph,
            name,
            vars,
            mut_vars,
            outputs_i2o,
            limbs,
        }
    }

    pub fn gen(&self, head_annotation: impl Borrow<str>) {
        let header = self.gen_header();
        let kernel = self.gen_kernel();
        let wrapper = self.gen_wrapper();

        let s = format!(
            "{}\n{}{}{}",
            head_annotation.borrow(),
            header,
            kernel,
            wrapper
        );

        let project_root = get_project_root();
        let path = project_root + "/core/src/fused_kernels/src/" + self.name.as_str() + ".cu";

        if let Ok(mut f) = fs::File::open(&path) {
            let mut old = String::new();
            f.read_to_string(&mut old).unwrap();
            if s == old {
                return;
            }
        }

        let mut f = fs::File::create(&path).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    fn gen_header(&self) -> String {
        let mut header = String::new();
        header.push_str("#include \"../../common/mont/src/field_impls.cuh\"\n");
        header.push_str("#include \"../../common/iter/src/iter.cuh\"\n");
        header.push_str("#include <cuda_runtime.h>\n");
        header.push_str("using iter::SliceIterator;\n");
        header.push_str("using mont::u32;\n");
        header.push_str("using iter::make_slice_iter;\n");
        header
    }

    fn gen_wrapper(&self) -> String {
        let mut wrapper = String::new();

        // signature
        wrapper.push_str("extern \"C\" cudaError_t ");
        wrapper.push_str(&self.name);
        wrapper.push_str(
            "(ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long len, cudaStream_t stream) {\n",
        );

        for (i, (typ, id)) in self.vars.iter().enumerate() {
            let id: usize = id.clone().into();
            match typ {
                FusedType::ScalarArray => {
                    let iter = format!(
                        "auto {ITER_PREFIX}{id} = make_slice_iter<FUSED_FIELD>(vars[{i}]);\n"
                    );
                    wrapper.push_str(&iter);
                }
                FusedType::Scalar => {
                    let scalar = format!("auto {SCALAR_PREFIX}{id} = reinterpret_cast<const FUSED_FIELD*>(vars[{i}].ptr);\n");
                    wrapper.push_str(&scalar);
                }
            }
        }
        for (i, (typ, id)) in self.mut_vars.iter().enumerate() {
            let id: usize = id.clone().into();
            match typ {
                FusedType::ScalarArray => {
                    let iter = format!(
                        "auto {ITER_PREFIX}{id} = make_slice_iter<FUSED_FIELD>(mut_vars[{i}]);\n"
                    );
                    wrapper.push_str(&iter);
                }
                FusedType::Scalar => {
                    let scalar = format!(
                        "auto {SCALAR_PREFIX}{id} = reinterpret_cast<FUSED_FIELD*>(mut_vars[{i}].ptr);\n"
                    );
                    wrapper.push_str(&scalar);
                }
            }
        }
        wrapper.push_str("uint block_size = 256;\n");
        wrapper.push_str("uint grid_size = (len + block_size - 1) / block_size;\n");
        wrapper.push_str("detail::");
        wrapper.push_str(&self.name);
        wrapper.push_str("<FUSED_FIELD>");
        wrapper.push_str(" <<< grid_size, block_size, 0, stream >>> (\n");
        for (typ, id) in self.vars.iter().chain(self.mut_vars.iter()) {
            let id = id.clone().into();
            match typ {
                FusedType::ScalarArray => {
                    wrapper.push_str(ITER_PREFIX);
                    wrapper.push_str(&id.to_string());
                    wrapper.push_str(", ");
                }
                FusedType::Scalar => {
                    wrapper.push_str(SCALAR_PREFIX);
                    wrapper.push_str(&id.to_string());
                    wrapper.push_str(", ");
                }
            }
        }
        wrapper.push_str("len);\n");
        wrapper.push_str("return cudaGetLastError();\n");
        wrapper.push_str("}\n");
        wrapper
    }

    fn gen_kernel(&self) -> String {
        let mut kernel = String::new();

        let (schedule, regs) = scheduler::schedule(&self.graph);

        let u32_regs = regs * self.limbs;

        let launch_bounds = if u32_regs < 128 {
            1024
        } else if u32_regs < 256 {
            512
        } else {
            256
        };

        // generate kernel namespace
        kernel.push_str("namespace detail {\n");

        // generate kernel signature
        kernel.push_str("template <typename Field>\n");
        kernel += &format!("__launch_bounds__({launch_bounds}) __global__ void ");
        kernel.push_str(&self.name);
        kernel.push_str("(");

        // generate kernel arguments
        for (typ, id) in self.vars.iter() {
            let id: usize = id.clone().into();
            match typ {
                FusedType::Scalar => {
                    kernel.push_str("const Field* ");
                    kernel.push_str(SCALAR_PREFIX);
                    kernel.push_str(&id.to_string());
                    kernel.push_str(", ");
                }
                FusedType::ScalarArray => {
                    kernel.push_str("SliceIterator<const Field> ");
                    kernel.push_str(ITER_PREFIX);
                    kernel.push_str(&id.to_string());
                    kernel.push_str(", ");
                }
            }
        }

        for (typ, id) in self.mut_vars.iter() {
            let id: usize = id.clone().into();
            match typ {
                FusedType::Scalar => {
                    kernel.push_str("Field* ");
                    kernel.push_str(SCALAR_PREFIX);
                    kernel.push_str(&id.to_string());
                    kernel.push_str(", ");
                }
                FusedType::ScalarArray => {
                    kernel.push_str("SliceIterator<Field> ");
                    kernel.push_str(ITER_PREFIX);
                    kernel.push_str(&id.to_string());
                    kernel.push_str(", ");
                }
            }
        }

        kernel += "unsigned long long len) {\n";
        kernel += "unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        kernel += "if (idx >= len) return;\n";

        // topological ordering
        for head in schedule {
            let vertex = self.graph.g.vertex(head.clone());
            match &vertex.op {
                Operation::Output {
                    typ, store_node, ..
                } => {
                    let src: usize = store_node.clone().into();
                    let id: usize = self.outputs_i2o[&head].clone().into();
                    match typ {
                        FusedType::Scalar => {
                            kernel += &format!("*{SCALAR_PREFIX}{id} = {TMP_PREFIX}{src};\n");
                        }
                        FusedType::ScalarArray => {
                            kernel += &format!("{ITER_PREFIX}{id}[idx] = {TMP_PREFIX}{src};\n");
                        }
                    }
                }
                Operation::Input { outer_id, typ, .. } => {
                    let id = outer_id.clone().into();
                    let head = head.clone().into();
                    match typ {
                        FusedType::Scalar => {
                            kernel += &format!("auto {TMP_PREFIX}{head} = *{SCALAR_PREFIX}{id};\n");
                        }
                        FusedType::ScalarArray => {
                            kernel +=
                                &format!("auto {TMP_PREFIX}{head} = {ITER_PREFIX}{id}[idx];\n");
                        }
                    }
                }
                Operation::Arith(arith) => match arith {
                    Arith::Bin(op, lhs, rhs) => {
                        let lhs: usize = lhs.clone().into();
                        let rhs: usize = rhs.clone().into();
                        let head: usize = head.into();
                        match op {
                            BinOp::Pp(op) => match op {
                                ArithBinOp::Add => {
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} + {TMP_PREFIX}{};\n",
                                        head, lhs, rhs
                                    )
                                }
                                ArithBinOp::Sub => {
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} - {TMP_PREFIX}{};\n",
                                        head, lhs, rhs
                                    )
                                }
                                ArithBinOp::Mul => {
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} * {TMP_PREFIX}{};\n",
                                        head, lhs, rhs
                                    )
                                }
                                ArithBinOp::Div => {
                                    eprintln!("Warning: division is very expensive, consider using batched inv first");
                                    kernel += &format!(
                                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} * {TMP_PREFIX}{}.invert();\n",
                                                    head, lhs, rhs
                                                );
                                }
                            },
                            BinOp::Ss(op) => match op {
                                ArithBinOp::Add => {
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} + {TMP_PREFIX}{};\n",
                                        head, lhs, rhs
                                    )
                                }
                                ArithBinOp::Sub => {
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} - {TMP_PREFIX}{};\n",
                                        head, lhs, rhs
                                    )
                                }
                                ArithBinOp::Mul => {
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} * {TMP_PREFIX}{};\n",
                                        head, lhs, rhs
                                    )
                                }
                                ArithBinOp::Div => {
                                    eprintln!("Warning: division is very expensive, consider inverse the scalar first");
                                    kernel += &format!(
                                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} * {TMP_PREFIX}{}.invert();\n",
                                                    head, lhs, rhs
                                                );
                                }
                            },
                            BinOp::Sp(op) => match op {
                                SpOp::Add => {
                                    let stmt = match self.graph.poly_repr {
                                        PolyType::Coef => {
                                            format!("auto {TMP_PREFIX}{head} = idx == 0 ? {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs};\n")
                                        }
                                        PolyType::Lagrange => {
                                            format!("auto {TMP_PREFIX}{head} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};\n")
                                        }
                                    };
                                    kernel += &stmt;
                                }
                                SpOp::Sub => {
                                    let stmt = match self.graph.poly_repr {
                                        PolyType::Coef => {
                                            format!("auto {TMP_PREFIX}{head} = idx == 0 ? {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs}.neg();")
                                        }
                                        PolyType::Lagrange => {
                                            format!("auto {TMP_PREFIX}{head} = {TMP_PREFIX}{lhs} - {TMP_PREFIX}{rhs};")
                                        }
                                    };
                                    kernel += &stmt;
                                }
                                SpOp::SubBy => {
                                    let stmt = match self.graph.poly_repr {
                                        PolyType::Coef => {
                                            format!("auto {TMP_PREFIX}{head} = idx == 0 ? {TMP_PREFIX}{rhs} - {TMP_PREFIX}{lhs} : {TMP_PREFIX}{rhs};")
                                        }
                                        PolyType::Lagrange => {
                                            format!("auto {TMP_PREFIX}{head} = {TMP_PREFIX}{rhs} - {TMP_PREFIX}{lhs};")
                                        }
                                    };
                                    kernel += &stmt;
                                }
                                SpOp::Mul => {
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} * {TMP_PREFIX}{};\n",
                                        head, lhs, rhs
                                    )
                                }
                                SpOp::Div => {
                                    eprintln!("Warning: division is very expensive, consider inverse first");
                                    kernel += &format!(
                                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} * {TMP_PREFIX}{}.invert();\n",
                                                    head, lhs, rhs
                                                );
                                }
                                SpOp::DivBy => {
                                    eprintln!("Warning: division is very expensive, consider inverse first");
                                    kernel += &format!(
                                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} * {TMP_PREFIX}{}.invert();\n",
                                                    head, rhs, lhs
                                                );
                                }
                            },
                        }
                    }
                    Arith::Unr(op, arg) => {
                        let arg: usize = arg.clone().into();
                        let head: usize = head.into();
                        match op {
                            UnrOp::P(ArithUnrOp::Neg) => {
                                kernel += &format!(
                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{}.neg();\n",
                                    head, arg
                                );
                            }
                            UnrOp::P(ArithUnrOp::Inv) => {
                                unreachable!("invert poly should be handled in batche invert")
                            }
                            UnrOp::P(ArithUnrOp::Pow(power)) => {
                                kernel += &format!(
                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{}.pow({});\n",
                                    head, arg, power
                                );
                            }
                            UnrOp::S(ArithUnrOp::Neg) => {
                                kernel += &format!(
                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{}.neg();\n",
                                    head, arg
                                );
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
                Operation::Todo => unreachable!("todo can't appear here"),
            }
        }

        kernel.push_str("}\n");
        kernel.push_str("}\n");
        kernel
    }
}

impl<T: RuntimeType> FusedKernel<T> {
    pub fn new(libs: &mut Libs, meta: FusedKernelMeta) -> Self {
        if !libs.contains(LIB_NAME) {
            let field_type = resolve_type(type_name::<T::Field>());
            xmake_config("FUSED_FIELD", field_type);
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
        let c_func = self.c_func.clone();
        let meta = self.meta.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            let mut len = 0;
            assert_eq!(meta.num_vars, var.len() - 1);
            assert_eq!(meta.num_mut_vars, mut_var.len());
            let stream = var[0].unwrap_stream();
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
            let mut_vars = mut_var
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
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!((c_func)(
                    vars.as_ptr(),
                    mut_vars.as_ptr(),
                    len.try_into().unwrap(),
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
    pub fn new(
        libs: &mut Libs,
        meta: FusedKernelMeta,
        len: usize,
        divide_parts: usize,
        num_scalars: usize,
        num_mut_scalars: usize,
    ) -> Self {
        assert!(divide_parts > 3, "divide_parts must be greater than 3");
        assert!(
            len % divide_parts == 0,
            "len must be a multiple of divide_parts"
        );
        Self {
            kernel: FusedKernel::new(libs, meta),
            divide_parts,
            num_scalars,
            num_mut_scalars,
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for PipelinedFusedKernel<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.kernel.c_func.clone();
        let num_scalars = self.num_scalars;
        let num_mut_scalars = self.num_mut_scalars;
        let divide_parts = self.divide_parts;
        /*
        args:
        assume there are n mut polys, m polys, p mut scalars, q scalrs to compute the fused kernel
        mut_var will have 4n + 2p elements, the first p are mut scalars, next n are the mut polys,
        then next p is gpu buffer,
        the next 3n are the gpu polys (triple buffer, one load, one compute, one store)
        var will have 3m + 3 + 2q elements, the first 3 are streams(load, compute, store),
        the next q are scalars, next m are polys, next q are scalar buffers, the next 2m are the gpu polys (double buffer, one load, one compute)
         */
        let rust_func = move |mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert!((mut_var.len() - 2 * num_mut_scalars) % 4 == 0);
            assert!((var.len() - 2 * num_scalars) % 3 == 0);

            let num_mut_poly = (mut_var.len() - 2 * num_mut_scalars) / 4;
            let num_mut_var = num_mut_poly + num_mut_scalars;
            let num_poly = (var.len() - 3 - 2 * num_scalars) / 3;
            let num_var = num_poly + num_scalars;

            // get the length
            let len = if num_mut_poly > 0 {
                mut_var[num_mut_scalars].unwrap_scalar_array().len()
            } else if num_poly > 0 {
                var[3 + num_scalars].unwrap_scalar_array().len()
            } else {
                1
            };

            let chunk_len = len / divide_parts;

            // get streams
            let h2d_stream = var[0].unwrap_stream();
            let compute_stream = var[1].unwrap_stream();
            let d2h_stream = var[2].unwrap_stream();

            // get scalars
            let mut mut_scalars = Vec::new();
            for i in 0..num_mut_scalars {
                mut_scalars.push(mut_var[i].unwrap_scalar().clone());
            }
            let mut scalars = Vec::new();
            for i in 0..num_scalars {
                scalars.push(var[i + 3].unwrap_scalar().clone());
            }

            // get scalar buffers
            let mut mut_gpu_scalars = Vec::new();
            for i in 0..num_mut_scalars {
                let buffer = mut_var[i + num_mut_scalars].unwrap_gpu_buffer();
                let gpu_scalar =
                    Scalar::new_gpu(buffer.ptr as *mut T::Field, buffer.device.unwrap_gpu());
                mut_gpu_scalars.push(gpu_scalar);
            }
            let mut gpu_scalars = Vec::new();
            for i in 0..num_scalars {
                let buffer = var[i + 3 + num_scalars].unwrap_gpu_buffer();
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
                polys.push(var[i + 3 + num_scalars].unwrap_scalar_array().clone());
                assert!(
                    polys[i].slice_info.is_none(),
                    "pipelined fused kernel doesn't support slice"
                );
            }

            // get poly buffers
            let mut mut_gpu_polys = vec![Vec::new(); 3];
            let base_index = num_mut_var + num_mut_scalars;
            for i in 0..num_mut_poly {
                for j in 0..3 {
                    let buffer = mut_var[i + base_index + j * num_mut_poly].unwrap_gpu_buffer();
                    let gpu_poly =
                        ScalarArray::new(len, buffer.ptr as *mut T::Field, buffer.device.clone());
                    mut_gpu_polys[j].push(gpu_poly);
                }
            }
            let mut gpu_polys = vec![Vec::new(); 2];
            let base_index = 3 + num_var + num_scalars;
            for i in 0..num_var {
                for j in 0..2 {
                    let buffer = var[i + base_index + j * num_poly].unwrap_gpu_buffer();
                    let gpu_poly =
                        ScalarArray::new(len, buffer.ptr as *mut T::Field, buffer.device.clone());
                    gpu_polys[j].push(gpu_poly);
                }
            }

            // create events
            let mut_h2d_complete = [CudaEvent::new(), CudaEvent::new(), CudaEvent::new()];
            let mut_compute_complete = [CudaEvent::new(), CudaEvent::new(), CudaEvent::new()];
            let mut_d2h_complete = [CudaEvent::new(), CudaEvent::new(), CudaEvent::new()];

            let h2d_complete = [CudaEvent::new(), CudaEvent::new()];
            let compute_complete = vec![CudaEvent::new(), CudaEvent::new()];

            let mut mut_buffer_id = 0;
            let mut buffer_id = 0;

            // start computing
            for chunk_id in 0..divide_parts {
                // load mutable data to gpu
                h2d_stream.wait(&mut_d2h_complete[mut_buffer_id]);

                let compute_start = chunk_id * chunk_len;
                let compute_end = (chunk_id + 1) * chunk_len;
                // mut polys
                for i in 0..num_mut_poly {
                    let mut_poly = mut_polys[i].slice(compute_start, compute_end);
                    mut_poly.cpu2gpu(&mut mut_gpu_polys[mut_buffer_id][i], h2d_stream);
                }
                mut_h2d_complete[mut_buffer_id].record(h2d_stream);

                h2d_stream.wait(&compute_complete[buffer_id]);
                // polys
                for i in 0..num_poly {
                    let poly = polys[i].slice(compute_start, compute_end);
                    poly.cpu2gpu(&mut gpu_polys[buffer_id][i], h2d_stream);
                }
                h2d_complete[buffer_id].record(h2d_stream);

                // wait for the previous transfer to finish
                compute_stream.wait(&mut_h2d_complete[mut_buffer_id]);
                compute_stream.wait(&h2d_complete[buffer_id]);

                // compute
                let mut mut_vars = Vec::new();
                for scalar in mut_gpu_scalars.iter() {
                    mut_vars.push(PolyPtr {
                        ptr: scalar.value as *mut c_uint,
                        len: 1,
                        rotate: 0,
                        offset: 0,
                        whole_len: 1,
                    })
                }
                for poly in mut_gpu_polys[mut_buffer_id].iter_mut() {
                    mut_vars.push(PolyPtr::from(poly))
                }
                let mut vars = Vec::new();
                for scalar in gpu_scalars.iter() {
                    vars.push(ConstPolyPtr {
                        ptr: scalar.value as *mut c_uint,
                        len: 1,
                        rotate: 0,
                        offset: 0,
                        whole_len: 1,
                    })
                }
                for poly in gpu_polys[buffer_id].iter() {
                    vars.push(ConstPolyPtr::from(poly))
                }

                unsafe {
                    cuda_check!(cudaSetDevice(compute_stream.get_device()));
                    cuda_check!((c_func)(
                        vars.as_ptr(),
                        mut_vars.as_ptr(),
                        chunk_len.try_into().unwrap(),
                        compute_stream.raw()
                    ));
                }

                compute_complete[buffer_id].record(compute_stream);
                mut_compute_complete[mut_buffer_id].record(compute_stream);

                // wait for the previous compute to finish
                d2h_stream.wait(&mut_compute_complete[mut_buffer_id]);

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
