use std::{
    any::type_name,
    collections::BTreeSet,
    fs,
    marker::PhantomData,
    os::raw::{c_uint, c_ulonglong},
};

use libloading::Symbol;
use zkpoly_common::{
    arith::{
        Arith, ArithBinOp, ArithGraph, ArithUnrOp, BinOp, FusedType, Mutability, Operation, SpOp,
        UnrOp,
    },
    heap::UsizeId,
    load_dynamic::Libs,
};
use zkpoly_cuda_api::{
    bindings::{
        cudaError_cudaSuccess, cudaError_t, cudaGetErrorString, cudaSetDevice, cudaStream_t,
    },
    cuda_check,
};
use zkpoly_runtime::{
    args::{RuntimeType, Variable},
    error::RuntimeError,
    functions::{Function, FunctionValue, RegisteredFunction},
};

use crate::{
    build_func::{resolve_type, xmake_config, xmake_run},
    poly_ptr::{ConstPolyPtr, PolyPtr},
};

static LIB_NAME: &str = "libfused_kernels.so";

pub struct FusedKernel<T: RuntimeType> {
    _marker: PhantomData<T>,
    name: String,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            vars: *const ConstPolyPtr,
            mut_vars: *const PolyPtr,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub fn gen_var_lists<OuterId: Ord + Clone, InnerId: UsizeId>(
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
            if var_set.get(&outer_id).is_none() {
                var_set.insert(outer_id);
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
    for inner_id in graph.outputs.iter() {
        if let Operation::Output { outer_id, typ, .. } = &graph.g.vertex(*inner_id).op {
            if var_set.get(&outer_id).is_none() {
                var_set.insert(outer_id);
                let outer_id = (*outer_id).clone();
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

pub struct FusedOp<OuterId, InnerId> {
    graph: ArithGraph<OuterId, InnerId>,
    name: String,
    vars: Vec<(FusedType, OuterId)>,
    mut_vars: Vec<(FusedType, OuterId)>,
}

const TMP_PREFIX: &str = "tmp";
const SCALAR_PREFIX: &str = "var";
const ITER_PREFIX: &str = "iter";

impl<OuterId: UsizeId, InnerId: UsizeId + 'static> FusedOp<OuterId, InnerId> {
    pub fn new(graph: ArithGraph<OuterId, InnerId>, name: String) -> Self {
        let (vars, mut_vars) = gen_var_lists(&graph);
        Self {
            graph,
            name,
            vars,
            mut_vars,
        }
    }

    pub fn gen(&self) {
        let header = self.gen_header();
        let kernel = self.gen_kernel();
        let wrapper = self.gen_wrapper();
        fs::write("src/fused_kernels/src", header + &kernel + &wrapper).unwrap();
    }

    fn gen_header(&self) -> String {
        let mut header = String::new();
        header.push_str("#include \"../../common/mont/src/field_impls.cuh\"\n");
        header.push_str("#include \"../../common/iter/src/iter.cuh\"\n");
        header.push_str("#include <cuda_runtime.h>\n");
        header.push_str("using iter::SliceIterator;\n");
        header.push_str("using mont::u32;\n");
        header.push_str("using iter::make_slice_iter\n");
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
                        "auto {ITER_PREFIX}{id} = mont::make_slice_iter<FUSED_FIELD>(vars[{i}]);"
                    );
                    wrapper.push_str(&iter);
                }
                FusedType::Scalar => {
                    let scalar = format!("auto {SCALAR_PREFIX}{id} = reinterpret_cast<const FUSED_FIELD*>(vars[{i}].ptr);");
                    wrapper.push_str(&scalar);
                }
            }
        }
        for (i, (typ, id)) in self.mut_vars.iter().enumerate() {
            let id: usize = id.clone().into();
            match typ {
                FusedType::ScalarArray => {
                    let iter = format!("auto {ITER_PREFIX}{id} = mont::make_slice_iter<FUSED_FIELD>(mut_vars[{i}]);");
                    wrapper.push_str(&iter);
                }
                FusedType::Scalar => {
                    let scalar = format!(
                        "auto {SCALAR_PREFIX}{id} = reinterpret_cast<FUSED_FIELD*>(mut_vars[{i}].ptr);"
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
        wrapper.push_str("len)\n");
        wrapper.push_str("return cudaGetLastError();\n");
        wrapper.push_str("}\n");
        wrapper
    }

    fn gen_kernel(&self) -> String {
        let mut kernel = String::new();

        // generate kernel namespace
        kernel.push_str("namespace detail {\n");

        // generate kernel signature
        kernel.push_str("template <typename Field>\n");
        kernel.push_str("__global__ void ");
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
        for (head, vertex) in self.graph.topology_sort() {
            match &vertex.op {
                Operation::Output {
                    outer_id,
                    typ,
                    store_node,
                    ..
                } => {
                    let src: usize = store_node.clone().into();
                    let id: usize = outer_id.clone().into();
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
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} + {TMP_PREFIX}{};\n",
                                        head, lhs, rhs
                                    )
                                }
                                SpOp::Sub => {
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} - {TMP_PREFIX}{};\n",
                                        head, lhs, rhs
                                    )
                                }
                                SpOp::SubBy => {
                                    kernel += &format!(
                                        "auto {TMP_PREFIX}{} = {TMP_PREFIX}{} - {TMP_PREFIX}{};\n",
                                        head, rhs, lhs
                                    )
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
                                kernel +=
                                    &format!("auto {TMP_PREFIX}{} = -{TMP_PREFIX}{};\n", head, arg);
                            }
                            UnrOp::P(ArithUnrOp::Inv) => {
                                eprintln!("Warning: inversion is very expensive, consider using batched inv first");
                                kernel += &format!(
                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{}.invert();\n",
                                    head, arg
                                );
                            }
                            UnrOp::P(ArithUnrOp::Pow(power)) => {
                                kernel += &format!(
                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{}.pow({});\n",
                                    head, arg, power
                                );
                            }
                            UnrOp::S(ArithUnrOp::Neg) => {
                                kernel +=
                                    &format!("auto {TMP_PREFIX}{} = -{TMP_PREFIX}{};\n", head, arg);
                            }
                            UnrOp::S(ArithUnrOp::Inv) => {
                                eprintln!("Warning: inversion is very expensive, consider inverse the scalar first");
                                kernel += &format!(
                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{}.invert();\n",
                                    head, arg
                                );
                            }
                            UnrOp::S(ArithUnrOp::Pow(power)) => {
                                kernel += &format!(
                                    "auto {TMP_PREFIX}{} = {TMP_PREFIX}{}.pow({});\n",
                                    head, arg, power
                                );
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
    pub fn new(libs: &mut Libs, name: String) -> Self {
        if !libs.contains(LIB_NAME) {
            let field_type = resolve_type(type_name::<T::Field>());
            xmake_config("FUSED_FIELD", field_type);
            xmake_run("fused_kernels");
        }
        let lib = libs.load(LIB_NAME);
        // get the function pointer with the provided name (with null terminator)
        let c_func = unsafe { lib.get(format!("{}\0", name).as_bytes()) }
            .expect("Failed to load function pointer");
        Self {
            _marker: PhantomData,
            name,
            c_func,
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for FusedKernel<T> {
    fn get_fn(&self) -> Function<T> {
        let c_func = self.c_func.clone();
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            let mut len = 0;
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
            name: self.name.clone(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}
