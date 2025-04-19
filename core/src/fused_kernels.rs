use std::io::{Read, Write};
use std::{
    any::type_name,
    borrow::Borrow,
    fs,
    marker::PhantomData,
    os::raw::{c_uint, c_ulonglong},
};

use libloading::Symbol;
use zkpoly_common::{
    arith::{Arith, ArithBinOp, ArithGraph, ArithUnrOp, BinOp, FusedType, Operation, SpOp, UnrOp},
    get_project_root::get_project_root,
    heap::UsizeId,
    load_dynamic::Libs,
    typ::PolyType,
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
    functions::{FuncMeta, Function, FunctionValue, KernelType, RegisteredFunction},
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

pub struct FusedOp<OuterId, InnerId> {
    graph: ArithGraph<OuterId, InnerId>,
    name: String,
    vars: Vec<(FusedType, InnerId)>,
    mut_vars: Vec<(FusedType, InnerId)>,
}

const TMP_PREFIX: &str = "tmp";
const SCALAR_PREFIX: &str = "var";
const ITER_PREFIX: &str = "iter";

// fn scalar_poly_arith(
//     head: usize,
//     lhs: usize,
//     rhs: usize,
//     op: &'static str,
//     repr: PolyType,
// ) -> String {
//     match repr {
//         PolyType::Coef => {
//             format!("auto {TMP_PREFIX}{head} = idx == 0 ? {TMP_PREFIX}{lhs} {op} {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs};")
//         }
//         PolyType::Lagrange => {
//             format!("auto {TMP_PREFIX}{head} = {TMP_PREFIX}{lhs} {op} {TMP_PREFIX}{rhs};")
//         }
//     }
// }

impl<OuterId: UsizeId, InnerId: UsizeId + 'static> FusedOp<OuterId, InnerId> {
    pub fn new(graph: ArithGraph<OuterId, InnerId>, name: String) -> Self {
        let (vars, mut_vars) = graph.gen_var_lists();
        Self {
            graph,
            name,
            vars,
            mut_vars,
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
                    typ, store_node, ..
                } => {
                    let src: usize = store_node.clone().into();
                    let id: usize = head.clone().into();
                    match typ {
                        FusedType::Scalar => {
                            kernel += &format!("*{SCALAR_PREFIX}{id} = {TMP_PREFIX}{src};\n");
                        }
                        FusedType::ScalarArray => {
                            kernel += &format!("{ITER_PREFIX}{id}[idx] = {TMP_PREFIX}{src};\n");
                        }
                    }
                }
                Operation::Input { typ, .. } => {
                    let head = head.clone().into();
                    match typ {
                        FusedType::Scalar => {
                            kernel +=
                                &format!("auto {TMP_PREFIX}{head} = *{SCALAR_PREFIX}{head};\n");
                        }
                        FusedType::ScalarArray => {
                            kernel +=
                                &format!("auto {TMP_PREFIX}{head} = {ITER_PREFIX}{head}[idx];\n");
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
                                            format!("auto {TMP_PREFIX}{head} = idx == 0 ? {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs} : {TMP_PREFIX}{rhs};")
                                        }
                                        PolyType::Lagrange => {
                                            format!("auto {TMP_PREFIX}{head} = {TMP_PREFIX}{lhs} + {TMP_PREFIX}{rhs};")
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
            meta: FuncMeta::new(self.name.clone(), KernelType::FusedArith(self.name.clone())),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}
