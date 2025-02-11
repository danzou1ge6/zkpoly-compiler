use std::{
    any::type_name,
    collections::VecDeque,
    marker::PhantomData,
    os::raw::{c_uint, c_ulonglong},
};

use libloading::Symbol;
use zkpoly_common::{
    arith::{Arith, ArithBinOp, ArithGraph, ArithUnrOp, BinOp, Operation, POp, SpOp, UnrOp},
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

use crate::build_func::{resolve_type, xmake_config, xmake_run};

pub struct FusedKernel<T: RuntimeType> {
    _marker: PhantomData<T>,
    name: String,
    c_func: Symbol<
        'static,
        unsafe extern "C" fn(
            vars: *const *const c_uint,
            mut_vars: *const *mut c_uint,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub enum Var {
    Const(usize, FusedType),
    Mut(usize, FusedType),
}

pub enum FusedType {
    Scalar,
    ScalarArray,
}

pub struct FusedOp<Id> {
    graph: ArithGraph<Var, Id>,
    name: String,
    const_num: usize,
    mut_num: usize,
}

impl<Id: UsizeId> FusedOp<Id> {
    pub fn new(graph: ArithGraph<Var, Id>, name: String, const_num: usize, mut_num: usize) -> Self {
        Self {
            graph,
            name,
            const_num,
            mut_num,
        }
    }

    pub fn gen_header(&self) -> String {
        let mut header = String::new();
        header.push_str("#include \"../../common/mont/src/field_impls.cuh\"\n");
        // header.push_str("#include \"../../common/error/src/check.cuh\"\n");
        header
    }

    pub fn gen_wrapper(&self) -> String {
        let mut wrapper = String::new();
        wrapper.push_str("extern \"C\" cudaError_t ");
        wrapper.push_str(&self.name);
        wrapper.push_str(
            "(uint const * const* vars, uint * const* mut_vars, unsigned long long len, cudaStream_t stream) {\n",
        );
        wrapper.push_str("uint block_size = 256;\n");
        wrapper.push_str("uint grid_size = (len + block_size - 1) / block_size;\n");
        wrapper.push_str("detail::");
        wrapper.push_str(&self.name);
        wrapper.push_str(" <<< grid_size, block_size, 0, stream >>> (\n");
        for i in 0..self.const_num {
            wrapper.push_str("vars[");
            wrapper.push_str(&i.to_string());
            wrapper.push_str("], ");
        }
        for i in 0..self.mut_num {
            wrapper.push_str("mut_vars[");
            wrapper.push_str(&i.to_string());
            wrapper.push_str("], ");
        }
        wrapper.push_str("len)\n");
        wrapper.push_str("return cudaGetLastError();\n");
        wrapper.push_str("}\n");
        wrapper
    }

    pub fn gen_kernel(&self) -> String {
        let mut kernel = String::new();

        // generate kernel namespace
        kernel.push_str("namespace detail {\n");

        // generate kernel signature
        kernel.push_str("template <typename Field>\n");
        kernel.push_str("__global__ void ");
        kernel.push_str(&self.name);
        kernel.push_str("(");

        // generate kernel arguments
        for id in 0..self.const_num {
            kernel += &format!("uint const* const_var{}, ", id);
        }
        for id in 0..self.mut_num {
            kernel += &format!("uint* mut_var{}, ", id);
        }
        kernel += "unsigned long long len) {\n";
        kernel += "unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        kernel += "if (idx >= len) return;\n";

        // this is for topological ordering
        let mut deg = Vec::new();
        let mut queue = VecDeque::new();
        deg.resize(self.graph.heap.len(), 0);
        for node in self.graph.heap.iter_with_id() {
            let (id, vertex) = node;
            match &vertex.op {
                Operation::Input(_) => {
                    queue.push_back(id);
                    deg[id.into()] = 0;
                }
                Operation::Arith(arith) => match arith {
                    Arith::Bin(_, id1, id2) => {
                        if id1 != id2 {
                            deg[id.into()] = 2;
                        } else {
                            deg[id.into()] = 1;
                        }
                    }
                    Arith::Unr(..) => {
                        deg[id.into()] = 1;
                    }
                },
                Operation::Output(..) => {
                    deg[id.into()] = 1;
                }
            }
        }
        // topological ordering
        while !queue.is_empty() {
            let head = queue.pop_front().unwrap();
            let vertex = &self.graph.heap[head];
            for target in vertex.target.iter() {
                let id: usize = target.clone().into();
                deg[id] -= 1;
                if deg[id] == 0 {
                    queue.push_front(target.clone());
                }
            }
            match &vertex.op {
                Operation::Output(var, src) => match var {
                    Var::Mut(id, typ) => {
                        let src: usize = src.clone().into();
                        match typ {
                            FusedType::Scalar => {
                                kernel += &format!("var{}.store(mut_var{}, );\n", src, id);
                            }
                            FusedType::ScalarArray => {
                                kernel += &format!(
                                    "var{}.store(mut_var{} + idx * Field::LIMBS);\n",
                                    src, id
                                );
                            }
                        }
                    }
                    Var::Const(_, _) => unreachable!("Output should not be a constant"),
                },
                Operation::Input(var) => match var {
                    Var::Const(id, typ) => match typ {
                        FusedType::Scalar => {
                            kernel += &format!(
                                "auto var{} = Field::load(const_var{});\n",
                                head.into(),
                                id
                            );
                        }
                        FusedType::ScalarArray => {
                            kernel += &format!(
                                "auto var{} = Field::load(const_var{} + idx * Field::LIMBS);\n",
                                head.into(),
                                id
                            );
                        }
                    },
                    Var::Mut(id, typ) => match typ {
                        FusedType::Scalar => {
                            kernel +=
                                &format!("auto var{} = Field::load(mut_var{});\n", head.into(), id);
                        }
                        FusedType::ScalarArray => {
                            kernel += &format!(
                                "auto var{} = Field::load(mut_var{} + idx * Field::LIMBS);\n",
                                head.into(),
                                id
                            );
                        }
                    },
                },
                Operation::Arith(arith) => match arith {
                    Arith::Bin(op, lhs, rhs) => {
                        let lhs: usize = lhs.clone().into();
                        let rhs: usize = rhs.clone().into();
                        let head: usize = head.into();
                        match op {
                            BinOp::Pp(op) => match op {
                                ArithBinOp::Add => {
                                    kernel +=
                                        &format!("auto var{} = var{} + var{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Sub => {
                                    kernel +=
                                        &format!("auto var{} = var{} - var{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Mul => {
                                    kernel +=
                                        &format!("auto var{} = var{} * var{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Div => {
                                    eprintln!("Warning: division is very expensive, consider using batched inv first");
                                    kernel += &format!(
                                        "auto var{} = var{} * var{}.invert();\n",
                                        head, lhs, rhs
                                    );
                                }
                            },
                            BinOp::Ss(op) => match op {
                                ArithBinOp::Add => {
                                    kernel +=
                                        &format!("auto var{} = var{} + var{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Sub => {
                                    kernel +=
                                        &format!("auto var{} = var{} - var{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Mul => {
                                    kernel +=
                                        &format!("auto var{} = var{} * var{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Div => {
                                    eprintln!("Warning: division is very expensive, consider inverse the scalar first");
                                    kernel += &format!(
                                        "auto var{} = var{} * var{}.invert();\n",
                                        head, lhs, rhs
                                    );
                                }
                            },
                            BinOp::Sp(op) => match op {
                                SpOp::Add => {
                                    kernel +=
                                        &format!("auto var{} = var{} + var{};\n", head, lhs, rhs)
                                }
                                SpOp::Sub => {
                                    kernel +=
                                        &format!("auto var{} = var{} - var{};\n", head, lhs, rhs)
                                }
                                SpOp::SubBy => {
                                    kernel +=
                                        &format!("auto var{} = var{} - var{};\n", head, rhs, lhs)
                                }
                                SpOp::Mul => {
                                    kernel +=
                                        &format!("auto var{} = var{} * var{};\n", head, lhs, rhs)
                                }
                                SpOp::Div => {
                                    eprintln!("Warning: division is very expensive, consider inverse first");
                                    kernel += &format!(
                                        "auto var{} = var{} * var{}.invert();\n",
                                        head, lhs, rhs
                                    );
                                }
                                SpOp::DivBy => {
                                    eprintln!("Warning: division is very expensive, consider inverse first");
                                    kernel += &format!(
                                        "auto var{} = var{} * var{}.invert();\n",
                                        head, rhs, lhs
                                    );
                                }
                                SpOp::Eval => {
                                    unreachable!("SpOp::Eval is not supported in kernel fusion")
                                }
                            },
                        }
                    }
                    Arith::Unr(op, arg) => {
                        let arg: usize = arg.clone().into();
                        let head: usize = head.into();
                        match op {
                            UnrOp::P(POp::Neg) => {
                                kernel += &format!("auto var{} = -var{};\n", head, arg);
                            }
                            UnrOp::P(POp::Inv) => {
                                eprintln!("Warning: inversion is very expensive, consider using batched inv first");
                                kernel += &format!("auto var{} = var{}.invert();\n", head, arg);
                            }
                            UnrOp::S(ArithUnrOp::Neg) => {
                                kernel += &format!("auto var{} = -var{};\n", head, arg);
                            }
                            UnrOp::S(ArithUnrOp::Inv) => {
                                eprintln!("Warning: inversion is very expensive, consider inverse the scalar first");
                                kernel += &format!("auto var{} = var{}.invert();\n", head, arg);
                            }
                        }
                    }
                },
            }
        }
        kernel.push_str("}\n");
        kernel.push_str("}\n");
        kernel
    }
}

impl<T: RuntimeType> FusedKernel<T> {
    pub fn new(libs: &mut Libs, name: String) -> Self {
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("FUSED_FIELD", field_type);
        xmake_run("fused_kernels");
        let lib = libs.load("../lib/libfused_kernels.so");
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
        let rust_func = move |mut_var: Vec<&mut Variable<T>>,
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
                        poly.values as *const c_uint
                    }
                    Variable::Scalar(scalar) => {
                        if len == 0 {
                            len = 1;
                        }
                        scalar.value as *const c_uint
                    }
                    _ => unreachable!("Only scalars and scalar arrays are supported"),
                })
                .collect::<Vec<_>>();
            let mut_vars = mut_var
                .iter()
                .map(|v| match v {
                    Variable::ScalarArray(poly) => {
                        if len == 0 || len == 1 {
                            len = poly.len;
                        } else {
                            assert_eq!(len, poly.len);
                        }
                        poly.values as *mut c_uint
                    }
                    Variable::Scalar(scalar) => {
                        if len == 0 {
                            len = 1;
                        }
                        scalar.value as *mut c_uint
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
