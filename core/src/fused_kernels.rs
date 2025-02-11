use std::{
    any::type_name,
    collections::VecDeque,
    ffi::c_longlong,
    fmt::format,
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
            var_rotates: *const c_longlong,
            mut_vars: *const *mut c_uint,
            mut_var_rotates: *const c_longlong,
            len: c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
    >,
}

pub enum Var {
    Const(usize),
    Mut(usize),
}

pub enum FusedType {
    Scalar,
    ScalarArray,
}

pub struct FusedOp<Id> {
    graph: ArithGraph<Var, Id>,
    name: String,
    vars: Vec<FusedType>,
    mut_vars: Vec<FusedType>,
}

impl<Id: UsizeId> FusedOp<Id> {
    pub fn new(
        graph: ArithGraph<Var, Id>,
        name: String,
        vars: Vec<FusedType>,
        mut_vars: Vec<FusedType>,
    ) -> Self {
        Self {
            graph,
            name,
            vars,
            mut_vars,
        }
    }

    pub fn gen_header(&self) -> String {
        let mut header = String::new();
        header.push_str("#include \"../../common/mont/src/field_impls.cuh\"\n");
        header.push_str("#include \"../../common/mont/src/iter.cuh\"\n");
        header
    }

    pub fn gen_wrapper(&self) -> String {
        let mut wrapper = String::new();

        // signature
        wrapper.push_str("extern \"C\" cudaError_t ");
        wrapper.push_str(&self.name);
        wrapper.push_str(
            "(uint const * const* vars, long long const * var_rotates, uint * const* mut_vars, long long const * mut_var_rotates unsigned long long len, cudaStream_t stream) {\n",
        );

        for (i, var) in self.vars.iter().enumerate() {
            match var {
                FusedType::ScalarArray => {
                    let iter = format!("auto iter_{} = mont::make_rotating_iter(reinterpret_cast<const FUSED_FIELD*>(vars[{}]), var_rotates[{}], len);", i, i, i);
                    wrapper.push_str(&iter);
                }
                _ => {}
            }
        }
        for (i, mut_var) in self.mut_vars.iter().enumerate() {
            match mut_var {
                FusedType::ScalarArray => {
                    let iter = format!("auto iter_mut_{} = mont::make_rotating_iter(reinterpret_cast<FUSED_FIELD*>(mut_vars[{}]), mut_var_rotates[{}], len);", i, i, i);
                    wrapper.push_str(&iter);
                }
                _ => {}
            }
        }
        wrapper.push_str("uint block_size = 256;\n");
        wrapper.push_str("uint grid_size = (len + block_size - 1) / block_size;\n");
        wrapper.push_str("detail::");
        wrapper.push_str(&self.name);
        wrapper.push_str("<FUSED_FIELD>");
        wrapper.push_str(" <<< grid_size, block_size, 0, stream >>> (\n");
        for (i, var) in self.vars.iter().enumerate() {
            match var {
                FusedType::ScalarArray => {
                    wrapper.push_str("iter_");
                    wrapper.push_str(&i.to_string());
                    wrapper.push_str(", ");
                }
                FusedType::Scalar => {
                    wrapper.push_str("reinterpret_cast<const FUSED_FIELD*>(vars[");
                    wrapper.push_str(&i.to_string());
                    wrapper.push_str("]), ");
                }
            }
        }
        for (i, var) in self.mut_vars.iter().enumerate() {
            match var {
                FusedType::ScalarArray => {
                    wrapper.push_str("iter_mut_");
                    wrapper.push_str(&i.to_string());
                    wrapper.push_str(", ");
                }
                FusedType::Scalar => {
                    wrapper.push_str("reinterpret_cast<FUSED_FIELD*>(mut_vars[");
                    wrapper.push_str(&i.to_string());
                    wrapper.push_str("]), ");
                }
            }
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
        for (i, var) in self.vars.iter().enumerate() {
            match var {
                FusedType::Scalar => {
                    kernel.push_str("const Field* var");
                    kernel.push_str(&i.to_string());
                    kernel.push_str(", ");
                }
                FusedType::ScalarArray => {
                    kernel.push_str("mont::RotatingIter<const Field> iter_");
                    kernel.push_str(&i.to_string());
                    kernel.push_str(", ");
                }
            }
        }

        for (i, mut_var) in self.mut_vars.iter().enumerate() {
            match mut_var {
                FusedType::Scalar => {
                    kernel.push_str("Field* mut_var");
                    kernel.push_str(&i.to_string());
                    kernel.push_str(", ");
                }
                FusedType::ScalarArray => {
                    kernel.push_str("mont::RotatingIter<Field> iter_mut_");
                    kernel.push_str(&i.to_string());
                    kernel.push_str(", ");
                }
            }
        }

        kernel += "unsigned long long len) {\n";
        kernel += "unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        kernel += "if (idx >= len) return;\n";

        // this is for topological ordering
        let mut deg = Vec::new();
        let mut queue = VecDeque::new();
        deg.resize(self.graph.g.order(), 0);
        for id in self.graph.g.vertices() {
            let vertex = self.graph.g.vertex(id);
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
            let vertex = &self.graph.g.vertex(head);
            for target in vertex.target.iter() {
                let id: usize = target.clone().into();
                deg[id] -= 1;
                if deg[id] == 0 {
                    queue.push_front(target.clone());
                }
            }
            match &vertex.op {
                Operation::Output(var, src) => match var {
                    Var::Mut(id) => {
                        let src: usize = src.clone().into();
                        match self.mut_vars[*id] {
                            FusedType::Scalar => {
                                kernel += &format!("*mut_var{} = tmp{};\n", id, src);
                            }
                            FusedType::ScalarArray => {
                                kernel += &format!("iter_mut_{}[idx] = tmp{};\n", id, src);
                            }
                        }
                    }
                    Var::Const(..) => unreachable!("Output should not be a constant"),
                },
                Operation::Input(var) => match var {
                    Var::Const(id) => match self.vars[*id] {
                        FusedType::Scalar => {
                            kernel += &format!("auto tmp{} = *var{};\n", head.into(), id);
                        }
                        FusedType::ScalarArray => {
                            kernel += &format!("auto tmp{} = iter_{}[idx];\n", head.into(), id);
                        }
                    },
                    Var::Mut(id) => match self.mut_vars[*id] {
                        FusedType::Scalar => {
                            kernel +=
                                &format!("auto tmp{} = *mut_var{};\n", head.into(), id);
                        }
                        FusedType::ScalarArray => {
                            kernel += &format!(
                                "auto tmp{} = iter_mut_{}[idx];\n",
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
                                        &format!("auto tmp{} = tmp{} + tmp{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Sub => {
                                    kernel +=
                                        &format!("auto tmp{} = tmp{} - tmp{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Mul => {
                                    kernel +=
                                        &format!("auto tmp{} = tmp{} * tmp{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Div => {
                                    eprintln!("Warning: division is very expensive, consider using batched inv first");
                                    kernel += &format!(
                                        "auto tmp{} = tmp{} * tmp{}.invert();\n",
                                        head, lhs, rhs
                                    );
                                }
                            },
                            BinOp::Ss(op) => match op {
                                ArithBinOp::Add => {
                                    kernel +=
                                        &format!("auto tmp{} = tmp{} + tmp{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Sub => {
                                    kernel +=
                                        &format!("auto tmp{} = tmp{} - tmp{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Mul => {
                                    kernel +=
                                        &format!("auto tmp{} = tmp{} * tmp{};\n", head, lhs, rhs)
                                }
                                ArithBinOp::Div => {
                                    eprintln!("Warning: division is very expensive, consider inverse the scalar first");
                                    kernel += &format!(
                                        "auto tmp{} = tmp{} * tmp{}.invert();\n",
                                        head, lhs, rhs
                                    );
                                }
                            },
                            BinOp::Sp(op) => match op {
                                SpOp::Add => {
                                    kernel +=
                                        &format!("auto tmp{} = tmp{} + tmp{};\n", head, lhs, rhs)
                                }
                                SpOp::Sub => {
                                    kernel +=
                                        &format!("auto tmp{} = tmp{} - tmp{};\n", head, lhs, rhs)
                                }
                                SpOp::SubBy => {
                                    kernel +=
                                        &format!("auto tmp{} = tmp{} - tmp{};\n", head, rhs, lhs)
                                }
                                SpOp::Mul => {
                                    kernel +=
                                        &format!("auto tmp{} = tmp{} * tmp{};\n", head, lhs, rhs)
                                }
                                SpOp::Div => {
                                    eprintln!("Warning: division is very expensive, consider inverse first");
                                    kernel += &format!(
                                        "auto tmp{} = tmp{} * tmp{}.invert();\n",
                                        head, lhs, rhs
                                    );
                                }
                                SpOp::DivBy => {
                                    eprintln!("Warning: division is very expensive, consider inverse first");
                                    kernel += &format!(
                                        "auto tmp{} = tmp{} * tmp{}.invert();\n",
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
                                kernel += &format!("auto tmp{} = -tmp{};\n", head, arg);
                            }
                            UnrOp::P(POp::Inv) => {
                                eprintln!("Warning: inversion is very expensive, consider using batched inv first");
                                kernel += &format!("auto tmp{} = tmp{}.invert();\n", head, arg);
                            }
                            UnrOp::S(ArithUnrOp::Neg) => {
                                kernel += &format!("auto tmp{} = -tmp{};\n", head, arg);
                            }
                            UnrOp::S(ArithUnrOp::Inv) => {
                                eprintln!("Warning: inversion is very expensive, consider inverse the scalar first");
                                kernel += &format!("auto tmp{} = tmp{}.invert();\n", head, arg);
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
            let (vars, var_rotates) = var
                .iter()
                .skip(1)
                .map(|v| match v {
                    Variable::ScalarArray(poly) => {
                        if len == 0 || len == 1 {
                            len = poly.len;
                        } else {
                            assert_eq!(len, poly.len);
                        }
                        (poly.values as *const c_uint, poly.rotate as c_longlong)
                    }
                    Variable::Scalar(scalar) => {
                        if len == 0 {
                            len = 1;
                        }
                        (scalar.value as *const c_uint, 0 as c_longlong)
                    }
                    _ => unreachable!("Only scalars and scalar arrays are supported"),
                })
                .collect::<(Vec<_>, Vec<_>)>();
            let (mut_vars, mut_var_rotates) = mut_var
                .iter()
                .map(|v| match v {
                    Variable::ScalarArray(poly) => {
                        if len == 0 || len == 1 {
                            len = poly.len;
                        } else {
                            assert_eq!(len, poly.len);
                        }
                        (poly.values as *mut c_uint, poly.rotate as c_longlong)
                    }
                    Variable::Scalar(scalar) => {
                        if len == 0 {
                            len = 1;
                        }
                        (scalar.value as *mut c_uint, 0 as c_longlong)
                    }
                    _ => unreachable!("Only scalars and scalar arrays are supported"),
                })
                .collect::<(Vec<_>, Vec<_>)>();
            assert!(len > 0);
            unsafe {
                cuda_check!(cudaSetDevice(stream.get_device()));
                cuda_check!((c_func)(
                    vars.as_ptr(),
                    var_rotates.as_ptr(),
                    mut_vars.as_ptr(),
                    mut_var_rotates.as_ptr(),
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
