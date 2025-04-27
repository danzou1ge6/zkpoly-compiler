use crate::args::{RuntimeType, Variable};
use crate::error::RuntimeError;
use serde::{Deserialize, Serialize};
use zkpoly_common::heap;
use zkpoly_common::msm_config::MsmConfig;

zkpoly_common::define_usize_id!(FunctionId);
zkpoly_common::define_usize_id!(UserFunctionId);

pub type FunctionTable<T> = heap::Heap<FunctionId, Function<T>>;

pub trait RegisteredFunction<T: RuntimeType> {
    fn get_fn(&self) -> Function<T>;
}

pub enum FunctionValue<T: RuntimeType> {
    FnOnce(
        Option<
            Box<
                dyn FnOnce(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                    + Sync
                    + Send
                    + 'static,
            >,
        >,
    ),
    FnMut(
        Box<
            dyn FnMut(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ),
    Fn(
        Box<
            dyn Fn(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ),
}

pub struct Function<T: RuntimeType> {
    pub meta: FuncMeta,
    pub f: FunctionValue<T>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd, Eq, Ord)]
pub struct FusedKernelMeta {
    pub name: String,
    pub num_vars: usize,
    pub num_mut_vars: usize,
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KernelType {
    NttPrcompute,
    NttRecompute,
    Msm(MsmConfig),
    BatchedInvert,
    KateDivision,
    EvaluatePoly,
    ScanMul,
    FusedArith(FusedKernelMeta),
    Interpolate,
    AssmblePoly,
    HashTranscript,
    HashTranscriptWrite,
    SqueezeScalar,
    UserFunction(UserFunctionId),
    DistributePowers,
    NewOneLagrange,
    NewOneCoef,
    NewZero,
    ScalarInvert,   // TODO: support both cpu and gpu
    ScalarPow(u64), // TODO: support both cpu and gpu
    // there two kernels are mainly for input with different len
    // which auto generated kernels can't handle
    PolyAdd,
    PolySub,
    Other, // for test use
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FuncMeta {
    pub name: String,
    pub typ: KernelType,
}

impl FuncMeta {
    pub fn new(name: String, typ: KernelType) -> Self {
        Self { name, typ }
    }
}

impl<T: RuntimeType> std::fmt::Debug for Function<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut st = f.debug_struct("Function");

        if let FunctionValue::FnOnce(_) = &self.f {
            st.field("mutability", &"FnOnce");
        } else {
            st.field("mutability", &"FnMut");
        }
        st.finish()
    }
}

impl<T: RuntimeType> Function<T> {
    pub fn new_once(
        meta: FuncMeta,
        f: Box<
            dyn FnOnce(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ) -> Self {
        Self {
            meta,
            f: FunctionValue::FnOnce(Some(f)),
        }
    }

    pub fn new_mut(
        meta: FuncMeta,
        f: Box<
            dyn FnMut(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ) -> Self {
        Self {
            meta,
            f: FunctionValue::FnMut(f),
        }
    }

    pub fn new(
        meta: FuncMeta,
        f: Box<
            dyn Fn(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ) -> Self {
        Self {
            meta,
            f: FunctionValue::Fn(f),
        }
    }
}
