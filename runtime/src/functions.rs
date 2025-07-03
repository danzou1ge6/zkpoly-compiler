use std::sync::Arc;

use crate::args::{RuntimeType, Variable};
use crate::error::RuntimeError;
use serde::{Deserialize, Serialize};
use zkpoly_common::devices::DeviceType;
use zkpoly_common::heap;
use zkpoly_common::msm_config::MsmConfig;

zkpoly_common::define_usize_id!(FunctionId);
zkpoly_common::define_usize_id!(UserFunctionId);

pub type FunctionTable<T> = heap::Heap<FunctionId, Function<T>>;

pub trait RegisteredFunction<T: RuntimeType> {
    fn get_fn(&self) -> Function<T>;
}

pub type Closure<T> = Arc<
    dyn Fn(Vec<&mut Variable<T>>, Vec<&Variable<T>>, Arc<dyn Fn(i32) -> i32 + Send + Sync>) -> Result<(), RuntimeError>
        + Sync
        + Send
        + 'static,
>;

#[derive(Clone)]
pub struct Function<T: RuntimeType> {
    pub meta: FuncMeta,
    pub f: Closure<T>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd, Eq, Ord)]
pub struct FusedKernelMeta {
    pub name: String,
    pub num_vars: usize,
    pub num_mut_vars: usize,
    pub pipelined_meta: Option<PipelinedMeta>,
    pub device: DeviceType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, PartialOrd, Eq, Ord)]
pub struct PipelinedMeta {
    pub divide_parts: usize, // how many parts to divide the poly into, must > 3 and later calls must have len which is a multiple of this
    pub num_scalars: usize,
    pub num_mut_scalars: usize,
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KernelType {
    NttPrcompute,
    NttRecompute,
    Msm(MsmConfig),
    BatchedInvert,
    KateDivision,
    EvaluatePoly,
    PolyPermute,
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
        st.finish()
    }
}

impl<T: RuntimeType> Function<T> {
    pub fn new(
        meta: FuncMeta,
        f: Arc<
            dyn Fn(Vec<&mut Variable<T>>, Vec<&Variable<T>>, Arc<dyn Fn(i32) -> i32 + Send + Sync>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ) -> Self {
        Self { meta, f }
    }
}
