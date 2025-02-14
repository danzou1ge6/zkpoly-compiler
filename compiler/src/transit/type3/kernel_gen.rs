use std::collections::{BTreeMap, HashMap};
use zkpoly_runtime::functions::{FunctionId, FunctionTable, Function};
use zkpoly_runtime::args::RuntimeType;
use super::{Chunk, VertexNode};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum KernelType {
    Ntt,
    BatchedInvert,
    KateDivision,
    // 添加其他kernel类型...
}

impl KernelType {
    pub fn from_vertex(vertex: &VertexNode) -> Option<Self> {
        // TODO: 实现从vertex到KernelType的映射
        None
    }

}

pub fn get_function_id<Rt: RuntimeType>(f_table: &mut FunctionTable<Rt>) -> BTreeMap<KernelType, FunctionId> {
    todo!()
}