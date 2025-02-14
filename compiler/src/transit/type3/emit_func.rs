use zkpoly_runtime::{
    args::{RuntimeType, VariableId},
    devices::ThreadId,
    functions::{FunctionId, FunctionTable},
    instructions::Instruction,
};
use crate::ast::template::VertexNode;

use super::RegisterId;

pub fn generate<'s, Rt: RuntimeType>(
    outputs: &[super::RegisterId], // output registers
    temp: Option<super::RegisterId>, // temporary register to store intermediate results
    stream: VariableId, // stream variable
    vertex: &super::VertexNode, // vertex node to generate kernel for
    t3chunk: &super::Chunk<'s, Rt>, // the main program
    reg_id2var_id: &impl Fn(super::RegisterId) -> VariableId, // function to get variable id from register id
    f_table: &mut FunctionTable<Rt>,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    match vertex {
        _ => unimplemented!(),
    }
}

pub fn generate_ntt_standard<'s, Rt: RuntimeType>(
    poly: VariableId,
    omega: VariableId,
    stream: VariableId,
    reg_id2var_id: &impl Fn(super::RegisterId) -> VariableId, // function to get variable id from register id
    f_table: &mut FunctionTable<Rt>,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    
}