use zkpoly_runtime::{
    args::{RuntimeType, VariableId},
    devices::ThreadId,
    functions::{FunctionId, FunctionTable},
    instructions::Instruction,
};

pub fn generate<'s, Rt: RuntimeType>(
    outputs: &[super::RegisterId],
    temp: Option<super::RegisterId>,
    vertex: &super::VertexNode,
    t3chunk: &super::Chunk<'s, Rt>,
    thread: ThreadId,
    reg_id2var_id: &impl Fn(super::RegisterId) -> VariableId,
    f_table: &mut FunctionTable<Rt>,
    emit: &mut impl FnMut(Instruction),
) {
    todo!("Compile function for vertex and push function to function table, then emit function call instruction")
}
