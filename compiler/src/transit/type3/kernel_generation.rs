use zkpoly_runtime::{
    args::{RuntimeType, VariableId},
    devices::ThreadId,
    functions::{FunctionId, FunctionTable},
    instructions::Instruction,
};

pub fn generate<'s, Rt: RuntimeType>(
    t3idx: super::InstructionIndex,
    inst: &super::Instruction,
    t3chunk: &super::Chunk<'s, Rt>,
    thread: ThreadId,
    reg_id2var_id: &impl Fn(super::RegisterId) -> VariableId,
    f_table: &mut FunctionTable<Rt>,
    emit: &mut impl FnMut(Instruction),
) {
    unimplemented!()
}
