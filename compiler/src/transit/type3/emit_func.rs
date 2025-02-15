use crate::ast::template::VertexNode;
use zkpoly_common::msm_config::MsmConfig;
use zkpoly_core::fused_kernels::gen_var_lists;
use zkpoly_runtime::{
    args::{RuntimeType, VariableId},
    devices::ThreadId,
    functions::{FunctionId, FunctionTable},
    instructions::Instruction, scalar,
};

pub fn emit_func<'s, Rt: RuntimeType>(
    outputs: &[super::RegisterId],   // output registers
    temp: Vec<super::RegisterId>, // temporary register to store intermediate results
    stream: VariableId,              // stream variable
    vertex: &super::VertexNode,      // vertex node to generate kernel for
    t3chunk: &super::Chunk<'s, Rt>,  // the main program
    reg_id2var_id: &impl Fn(super::RegisterId) -> VariableId, // function to get variable id from register id
    f_id: FunctionId, // function id to generate
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    match vertex {
        VertexNode::Arith(arith, _) => {
            let (var, var_mut) = gen_var_lists(arith);
            let mut arg = vec![stream];
            for (_, id) in var.iter() {
                arg.push(reg_id2var_id(*id));
            }
            let arg_mut = var_mut.iter().map(|(_, id)| reg_id2var_id(*id)).collect::<Vec<_>>();
            generate_arith(arg, arg_mut, f_id, emit);
        },
        VertexNode::Ntt { s, to, from, alg, twiddle_factors } => {
            let poly = reg_id2var_id(*s);
            match alg {
                crate::transit::type2::NttAlgorithm::Precomputed{twiddle} => {
                    let twiddle = reg_id2var_id(*twiddle);
                    generate_ntt_precompute(poly, twiddle, stream, f_id, emit);
                }
                crate::transit::type2::NttAlgorithm::Standard{pq, omega} => {
                    let pq = reg_id2var_id(*pq);
                    let omega = reg_id2var_id(*omega);
                    generate_ntt_recompute(poly, pq, omega, stream, f_id, emit);
                }
            }
        }
        VertexNode::Msm { scalars, points, alg } => {
            let scalar_batch = scalars.iter().map(| id| reg_id2var_id(*id)).collect::<Vec<_>>();
            let point_batch = points.iter().map(| id| reg_id2var_id(*id)).collect::<Vec<_>>();
            let temp_buffers = temp.iter().map(| id| reg_id2var_id(*id)).collect::<Vec<_>>();
            let answers = outputs.iter().map(| id| reg_id2var_id(*id)).collect::<Vec<_>>();
            generate_msm(&point_batch, &scalar_batch, &temp_buffers, &answers, f_id, emit);
        }
        VertexNode::KateDivision(poly, b) => {
            let poly = reg_id2var_id(*poly);
            let b = reg_id2var_id(*b);
            let temp = reg_id2var_id(temp[0]);
            let res = reg_id2var_id(outputs[0]);
            generate_kate_division(poly, res, b, temp, stream, f_id, emit);
        }
        VertexNode::EvaluatePoly { poly, at } => {
            let poly = reg_id2var_id(*poly);
            let x = reg_id2var_id(*at);
            let temp = reg_id2var_id(temp[0]);
            let res = reg_id2var_id(outputs[0]);
            generate_eval_poly(temp, res, poly, x, stream, f_id, emit);
        }
        VertexNode::BatchedInvert(poly) => {
            let poly = reg_id2var_id(*poly);
            let temp = reg_id2var_id(temp[0]);
            let inv = reg_id2var_id(outputs[0]);
            generate_batched_invert(poly, temp, inv, stream, f_id, emit);
        }
        _ => unreachable!(),
    }
}

fn generate_ntt_precompute(
    poly: VariableId,
    twiddle: VariableId,
    stream: VariableId,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall { func_id, arg_mut: vec![poly], arg: vec![twiddle, stream] })
}

fn generate_ntt_recompute(
    poly: VariableId,
    pq: VariableId,
    omega: VariableId,
    stream: VariableId,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall { func_id, arg_mut: vec![poly], arg: vec![pq, omega, stream] })
}

fn generate_msm(
    precompute_points: &Vec<VariableId>,
    scalar_batch: &Vec<VariableId>,
    temp_buffers: &Vec<VariableId>,
    answers: &Vec<VariableId>,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    let mut arg = Vec::new();
    for i in 0..precompute_points.len() {
        arg.push(precompute_points[i]);
    }
    for i in 0..scalar_batch.len() {
        arg.push(scalar_batch[i]);
    }
    let mut arg_mut = Vec::new();
    for i in 0..temp_buffers.len() {
        arg_mut.push(temp_buffers[i]);
    }
    for i in 0..answers.len() {
        arg_mut.push(answers[i]);
    }
    emit(Instruction::FuncCall { func_id, arg, arg_mut })
}

fn generate_batched_invert(
    poly: VariableId,
    temp: VariableId,
    inv: VariableId, // the inverse of the mul of all elements in the poly
    stream: VariableId,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall { func_id, arg_mut: vec![temp, poly, inv], arg: vec![stream] })
}

fn generate_kate_division(
    poly: VariableId,
    res: VariableId,
    b: VariableId,
    temp: VariableId,
    stream: VariableId,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall { func_id, arg_mut: vec![temp, res], arg: vec![poly, b, stream] })
}

fn generate_scan_mul(
    poly: VariableId,
    temp: VariableId,
    res: VariableId,
    x0: VariableId,
    stream: VariableId,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall { func_id, arg_mut: vec![temp, res], arg: vec![poly, x0, stream] })
}

fn generate_eval_poly(
    temp: VariableId,
    res: VariableId,
    poly: VariableId,
    x: VariableId,
    stream: VariableId,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall { func_id, arg_mut: vec![temp, res], arg: vec![poly, x, stream] })
}

fn generate_arith(
    var: Vec<VariableId>,
    var_mut: Vec<VariableId>,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall { func_id, arg_mut: var_mut, arg: var })
}