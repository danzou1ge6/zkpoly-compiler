use crate::transit::type3::{Device, InstructionIndex};

use super::super::{RegisterId, VertexNode};
use super::{Stream, StreamSpecific, Track};
use zkpoly_common::arith::{self, ArithUnrOp, BinOp, UnrOp};
use zkpoly_core::fused_kernels::gen_var_lists;
use zkpoly_runtime::args::RuntimeType;
use zkpoly_runtime::{args::VariableId, functions::FunctionId, instructions::Instruction};

pub fn emit_func<'s, Rt: RuntimeType>(
    t3idx: InstructionIndex,
    outputs: &[RegisterId], // output registers
    temp: &[RegisterId],    // temporary register to store intermediate results
    track: Track,
    vertex: &VertexNode, // vertex node to generate kernel for
    reg_id2var_id: &impl Fn(RegisterId) -> VariableId, // function to get variable id from register id
    stream2variable_id: &StreamSpecific<VariableId>,
    f_id: FunctionId,
    t3chunk: &crate::transit::type3::Chunk<'s, Rt>,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    let stream = Stream::of_track(track).map(|t| stream2variable_id.get(t).clone());
    match vertex {
        VertexNode::Arith { arith, .. } => {
            let (var, var_mut) = gen_var_lists(outputs.iter().copied(), arith);
            let mut arg = vec![stream.unwrap()];
            for (_, id) in var.iter() {
                arg.push(reg_id2var_id(*id));
            }
            let arg_mut = var_mut
                .iter()
                .map(|(_, id)| reg_id2var_id(*id))
                .collect::<Vec<_>>();
            generate_arith(arg, arg_mut, f_id, emit);
        }
        VertexNode::Ntt {
            s,
            to: _,
            from: _,
            alg,
        } => {
            let poly = reg_id2var_id(*s);

            match alg {
                crate::transit::type2::NttAlgorithm::Precomputed(twiddle) => {
                    let twiddle = reg_id2var_id(*twiddle);
                    generate_ntt_precompute(poly, twiddle, stream.unwrap(), f_id, emit);
                }
                crate::transit::type2::NttAlgorithm::Standard { pq, omega } => {
                    let pq = reg_id2var_id(*pq);
                    let omega = reg_id2var_id(*omega);
                    generate_ntt_recompute(poly, pq, omega, stream.unwrap(), f_id, emit);
                }
                crate::transit::type2::NttAlgorithm::Undecieded => {
                    panic!("NttAlgorithm must have been decided during precomputation");
                }
            }
        }
        VertexNode::Msm {
            polys: scalars,
            points,
            alg: _alg,
        } => {
            let scalar_batch = scalars
                .iter()
                .map(|id| reg_id2var_id(*id))
                .collect::<Vec<_>>();
            let point_batch = points
                .iter()
                .map(|id| reg_id2var_id(*id))
                .collect::<Vec<_>>();

            let temp_buffers = temp.iter().map(|id| reg_id2var_id(*id)).collect::<Vec<_>>();
            let answers = outputs
                .iter()
                .map(|id| reg_id2var_id(*id))
                .collect::<Vec<_>>();
            generate_msm(
                &point_batch,
                &scalar_batch,
                &temp_buffers,
                &answers,
                f_id,
                emit,
            );
        }
        VertexNode::KateDivision(poly, b) => {
            let poly = reg_id2var_id(*poly);
            let b = reg_id2var_id(*b);
            let temp = reg_id2var_id(temp[0]);
            let res = reg_id2var_id(outputs[0]);
            generate_kate_division(poly, res, b, temp, stream.unwrap(), f_id, emit);
        }
        VertexNode::EvaluatePoly { poly, at } => {
            let poly = reg_id2var_id(*poly);
            let x = reg_id2var_id(*at);
            let temp = reg_id2var_id(temp[0]);
            let res = reg_id2var_id(outputs[0]);
            generate_eval_poly(temp, res, poly, x, stream.unwrap(), f_id, emit);
        }
        VertexNode::BatchedInvert(poly) => {
            let poly = reg_id2var_id(*poly);
            let temp = reg_id2var_id(temp[0]);
            let inv = reg_id2var_id(outputs[0]);
            generate_batched_invert(poly, temp, inv, stream.unwrap(), f_id, emit);
        }
        VertexNode::ScanMul { poly, x0 } => {
            let x0 = reg_id2var_id(*x0);
            let poly = reg_id2var_id(*poly);
            let temp = reg_id2var_id(temp[0]);
            let res = reg_id2var_id(outputs[0]);
            generate_scan_mul(poly, temp, res, x0, stream.unwrap(), f_id, emit);
        }
        VertexNode::AssmblePoly(_, scalars) => {
            let target = reg_id2var_id(outputs[0]);
            emit(Instruction::FuncCall {
                func_id: f_id,
                arg_mut: vec![target],
                arg: scalars.iter().map(|r| reg_id2var_id(*r)).collect(),
            });
        }
        VertexNode::DistributePowers { poly, powers } => {
            let poly = reg_id2var_id(*poly);
            let powers = reg_id2var_id(*powers);
            emit(Instruction::FuncCall {
                func_id: f_id,
                arg_mut: vec![poly],
                arg: vec![powers],
            });
        }
        VertexNode::HashTranscript {
            transcript, value, ..
        } => {
            let transcript = reg_id2var_id(*transcript);
            let value = reg_id2var_id(*value);
            emit(Instruction::FuncCall {
                func_id: f_id,
                arg_mut: vec![transcript],
                arg: vec![value],
            });
        }
        VertexNode::Interpolate { xs, ys } => {
            let xs = xs.iter().map(|id| reg_id2var_id(*id)).collect::<Vec<_>>();
            let ys = ys.iter().map(|id| reg_id2var_id(*id)).collect::<Vec<_>>();
            let target = reg_id2var_id(outputs[0]);
            emit(Instruction::FuncCall {
                func_id: f_id,
                arg_mut: vec![target],
                arg: xs.into_iter().chain(ys).collect(),
            });
        }
        VertexNode::SqueezeScalar(transcript) => {
            let transcript = reg_id2var_id(*transcript);
            let out_scalar = reg_id2var_id(outputs[0]);
            emit(Instruction::FuncCall {
                func_id: f_id,
                arg_mut: vec![out_scalar],
                arg: vec![transcript],
            })
        }
        VertexNode::NewPoly(..) => {
            let device = t3chunk.register_devices[&outputs[0]];
            let dst = reg_id2var_id(outputs[0]);
            if device == Device::Cpu {
                emit(Instruction::FuncCall {
                    func_id: f_id,
                    arg_mut: vec![dst],
                    arg: vec![],
                });
            } else {
                emit(Instruction::FuncCall {
                    func_id: f_id,
                    arg_mut: vec![dst],
                    arg: vec![stream.unwrap()],
                });
            }
        }
        VertexNode::ScalarInvert { val } => {
            let device = t3chunk.register_devices[&outputs[0]];
            let target = reg_id2var_id(*val);
            if device == Device::Cpu {
                emit(Instruction::FuncCall {
                    func_id: f_id,
                    arg_mut: vec![target],
                    arg: vec![],
                });
            } else {
                emit(Instruction::FuncCall {
                    func_id: f_id,
                    arg_mut: vec![target],
                    arg: vec![stream.unwrap()],
                });
            }
        }
        VertexNode::UserFunction(_, args) => {
            let args = args.iter().map(|id| reg_id2var_id(*id)).collect::<Vec<_>>();
            let targets = outputs
                .iter()
                .map(|id| reg_id2var_id(*id))
                .collect::<Vec<_>>();

            emit(Instruction::FuncCall {
                func_id: f_id,
                arg_mut: targets,
                arg: args,
            });
        }
        VertexNode::SingleArith(arith) => match arith {
            arith::Arith::Bin(BinOp::Pp(op), lhs, rhs) => match op {
                arith::ArithBinOp::Add => {
                    let lhs = reg_id2var_id(*lhs);
                    let rhs = reg_id2var_id(*rhs);
                    let target = reg_id2var_id(outputs[0]);
                    let stream = stream.unwrap();
                    emit(Instruction::FuncCall {
                        func_id: f_id,
                        arg_mut: vec![target],
                        arg: vec![lhs, rhs, stream],
                    });
                }
                arith::ArithBinOp::Sub => {
                    let lhs = reg_id2var_id(*lhs);
                    let rhs = reg_id2var_id(*rhs);
                    let target = reg_id2var_id(outputs[0]);
                    let stream = stream.unwrap();
                    emit(Instruction::FuncCall {
                        func_id: f_id,
                        arg_mut: vec![target],
                        arg: vec![lhs, rhs, stream],
                    });
                }
                _ => unreachable!(),
            },
            arith::Arith::Unr(UnrOp::S(ArithUnrOp::Pow(_)), target) => {
                let target = reg_id2var_id(*target);
                let device = t3chunk.register_devices[&outputs[0]];
                if device == Device::Cpu {
                    emit(Instruction::FuncCall {
                        func_id: f_id,
                        arg_mut: vec![target],
                        arg: vec![],
                    });
                } else {
                    emit(Instruction::FuncCall {
                        func_id: f_id,
                        arg_mut: vec![target],
                        arg: vec![stream.unwrap()],
                    });
                }
            }
            _ => unreachable!(),
        },
        _ => panic!("Unsupported vertex node at {:?}", t3idx),
    }
}

fn generate_ntt_precompute(
    poly: VariableId,
    twiddle: VariableId,
    stream: VariableId,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall {
        func_id,
        arg_mut: vec![poly],
        arg: vec![twiddle, stream],
    })
}

fn generate_ntt_recompute(
    poly: VariableId,
    pq: VariableId,
    omega: VariableId,
    stream: VariableId,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall {
        func_id,
        arg_mut: vec![poly],
        arg: vec![pq, omega, stream],
    })
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
    emit(Instruction::FuncCall {
        func_id,
        arg,
        arg_mut,
    })
}

fn generate_batched_invert(
    poly: VariableId,
    temp: VariableId,
    inv: VariableId, // the inverse of the mul of all elements in the poly
    stream: VariableId,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall {
        func_id,
        arg_mut: vec![temp, poly, inv],
        arg: vec![stream],
    })
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
    emit(Instruction::FuncCall {
        func_id,
        arg_mut: vec![temp, res],
        arg: vec![poly, b, stream],
    })
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
    emit(Instruction::FuncCall {
        func_id,
        arg_mut: vec![temp, res],
        arg: vec![poly, x0, stream],
    })
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
    emit(Instruction::FuncCall {
        func_id,
        arg_mut: vec![temp, res],
        arg: vec![poly, x, stream],
    })
}

fn generate_arith(
    var: Vec<VariableId>,
    var_mut: Vec<VariableId>,
    func_id: FunctionId,
    emit: &mut impl FnMut(Instruction), // function to emit instruction
) {
    emit(Instruction::FuncCall {
        func_id,
        arg_mut: var_mut,
        arg: var,
    })
}
