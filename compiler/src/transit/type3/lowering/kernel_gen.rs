use crate::ast::lowering::UserFunctionId;
use crate::ast::{self, PolyInit};
use crate::transit::type2::{self, NttAlgorithm};
use crate::transit::type3::{self, RegisterId};
use std::collections::BTreeMap;
use std::sync::Mutex;
use zkpoly_common::arith::{BinOp, UnrOp};
use zkpoly_common::heap::Heap;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_common::msm_config::MsmConfig;
use zkpoly_common::typ::PolyType;
use zkpoly_core::cpu_kernels::{
    AssmblePoly, HashTranscript, HashTranscriptWrite, InterpolateKernel, SqueezeScalar,
};
use zkpoly_core::fused_kernels::{FusedKernel, FusedOp};
use zkpoly_core::msm::MSM;
use zkpoly_core::ntt::{DistributePowers, RecomputeNtt, SsipNtt};
use zkpoly_core::poly::{
    KateDivision, PolyAdd, PolyEval, PolyInvert, PolyOneCoef, PolyOneLagrange, PolyScan, PolySub,
    PolyZero, ScalarInv, ScalarPow,
};
use zkpoly_runtime::args::{RuntimeType, Variable, VariableId};
use zkpoly_runtime::error::RuntimeError;
use zkpoly_runtime::functions::{FunctionId, FunctionTable, RegisteredFunction};

use super::super::template::InstructionNode;
use super::super::{Chunk, InstructionIndex, VertexNode};

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum KernelType {
    NttPrcompute,
    NttRecompute,
    Msm(MsmConfig),
    BatchedInvert,
    KateDivision,
    EvaluatePoly,
    ScanMul,
    FusedArith,
    Interpolate,
    AssmblePoly,
    HashTranscript,
    HashTranscriptWrite,
    SqueezeScalar,
    UserFunction(type2::user_function::Id),
    DistributePowers,
    NewOneLagrange,
    NewOneCoef,
    NewZero,
    ScalarInvert,   // TODO: support both cpu and gpu
    ScalarPow(u64), // TODO: support both cpu and gpu
    // there three kernels are mainly for input with different len
    // which auto generated kernels can't handle
    PolyAdd,
    PolySub,
}

impl KernelType {
    pub fn from_vertex(vertex: &VertexNode) -> Option<Self> {
        assert!(!vertex.unexpcted_during_kernel_gen());
        match vertex {
            VertexNode::Arith { .. } => Some(Self::FusedArith),
            VertexNode::Ntt { alg, .. } => match alg {
                NttAlgorithm::Standard { .. } => Some(Self::NttRecompute),
                NttAlgorithm::Precomputed { .. } => Some(Self::NttPrcompute),
                NttAlgorithm::Undecieded => panic!("NttAlgorithm should be decided at this point"),
            },
            VertexNode::Msm { alg, .. } => Some(Self::Msm(alg.clone())),
            VertexNode::KateDivision(..) => Some(Self::KateDivision),
            VertexNode::EvaluatePoly { .. } => Some(Self::EvaluatePoly),
            VertexNode::BatchedInvert(..) => Some(Self::BatchedInvert),
            VertexNode::ScanMul { .. } => Some(Self::ScanMul),
            VertexNode::Interpolate { .. } => Some(Self::Interpolate),
            VertexNode::AssmblePoly(_, _) => Some(Self::AssmblePoly),
            VertexNode::HashTranscript { typ, .. } => match typ {
                &crate::transit::HashTyp::WriteProof => Some(Self::HashTranscriptWrite),
                &crate::transit::HashTyp::NoWriteProof => Some(Self::HashTranscript),
            },
            VertexNode::SqueezeScalar(_) => Some(Self::SqueezeScalar),
            VertexNode::UserFunction(id, _) => Some(Self::UserFunction(*id)),
            VertexNode::DistributePowers { .. } => Some(Self::DistributePowers),
            VertexNode::ScalarInvert { .. } => Some(Self::ScalarInvert),
            VertexNode::SingleArith(arith) => match arith {
                zkpoly_common::arith::Arith::Bin(BinOp::Pp(op), _, _) => match op {
                    zkpoly_common::arith::ArithBinOp::Add => Some(Self::PolyAdd),
                    zkpoly_common::arith::ArithBinOp::Sub => Some(Self::PolySub),
                    _ => unreachable!(
                        "div and mul can't have different len, so they should be handled in type2"
                    ),
                },
                zkpoly_common::arith::Arith::Unr(UnrOp::S(op), _) => match op {
                    zkpoly_common::arith::ArithUnrOp::Pow(exp) => Some(Self::ScalarPow(*exp)),
                    _ => unreachable!("unr inv and neg should be tackled in type2"),
                },
                _ => unreachable!("most arith should be handled by fused kernels"),
            },
            _ => None,
        }
    }
    pub fn from_inst(inst: &type3::Instruction) -> Option<Self> {
        match &inst.node {
            type3::InstructionNode::FillPoly { init, pty, .. } => match (init, pty) {
                (PolyInit::Zeros, _) => Some(Self::NewZero),
                (PolyInit::Ones, PolyType::Lagrange) => Some(Self::NewOneLagrange),
                (PolyInit::Ones, PolyType::Coef) => Some(Self::NewOneCoef),
            },
            type3::InstructionNode::Type2 { vertex, .. } => Self::from_vertex(vertex),
            _ => None,
        }
    }
}

const FUSED_PERFIX: &str = "fused_arith_";

pub fn gen_fused_kernels<'s, Rt: RuntimeType>(
    program: &Chunk<'s, Rt>,
    reg_id2var_id: &impl Fn(RegisterId) -> VariableId,
) {
    // first pass to generate fused arith kernels
    for (id, instruct) in program.iter_instructions() {
        if let InstructionNode::Type2 { ids, vertex, .. } = &instruct.node {
            if let VertexNode::Arith { arith, .. } = vertex {
                let arith = arith.relabeled(|r| reg_id2var_id(r));
                let id: usize = id.into();
                let name = format!("{FUSED_PERFIX}{id}");
                let outputs_i2o = arith
                    .outputs
                    .iter()
                    .copied()
                    .zip((*ids).clone().into_iter().map(|(ra, rb)| {
                        // see def in InstructionNode::Type2
                        if rb.is_some() {
                            reg_id2var_id(rb.unwrap()) // in-place should reuse the input variable
                        } else {
                            reg_id2var_id(ra) // otherwise, create a new variable
                        }
                    }))
                    .collect();
                FusedOp::new(arith, name, outputs_i2o).gen(); // generate fused kernel
            }
        }
    }
}

pub struct GeneratedFunctions {
    inst2fid: BTreeMap<InstructionIndex, FunctionId>,
}

impl GeneratedFunctions {
    pub fn at(&self, idx: InstructionIndex) -> FunctionId {
        self.inst2fid
            .get(&idx)
            .unwrap_or_else(|| panic!("error at {:?}", idx))
            .clone()
    }
}

fn assemble_tuple<Rt: RuntimeType>(mut_var: Vec<&mut Variable<Rt>>) -> Variable<Rt> {
    let mut var = Vec::new();
    for v in mut_var {
        var.push((*v).clone());
    }
    Variable::Tuple(var)
}

fn convert_to_runtime_func<Rt: RuntimeType>(
    func: ast::user_function::Function<Rt>,
) -> zkpoly_runtime::functions::Function<Rt> {
    let name = func.name.clone();
    let n_args = func.n_args;
    let need_assemble = match &func.ret_typ {
        type2::typ::template::Typ::Tuple(..) => true,
        type2::typ::template::Typ::Array(..) => true,
        _ => false,
    };
    let ret_type = func.ret_typ.clone();
    match func.value {
        ast::user_function::Value::Mut(mut fn_mut) => {
            let rust_func = move |mut mut_var: Vec<&mut Variable<Rt>>,
                                  var: Vec<&Variable<Rt>>|
                  -> Result<(), RuntimeError> {
                assert_eq!(var.len(), n_args);
                if need_assemble {
                    let mut ret = assemble_tuple(mut_var);
                    ret_type.match_arg(&ret);
                    fn_mut(&mut ret, var)
                } else {
                    assert_eq!(mut_var.len(), 1);
                    ret_type.match_arg(&mut_var[0]);
                    fn_mut(mut_var[0], var)
                }
            };
            zkpoly_runtime::functions::Function {
                name,
                f: zkpoly_runtime::functions::FunctionValue::FnMut(Mutex::new(Box::new(rust_func))),
            }
        }
        ast::user_function::Value::Once(fn_once) => {
            let rust_func = move |mut mut_var: Vec<&mut Variable<Rt>>,
                                  var: Vec<&Variable<Rt>>|
                  -> Result<(), RuntimeError> {
                assert_eq!(var.len(), n_args);
                if need_assemble {
                    let mut ret = assemble_tuple(mut_var);
                    ret_type.match_arg(&ret);
                    fn_once(&mut ret, var)
                } else {
                    assert_eq!(mut_var.len(), 1);
                    ret_type.match_arg(&mut_var[0]);
                    fn_once(mut_var[0], var)
                }
            };
            zkpoly_runtime::functions::Function {
                name,
                f: zkpoly_runtime::functions::FunctionValue::FnOnce(Mutex::new(Some(Box::new(
                    rust_func,
                )))),
            }
        }
        ast::user_function::Value::Fn(f) => {
            let rust_func = move |mut mut_var: Vec<&mut Variable<Rt>>,
                                  var: Vec<&Variable<Rt>>|
                  -> Result<(), RuntimeError> {
                assert_eq!(var.len(), n_args);
                if need_assemble {
                    let mut ret = assemble_tuple(mut_var);
                    ret_type.match_arg(&ret);
                    f(&mut ret, var)
                } else {
                    assert_eq!(mut_var.len(), 1);
                    ret_type.match_arg(&mut_var[0]);
                    f(mut_var[0], var)
                }
            };
            zkpoly_runtime::functions::Function {
                name,
                f: zkpoly_runtime::functions::FunctionValue::Fn(Box::new(rust_func)),
            }
        }
    }
}

pub fn get_function_id<'s, Rt: RuntimeType>(
    f_table: &mut FunctionTable<Rt>,
    program: &Chunk<'s, Rt>,
    user_ftable: type2::user_function::Table<Rt>,
    reg_id2var_id: &impl Fn(RegisterId) -> VariableId,
    libs: &mut Libs,
) -> GeneratedFunctions {
    gen_fused_kernels(program, reg_id2var_id);

    let mut uf_table: Heap<UserFunctionId, _> = user_ftable.map(&mut (|_, f| Some(f)));

    let mut inst2func = BTreeMap::new();
    let mut kernel2func: BTreeMap<KernelType, FunctionId> = BTreeMap::new();
    for (id, instruct) in program.iter_instructions() {
        let kernel_type = KernelType::from_inst(instruct);
        if kernel_type.is_none() {
            continue;
        }
        let kernel_type = kernel_type.unwrap();
        if kernel_type != KernelType::FusedArith && kernel2func.contains_key(&kernel_type) {
            inst2func.insert(id, kernel2func[&kernel_type]);
            continue;
        }
        match &kernel_type {
            KernelType::NttPrcompute => {
                let precompute_ntt = SsipNtt::new(libs);
                let func_id = f_table.push(precompute_ntt.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::NttRecompute => {
                let recompute_ntt = RecomputeNtt::new(libs);
                let func_id = f_table.push(recompute_ntt.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::Msm(config) => {
                let msm = MSM::new(libs, (*config).clone());
                let func_id = f_table.push(msm.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::BatchedInvert => {
                let batched_invert = PolyInvert::new(libs);
                let func_id = f_table.push(batched_invert.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::KateDivision => {
                let kate_division = KateDivision::new(libs);
                let func_id = f_table.push(kate_division.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::EvaluatePoly => {
                let eval_poly = PolyEval::new(libs);
                let func_id = f_table.push(eval_poly.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::ScanMul => {
                let scan_mul = PolyScan::new(libs);
                let func_id = f_table.push(scan_mul.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::FusedArith => {
                let id_usize: usize = id.into();
                let name = format!("{FUSED_PERFIX}{id_usize}");
                let fuse_kernel = FusedKernel::new(libs, name);
                let func_id = f_table.push(fuse_kernel.get_fn());
                inst2func.insert(id, func_id);
            }
            KernelType::Interpolate => {
                let interpolate = InterpolateKernel::new();
                let func_id = f_table.push(interpolate.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::AssmblePoly => {
                let assemble_poly = AssmblePoly::new();
                let func_id = f_table.push(assemble_poly.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::HashTranscript => {
                let hash_transcript = HashTranscript::new();
                let func_id = f_table.push(hash_transcript.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::SqueezeScalar => {
                let squeeze = SqueezeScalar::new();
                let func_id = f_table.push(squeeze.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::UserFunction(uf_id) => {
                let func = uf_table[*uf_id].take().unwrap();
                let func_id = f_table.push(convert_to_runtime_func(func.f));
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::DistributePowers => {
                let distribute_powers = DistributePowers::new(libs);
                let func_id = f_table.push(distribute_powers.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::NewOneLagrange => {
                let new_one = PolyOneLagrange::new(libs);
                let func_id = f_table.push(new_one.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::NewOneCoef => {
                let new_one = PolyOneCoef::new(libs);
                let func_id = f_table.push(new_one.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::NewZero => {
                let new_zero = PolyZero::new(libs);
                let func_id = f_table.push(new_zero.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::HashTranscriptWrite => {
                let hash_transcript_write = HashTranscriptWrite::new();
                let func_id = f_table.push(hash_transcript_write.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::ScalarInvert => {
                let scalar_inv = ScalarInv::new(libs);
                let func_id = f_table.push(scalar_inv.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::ScalarPow(exp) => {
                let scalar_pow = ScalarPow::new(libs, *exp);
                let func_id = f_table.push(scalar_pow.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::PolyAdd => {
                let poly_add = PolyAdd::new(libs);
                let func_id = f_table.push(poly_add.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
            KernelType::PolySub => {
                let poly_sub = PolySub::new(libs);
                let func_id = f_table.push(poly_sub.get_fn());
                kernel2func.insert(kernel_type, func_id);
                inst2func.insert(id, func_id);
            }
        }
    }

    GeneratedFunctions {
        inst2fid: inst2func,
    }
}
