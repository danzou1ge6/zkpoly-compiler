use std::fmt::Write;

use crate::ast::lowering::UserFunctionId;
use crate::ast::{self, PolyInit};
use crate::transit::type2::{self, NttAlgorithm};
use crate::transit::type3::{self, RegisterId};
use std::collections::{BTreeMap, HashMap};
use zkpoly_common::arith::{self, BinOp, UnrOp};
use zkpoly_common::heap::Heap;
use zkpoly_common::load_dynamic::Libs;
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
use zkpoly_runtime::functions::{
    FuncMeta, FunctionId, FunctionTable, FusedKernelMeta, KernelType, RegisteredFunction
};

use super::super::template::InstructionNode;
use super::super::{Chunk, InstructionIndex, VertexNode};

pub fn kernel_type_from_vertex(
    vertex: &VertexNode,
    fuse_meta: Option<&FusedKernelMeta>,
) -> Option<KernelType> {
    assert!(!vertex.unexpcted_during_kernel_gen());
    match vertex {
        VertexNode::Arith { .. } => Some(KernelType::FusedArith(fuse_meta.unwrap().clone())),
        VertexNode::Ntt { alg, .. } => match alg {
            NttAlgorithm::Standard { .. } => Some(KernelType::NttRecompute),
            NttAlgorithm::Precomputed { .. } => Some(KernelType::NttPrcompute),
            NttAlgorithm::Undecieded => panic!("NttAlgorithm should be decided at this point"),
        },
        VertexNode::Msm { alg, .. } => Some(KernelType::Msm(alg.clone())),
        VertexNode::KateDivision(..) => Some(KernelType::KateDivision),
        VertexNode::EvaluatePoly { .. } => Some(KernelType::EvaluatePoly),
        VertexNode::BatchedInvert(..) => Some(KernelType::BatchedInvert),
        VertexNode::ScanMul { .. } => Some(KernelType::ScanMul),
        VertexNode::Interpolate { .. } => Some(KernelType::Interpolate),
        VertexNode::AssmblePoly(_, _) => Some(KernelType::AssmblePoly),
        VertexNode::HashTranscript { typ, .. } => match typ {
            &crate::transit::HashTyp::WriteProof => Some(KernelType::HashTranscriptWrite),
            &crate::transit::HashTyp::NoWriteProof => Some(KernelType::HashTranscript),
        },
        VertexNode::SqueezeScalar(_) => Some(KernelType::SqueezeScalar),
        VertexNode::UserFunction(id, _) => Some(KernelType::UserFunction(*id)),
        VertexNode::DistributePowers { .. } => Some(KernelType::DistributePowers),
        VertexNode::ScalarInvert { .. } => Some(KernelType::ScalarInvert),
        VertexNode::SingleArith(arith) => match arith {
            zkpoly_common::arith::Arith::Bin(BinOp::Pp(op), _, _) => match op {
                zkpoly_common::arith::ArithBinOp::Add => Some(KernelType::PolyAdd),
                zkpoly_common::arith::ArithBinOp::Sub => Some(KernelType::PolySub),
                _ => unreachable!(
                    "div and mul can't have different len, so they should be handled in type2"
                ),
            },
            zkpoly_common::arith::Arith::Unr(UnrOp::S(op), _) => match op {
                zkpoly_common::arith::ArithUnrOp::Pow(exp) => Some(KernelType::ScalarPow(*exp)),
                _ => unreachable!("unr inv and neg should be tackled in type2"),
            },
            _ => unreachable!("most arith should be handled by fused kernels"),
        },
        _ => None,
    }
}
pub fn kernel_type_from_inst(
    inst: &type3::Instruction,
    fuse_meta: Option<&FusedKernelMeta>,
) -> Option<KernelType> {
    match &inst.node {
        type3::InstructionNode::FillPoly { init, pty, .. } => match (init, pty) {
            (PolyInit::Zeros, _) => Some(KernelType::NewZero),
            (PolyInit::Ones, PolyType::Lagrange) => Some(KernelType::NewOneLagrange),
            (PolyInit::Ones, PolyType::Coef) => Some(KernelType::NewOneCoef),
        },
        type3::InstructionNode::Type2 { vertex, .. } => kernel_type_from_vertex(vertex, fuse_meta),
        _ => None,
    }
}

const FUSED_PERFIX: &str = "fused_arith_";

pub fn gen_fused_kernels<'s, Rt: RuntimeType>(
    program: &Chunk<'s, Rt>,
    reg_id2var_id: &impl Fn(RegisterId) -> VariableId,
) -> BTreeMap<InstructionIndex, FusedKernelMeta> {
    let mut cache = HashMap::new();
    let mut inst_idx2name = BTreeMap::new();

    // first pass to generate fused arith kernels
    for (idx, instruct) in program.iter_instructions() {
        if let InstructionNode::Type2 { ids,vertex, .. } = &instruct.node {
            if let VertexNode::Arith { arith, .. } = vertex {
                let normalized = arith::hash::NormalizedDag::from(arith);
                let (name, op, included_indices) = cache.entry(normalized).or_insert_with(|| {
                    let arith = arith.relabeled(|r| reg_id2var_id(r));
                    let id: usize = idx.into();
                    let name = format!("{FUSED_PERFIX}{id}");

                    arith::check_degree_of_todo_vertices(name.to_string(), &arith);

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
                    let op = FusedOp::new(arith, name.clone(), outputs_i2o); // generate fused kernel
                    (name, op, vec![])
                    // let op = FusedOp::new(arith, name.clone()); // generate fused kernel
                    // (name, op, vec![])
                });
                included_indices.push(idx);
                inst_idx2name.insert(idx, FusedKernelMeta { name: name.to_string(), num_vars: op.vars.len(), num_mut_vars: op.mut_vars.len() });
            }
        }
    }

    cache.into_iter().for_each(|(_, (_, op, indices))| {
        let mut anno = String::new();
        write!(&mut anno, "// Included names: ").unwrap();
        for name in indices {
            write!(&mut anno, "{}, ", usize::from(name)).unwrap();
        }
        anno.push('\n');
        op.gen(anno);
    });

    inst_idx2name
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
    id: UserFunctionId,
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
            zkpoly_runtime::functions::Function::new_once(
                FuncMeta::new(name, KernelType::UserFunction(id)),
                Box::new(rust_func),
            )
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

            zkpoly_runtime::functions::Function::new_once(
                FuncMeta::new(name, KernelType::UserFunction(id)),
                Box::new(rust_func),
            )
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
            zkpoly_runtime::functions::Function::new(
                FuncMeta::new(name, KernelType::UserFunction(id)),
                Box::new(rust_func),
            )
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
    let inst_idx2fused_name = gen_fused_kernels(program, reg_id2var_id);
    let mut uf_table: Heap<UserFunctionId, _> = user_ftable.map(&mut (|_, f| Some(f)));

    let mut inst2func = BTreeMap::new();
    let mut kernel2func: BTreeMap<KernelType, FunctionId> = BTreeMap::new();
    for (id, instruct) in program.iter_instructions() {
        let fuse_name = inst_idx2fused_name.get(&id);
        let kernel_type = kernel_type_from_inst(instruct, fuse_name);
        if kernel_type.is_none() {
            continue;
        }
        let kernel_type = kernel_type.unwrap();
        if kernel2func.contains_key(&kernel_type) {
            inst2func.insert(id, kernel2func[&kernel_type]);
            continue;
        }
        let func = load_function(&kernel_type, libs, &mut uf_table);
        let func_id = f_table.push(func);
        kernel2func.insert(kernel_type, func_id);
        inst2func.insert(id, func_id);
    }

    GeneratedFunctions {
        inst2fid: inst2func,
    }
}

pub fn load_function<Rt: RuntimeType>(
    kernel_type: &KernelType,
    libs: &mut Libs,
    uf_table: &mut Heap<UserFunctionId, Option<type2::user_function::Function<Rt>>>,
) -> zkpoly_runtime::functions::Function<Rt> {
    match &kernel_type {
        KernelType::NttPrcompute => {
            let precompute_ntt = SsipNtt::new(libs);
            precompute_ntt.get_fn()
        }
        KernelType::NttRecompute => {
            let recompute_ntt = RecomputeNtt::new(libs);
            recompute_ntt.get_fn()
        }
        KernelType::Msm(config) => {
            let msm = MSM::new(libs, (*config).clone());
            msm.get_fn()
        }
        KernelType::BatchedInvert => {
            let batched_invert = PolyInvert::new(libs);
            batched_invert.get_fn()
        }
        KernelType::KateDivision => {
            let kate_division = KateDivision::new(libs);
            kate_division.get_fn()
        }
        KernelType::EvaluatePoly => {
            let eval_poly = PolyEval::new(libs);
            eval_poly.get_fn()
        }
        KernelType::ScanMul => {
            let scan_mul = PolyScan::new(libs);
            scan_mul.get_fn()
        }
        KernelType::FusedArith(meta) => {
            let fuse_kernel = FusedKernel::new(libs, meta.clone());
            fuse_kernel.get_fn()
        }
        KernelType::Interpolate => {
            let interpolate = InterpolateKernel::new();
            interpolate.get_fn()
        }
        KernelType::AssmblePoly => {
            let assemble_poly = AssmblePoly::new();
            assemble_poly.get_fn()
        }
        KernelType::HashTranscript => {
            let hash_transcript = HashTranscript::new();
            hash_transcript.get_fn()
        }
        KernelType::SqueezeScalar => {
            let squeeze = SqueezeScalar::new();
            squeeze.get_fn()
        }
        KernelType::UserFunction(uf_id) => {
            let func = uf_table[*uf_id].take().unwrap();
            convert_to_runtime_func(func.f, *uf_id)
        }
        KernelType::DistributePowers => {
            let distribute_powers = DistributePowers::new(libs);
            distribute_powers.get_fn()
        }
        KernelType::NewOneLagrange => {
            let new_one = PolyOneLagrange::new(libs);
            new_one.get_fn()
        }
        KernelType::NewOneCoef => {
            let new_one = PolyOneCoef::new(libs);
            new_one.get_fn()
        }
        KernelType::NewZero => {
            let new_zero = PolyZero::new(libs);
            new_zero.get_fn()
        }
        KernelType::HashTranscriptWrite => {
            let hash_transcript_write = HashTranscriptWrite::new();
            hash_transcript_write.get_fn()
        }
        KernelType::ScalarInvert => {
            let scalar_inv = ScalarInv::new(libs);
            scalar_inv.get_fn()
        }
        KernelType::ScalarPow(exp) => {
            let scalar_pow = ScalarPow::new(libs, *exp);
            scalar_pow.get_fn()
        }
        KernelType::PolyAdd => {
            let poly_add = PolyAdd::new(libs);
            poly_add.get_fn()
        }
        KernelType::PolySub => {
            let poly_sub = PolySub::new(libs);
            poly_sub.get_fn()
        }
        KernelType::Other => unreachable!("should be registered in kernel type"),
    }
}
