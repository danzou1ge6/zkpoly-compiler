use std::fmt::Write;
use std::path::PathBuf;
use std::sync::Arc;

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
use zkpoly_core::fused_kernels::{FusedKernel, FusedOp, PipelinedFusedKernel};
use zkpoly_core::msm::MSM;
use zkpoly_core::ntt::{DistributePowers, RecomputeNtt, SsipNtt};
use zkpoly_core::poly::{
    KateDivision, PolyAdd, PolyEval, PolyInvert, PolyOneCoef, PolyOneLagrange, PolyPermute,
    PolyScan, PolySub, PolyZero, ScalarInv, ScalarPow,
};
use zkpoly_runtime::args::{RuntimeType, Variable, VariableId};
use zkpoly_runtime::error::RuntimeError;
use zkpoly_runtime::functions::{
    FuncMeta, FunctionId, FunctionTable, FusedKernelMeta, KernelType, PipelinedMeta,
    RegisteredFunction,
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
        VertexNode::PolyPermute(_, _, _) => Some(KernelType::PolyPermute),
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
    target_path: Option<PathBuf>,
) -> BTreeMap<InstructionIndex, FusedKernelMeta> {
    let mut cache = HashMap::new();
    let mut inst_idx2fused_meta = BTreeMap::new();

    // first pass to generate fused arith kernels
    for (idx, instruct) in program.iter_instructions() {
        if let InstructionNode::Type2 { vertex, .. } = &instruct.node {
            if let VertexNode::Arith { arith, chunking } = vertex {
                let normalized = arith::hash::NormalizedDag::from(arith);
                let (name, op, included_indices) = cache.entry(normalized).or_insert_with(|| {
                    let arith = arith.relabeled(|r| reg_id2var_id(r));
                    let id: usize = idx.into();
                    let name = format!("{FUSED_PERFIX}{id}");

                    arith::check_degree_of_todo_vertices(name.to_string(), &arith);

                    let limbs = size_of::<Rt::Field>() / size_of::<u32>();
                    let op = FusedOp::new(arith, name.clone(), limbs); // generate fused kernel
                    (name, op, vec![])
                });
                included_indices.push(idx);
                let pipelined_meta = if let Some(chunking) = chunking {
                    let divide_parts = chunking.clone();
                    assert!(divide_parts > 3);
                    let (num_scalars, num_mut_scalars) = op.num_scalars();
                    Some(PipelinedMeta {
                        divide_parts: divide_parts as usize,
                        num_scalars,
                        num_mut_scalars,
                    })
                } else {
                    None
                };
                inst_idx2fused_meta.insert(
                    idx,
                    FusedKernelMeta {
                        name: name.to_string(),
                        num_vars: op.vars.len(),
                        num_mut_vars: op.mut_vars.len(),
                        pipelined_meta: pipelined_meta,
                        lib_path: target_path.clone(),
                    },
                );
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
        op.gen(anno, target_path.clone());
    });

    inst_idx2fused_meta
}

pub struct GeneratedFunctions {
    inst2fid: BTreeMap<InstructionIndex, FunctionId>,
    inst_idx2fused_meta: BTreeMap<InstructionIndex, FusedKernelMeta>,
}

impl GeneratedFunctions {
    pub fn at(&self, idx: InstructionIndex) -> (FunctionId, Option<FusedKernelMeta>) {
        let fid = self
            .inst2fid
            .get(&idx)
            .unwrap_or_else(|| panic!("error at {:?}", idx))
            .clone();
        let fused_meta_ref = self.inst_idx2fused_meta.get(&idx);
        let fused_meta = if fused_meta_ref.is_some() {
            Some(fused_meta_ref.unwrap().clone())
        } else {
            None
        };
        (fid, fused_meta)
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
    let f = func.value;
    let rust_func = move |mut mut_var: Vec<&mut Variable<Rt>>,
                          var: Vec<&Variable<Rt>>,
                          _: Arc<dyn Fn(i32) -> i32 + Send + Sync>|
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
        Arc::new(rust_func),
    )
}

pub fn get_function_id<'s, Rt: RuntimeType>(
    f_table: &mut FunctionTable<Rt>,
    program: &Chunk<'s, Rt>,
    user_ftable: type2::user_function::Table<Rt>,
    reg_id2var_id: &impl Fn(RegisterId) -> VariableId,
    libs: &mut Libs,
    target_path: Option<PathBuf>,
) -> GeneratedFunctions {
    let inst_idx2fused_meta = gen_fused_kernels(program, reg_id2var_id, target_path.clone());
    let mut uf_table: Heap<UserFunctionId, _> = user_ftable.map(&mut (|_, f| Some(f)));

    let mut inst2func = BTreeMap::new();
    let mut kernel2func: BTreeMap<KernelType, FunctionId> = BTreeMap::new();
    for (id, instruct) in program.iter_instructions() {
        let fuse_meta = inst_idx2fused_meta.get(&id);
        let kernel_type = kernel_type_from_inst(instruct, fuse_meta);
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
        inst_idx2fused_meta,
    }
}

pub fn load_function<Rt: RuntimeType>(
    kernel_type: &KernelType,
    libs: &mut Libs,
    uf_table: &mut Heap<UserFunctionId, Option<type2::user_function::Function<Rt>>>,
) -> zkpoly_runtime::functions::Function<Rt> {
    match &kernel_type {
        KernelType::PolyPermute => {
            let poly_permute = PolyPermute::new(libs);
            poly_permute.get_fn()
        }
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
            if meta.pipelined_meta.is_some() {
                PipelinedFusedKernel::new(libs, meta.clone()).get_fn()
            } else {
                FusedKernel::new(libs, meta.clone()).get_fn()
            }
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
