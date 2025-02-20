use crate::transit::type2::NttAlgorithm;
use std::collections::BTreeMap;
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
    KateDivision, PolyEval, PolyInvert, PolyOneCoef, PolyOneLagrange, PolyScan, PolyZero,
};
use zkpoly_runtime::args::RuntimeType;
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
    UserFunction,
    DistributePowers,
    NewOneLagrange,
    NewOneCoef,
    NewZero,
}

impl KernelType {
    pub fn from_vertex(vertex: &VertexNode) -> Option<Self> {
        assert!(!vertex.unexpcted_during_lowering());
        match vertex {
            VertexNode::Arith(..) => Some(Self::FusedArith),
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
            VertexNode::UserFunction(..) => Some(Self::UserFunction),
            VertexNode::DistributePowers { .. } => Some(Self::DistributePowers),
            VertexNode::NewPoly(_, init_num, typ) => {
                let typ = typ.clone();
                match init_num {
                    crate::ast::PolyInit::Zeros => Some(Self::NewZero),
                    crate::ast::PolyInit::Ones => match typ {
                        PolyType::Lagrange => Some(Self::NewOneLagrange),
                        PolyType::Coef => Some(Self::NewOneCoef),
                        _ => unreachable!(),
                    },
                }
            }
            _ => None,
        }
    }
}

const FUSED_PERFIX: &str = "fused_arith_";

pub fn gen_fused_kernels<'s, Rt: RuntimeType>(program: &Chunk<'s, Rt>) {
    // first pass to generate fused arith kernels
    for (id, instruct) in program.iter_instructions() {
        if let InstructionNode::Type2 { vertex, .. } = &instruct.node {
            if let VertexNode::Arith(graph, _) = vertex {
                let id: usize = id.into();
                let name = format!("{FUSED_PERFIX}{id}");
                FusedOp::new(graph.clone(), name).gen(); // generate fused kernel
            }
        }
    }
}

pub struct GeneratedFunctions {
    inst2fid: BTreeMap<InstructionIndex, FunctionId>,
}

impl GeneratedFunctions {
    pub fn at(&self, idx: InstructionIndex) -> FunctionId {
        self.inst2fid.get(&idx).unwrap().clone()
    }
}

pub fn get_function_id<'s, Rt: RuntimeType>(
    f_table: &mut FunctionTable<Rt>,
    program: &Chunk<'s, Rt>,
    libs: &mut Libs,
) -> GeneratedFunctions {
    gen_fused_kernels(program);
    let mut inst2func = BTreeMap::new();
    let mut kernel2func: BTreeMap<KernelType, FunctionId> = BTreeMap::new();
    for (id, instruct) in program.iter_instructions() {
        if let InstructionNode::Type2 { vertex, .. } = &instruct.node {
            let kernel_type = KernelType::from_vertex(vertex);
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
                KernelType::UserFunction => todo!(),
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
            }
        }
    }

    GeneratedFunctions {
        inst2fid: inst2func,
    }
}
