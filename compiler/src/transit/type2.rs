//! Data structures for Stage 2 Tree Transit IR.
//! Passes:
//! AST >
//! - INTT Mending
//! - Precompute NTT and MSM constants
//! - Manage Inversions
//! - Arithmetic Kernel Fusion
//! - Graph Scheduling
//! - Memory Planning
//! > Type3

use crate::ast;
use crate::transit::{self, PolyInit, SourceInfo};
pub use ast::lowering::{Constant, ConstantId, ConstantTable};
pub use typ::Typ;
use zkpoly_common::arith;
use zkpoly_common::digraph;
use zkpoly_common::digraph::internal::SubDigraph;
use zkpoly_common::heap::UsizeId;
use zkpoly_common::load_dynamic::Libs;
pub use zkpoly_common::typ::PolyType;
pub use zkpoly_runtime::args::{RuntimeType, Variable};
pub use zkpoly_runtime::error::RuntimeError;

zkpoly_common::define_usize_id!(VertexId);

impl std::fmt::Display for VertexId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", usize::from(*self))
    }
}

pub type Arith<I> = arith::ArithGraph<I, arith::ExprId>;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum NttAlgorithm<I> {
    Precomputed(I),
    Standard { pq: I, omega: I },
    Undecieded,
}

impl<I> Default for NttAlgorithm<I> {
    fn default() -> Self {
        NttAlgorithm::Undecieded
    }
}

impl<I> NttAlgorithm<I>
where
    I: Clone,
{
    pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use NttAlgorithm::*;
        match self {
            Precomputed(x) => Box::new([x].into_iter()),
            Standard { pq, omega: base } => Box::new([pq, base].into_iter()),
            Undecieded => Box::new(std::iter::empty()),
        }
    }

    pub fn uses<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.uses_ref().cloned()
    }

    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use NttAlgorithm::*;
        match self {
            Precomputed(x) => Box::new([x].into_iter()),
            Standard { pq, omega: base } => Box::new([pq, base].into_iter()),
            Undecieded => Box::new(std::iter::empty()),
        }
    }

    pub fn try_relabeled<I2, Err>(
        &self,
        mapping: &mut impl FnMut(I) -> Result<I2, Err>,
    ) -> Result<NttAlgorithm<I2>, Err> {
        use NttAlgorithm::*;
        Ok(match self {
            Precomputed(x) => Precomputed(mapping(x.clone())?),
            Standard { pq, omega: base } => Standard {
                pq: mapping(pq.clone())?,
                omega: mapping(base.clone())?,
            },
            Undecieded => Undecieded,
        })
    }

    pub fn relabeled<I2>(&self, mapping: &mut impl FnMut(I) -> I2) -> NttAlgorithm<I2> {
        self.try_relabeled::<_, ()>(&mut |i| Ok(mapping(i)))
            .unwrap()
    }
}

impl<I> NttAlgorithm<I>
where
    I: Copy + Default,
{
    pub fn decide_alg<Rt: RuntimeType>(
        len: usize,
        memory_limit: usize,
        recompute_len: usize,
    ) -> Self {
        let field_sz = size_of::<Rt::Field>();
        let total_sz = len * field_sz;
        let recompute_sz = recompute_len * field_sz;
        let temp_id = I::default(); // 0 for later allocate
        if total_sz * 3 < memory_limit {
            Self::Precomputed(temp_id)
        } else if total_sz * 2 + recompute_sz < memory_limit {
            Self::Standard {
                pq: temp_id,
                omega: temp_id,
            }
        } else if total_sz * 3 < memory_limit * 2 {
            // 1.5
            Self::Precomputed(temp_id)
        } else if total_sz + recompute_sz < memory_limit {
            Self::Standard {
                pq: temp_id,
                omega: temp_id,
            }
        } else {
            unimplemented!("out of core ntt is unimplemented")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu(usize),
    Cpu,
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        match self {
            Device::Gpu(_) => true,
            Device::Cpu => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    PreferGpu,
    Gpu,
    Cpu,
}

pub mod template;

pub mod sliceable_subgraph;

/// The computation graph with full type information and sliceable subgraphs,
/// but uses alternative labels.
pub mod alt_label {
    use super::*;
    pub type VertexNode<'s, I, Rt> = template::VertexNode<
        I,
        Arith<I>,
        ConstantId,
        user_function::Id,
        sliceable_subgraph::alt_label::Cg<'s, sliceable_subgraph::VertexId, I, Rt>,
    >;
    pub type Vertex<'s, I, Rt> = transit::Vertex<VertexNode<'s, I, Rt>, Typ<Rt>, SourceInfo<'s>>;
}

/// The standard computation graph with full type information and sliceable subgraphs,
/// after subgraph slicing.
pub type VertexNode<'s, Rt> = alt_label::VertexNode<'s, VertexId, Rt>;
pub type Vertex<'s, Rt> = alt_label::Vertex<'s, VertexId, Rt>;
pub type Cg<'s, Rt> = transit::Cg<VertexId, Vertex<'s, Rt>>;
pub type CgSubgraph<'g, 's, Rt> = SubDigraph<'g, VertexId, Vertex<'s, Rt>>;

/// The computation graph generated from AST.
/// It has only partial type information, and it does not contain any sliceable subgraph.
pub mod partial_typed {
    use super::*;
    pub type VertexNode =
        template::VertexNode<VertexId, Arith<VertexId>, ConstantId, user_function::Id, ()>;
    pub type Vertex<'s, T> = transit::Vertex<VertexNode, T, SourceInfo<'s>>;
}

/// The computation graph after type inference.
/// It has full tyep information, but still does not contain any sliceable subgraph.
pub mod no_subgraph {
    pub mod alt_label {
        use super::super::*;
        pub type VertexNode<I> =
            template::VertexNode<I, Arith<I>, ConstantId, user_function::Id, ()>;
        pub type Vertex<'s, I, T> = transit::Vertex<VertexNode<I>, T, SourceInfo<'s>>;
        pub type Cg<'s, I, T> = transit::Cg<I, Vertex<'s, I, T>>;
    }

    pub type VertexNode = alt_label::VertexNode<super::VertexId>;
    pub type Vertex<'s, Rt> = alt_label::Vertex<'s, super::VertexId, super::Typ<Rt>>;
    pub type Cg<'s, Rt> = alt_label::Cg<'s, super::VertexId, super::Typ<Rt>>;
}

impl<I: UsizeId, C, E, T, S, Sg> digraph::internal::Predecessors<I>
    for transit::Vertex<
        template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, E, Sg>,
        T,
        S,
    >
where
    Sg: template::SubgraphNode<I>,
    I: 'static,
{
    #[allow(refining_impl_trait)]
    fn predecessors<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.node().uses()
    }
}

impl<'s, I, C, Sg, T, S>
    transit::Vertex<
        template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, user_function::Id, Sg>,
        T,
        S,
    >
where
    I: Copy + 'static,
    Sg: template::SubgraphNode<I>,
{
    pub fn uses<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.node().uses()
    }
    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        self.node_mut().uses_mut()
    }
    pub fn device(&self) -> DevicePreference {
        self.node().device()
    }
    pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        use template::VertexNode::*;
        match self.node() {
            Sliceable(sn) => sn.mutable_uses(),
            LastSliceable(lsn) => lsn.mutable_uses(),
            Ntt { s, .. } => Box::new([*s].into_iter()),
            HashTranscript { transcript, .. } => Box::new([*transcript].into_iter()),
            SqueezeScalar(transcript) => Box::new([*transcript].into_iter()),
            _ => Box::new([].into_iter()),
        }
    }
    pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use template::VertexNode::*;
        match self.node_mut() {
            Sliceable(sn) => sn.mutable_uses_mut(),
            LastSliceable(lsn) => lsn.mutable_uses_mut(),
            Ntt { s, .. } => Box::new([s].into_iter()),
            HashTranscript { transcript, .. } => Box::new([transcript].into_iter()),
            SqueezeScalar(transcript) => Box::new([transcript].into_iter()),
            _ => Box::new([].into_iter()),
        }
    }

    /// For each output, if it took the space of some input `i`, returns `Some(i)`, otherwise `None`.
    /// If no output is inplace, returns an infinite iterator of `None`.
    pub fn outputs_inplace<'a, 'b, Rt: RuntimeType>(
        &'b self,
        uf_table: &'a user_function::Table<Rt>,
    ) -> Box<dyn Iterator<Item = Option<I>> + 'b> {
        use template::VertexNode::*;
        match self.node() {
            Sliceable(sn) => sn.outputs_inplace(),
            LastSliceable(lsn) => lsn.outputs_inplace(),
            Ntt { s, .. } => Box::new([Some(*s)].into_iter()),
            HashTranscript { transcript, .. } => Box::new([Some(*transcript)].into_iter()),
            SqueezeScalar(transcript) => Box::new([Some(*transcript), None].into_iter()),
            UserFunction(fid, args) => {
                let f_typ = &uf_table[*fid].typ;
                let r = f_typ
                    .ret_inplace
                    .iter()
                    .map(|&i| Some(args[i?]))
                    .collect::<Vec<_>>();
                Box::new(r.into_iter())
            }
            _ => Box::new(std::iter::repeat(None)),
        }
    }

    pub fn is_virtual(&self) -> bool {
        self.node().is_virtual()
    }
}

impl<I, C, S>
    template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, user_function::Id, S>
{
    pub fn deterministic<Rt: RuntimeType>(&self, uf_table: &user_function::Table<Rt>) -> bool {
        match self {
            template::VertexNode::Sliceable(template::SliceableNode::Blind(..)) => false,
            template::VertexNode::UserFunction(f, _) => uf_table[*f].f.deterministic,
            _ => true,
        }
    }
}

impl<'s, Rt: RuntimeType> Cg<'s, Rt> {
    pub fn temporary_space_needed(
        &self,
        vid: VertexId,
        device: Device,
        libs: &mut Libs,
    ) -> Option<(Vec<u64>, super::type3::Device)> {
        use template::LastSliceableNode::*;
        use template::SliceableNode::*;
        use template::VertexNode::*;

        let on_device = super::type3::Device::for_execution_on;
        let on_gpu = |dev: Device| {
            if !dev.is_gpu() {
                panic!(
                    "this vertex {:?} needs to be executed on some GPU, got {:?}",
                    vid, device
                );
            }
            on_device(dev)
        };

        match self.g.vertex(vid).node() {
            Extend(..) => None,
            Sliceable(ScalarInvert { .. }) => None,
            Sliceable(SingleArith(..)) => None,
            Sliceable(Arith { arith, chunking }) => Some((
                temporary_space::arith::<Rt>(arith, chunking.clone()),
                // We are using only one gpu for now
                on_gpu(Device::Gpu(0)),
            )),
            Subgraph(..) => None,
            Sliceable(NewPoly(..)) => None,
            Sliceable(Constant(..)) => None,
            UnsliceableConstant(..) => None,
            Return(..) => None,
            Entry(..) => None,
            Ntt { .. } => None,
            Sliceable(RotateIdx(..)) => None,
            Slice(..) => None,
            Interpolate { .. } => None,
            Sliceable(Blind(..)) => None,
            Array(..) => None,
            AssmblePoly(..) => None,
            LastSliceable(Msm { polys, alg, .. }) => {
                let (_, len) = self.g.vertex(polys[0]).typ().unwrap_poly();
                Some((
                    temporary_space::msm::<Rt>(alg, *len as usize, libs),
                    // We are using only one gpu for now
                    on_device(Device::Gpu(0)),
                ))
            }
            HashTranscript { .. } => None,
            SqueezeScalar(..) => None,
            TupleGet(..) => None,
            ArrayGet(..) => None,
            UserFunction(..) => None,
            KateDivision(lhs, _) => {
                let (_, len) = self.g.vertex(*lhs).typ().unwrap_poly();
                Some((
                    temporary_space::kate_division::<Rt>(*len as usize, libs),
                    on_gpu(device),
                ))
            }
            LastSliceable(EvaluatePoly { poly, .. }) => {
                let (_, len) = self.g.vertex(*poly).typ().unwrap_poly();
                Some((
                    temporary_space::poly_eval::<Rt>(*len as usize, libs),
                    on_device(device),
                ))
            }
            PolyPermute(_, _, usable) => Some((
                temporary_space::poly_permute::<Rt>(*usable as usize, libs),
                on_gpu(device),
            )),
            Sliceable(BatchedInvert(poly)) => {
                let (_, len) = self.g.vertex(*poly).typ().unwrap_poly();
                Some((
                    temporary_space::poly_invert::<Rt>(*len as usize, libs),
                    on_gpu(device),
                ))
            }
            Sliceable(ScanMul { poly, .. }) => {
                let (_, len) = self.g.vertex(*poly).typ().unwrap_poly();
                Some((
                    temporary_space::poly_scan::<Rt>(*len as usize, libs),
                    on_gpu(device),
                ))
            }
            Sliceable(DistributePowers { .. }) => None,
            IndexPoly(..) => None,
            AssertEq(..) => None,
            Print(..) => None,
        }
    }
}

pub struct Program<'s, Rt: RuntimeType> {
    pub(crate) cg: no_subgraph::Cg<'s, Rt>,
    pub(crate) user_function_table: user_function::Table<Rt>,
    pub(crate) consant_table: ConstantTable<Rt>,
}

pub mod arith_decide_mutable;
pub mod common_subexpression_elimination;
pub mod decide_device;
pub mod graph_scheduling;
pub mod intt_mending;
pub mod kernel_fusion;
pub mod manage_inverse;
pub mod memory_planning;
pub mod object_analysis;
pub mod precompute;
pub mod pretty;
pub mod pretty_print;
pub mod subgraph_slicing;
pub mod temporary_space;
pub mod typ;
pub mod user_function;
pub mod visualize;
