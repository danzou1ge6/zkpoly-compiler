//! Data structures for Stage 2 Tree Transit IR.
//! This type of IR has NTT automatically inserted,
//! therefore polynomial operations are transformed into vector expressions.
//! Also, vertices of the computation graph still contain expression trees.

use crate::transit::{self, BinOp, SourceInfo, UnrOp, PolyInit};
use std::any;
use zkpoly_common::{digraph, heap};
use zkpoly_runtime::args::RuntimeType;
pub use zkpoly_runtime::args::{Constant, ConstantId};
pub use zkpoly_runtime::functions::{Function, FunctionId as UFunctionId};
pub use zkpoly_runtime::poly::PolyType;
pub use typ::Typ;

zkpoly_common::define_usize_id!(ExprId);

impl ExprId {
    pub(super) fn from_type1(x: transit::type1::ExprId) -> Self {
        Self(x.into())
    }
}

pub type Arith = transit::Arith<ExprId>;

#[derive(Debug, Clone)]
pub enum NttAlgorithm {
    Precomputed,
    Standard
}

#[derive(Debug, Clone)]
pub enum MsmAlgorithm {
    Batched,
    Sequential
}

impl NttAlgorithm {
    pub fn decide(deg: u64) -> Self {
        if deg <= 2u64.pow(20) {
            NttAlgorithm::Precomputed
        } else {
            NttAlgorithm::Standard
        }
    }
    pub fn temporary_space_needed<F>(&self, deg: u64) -> u64 {
        match self {
            NttAlgorithm::Precomputed => deg * size_of::<F>() as u64,
            NttAlgorithm::Standard => 0,
        }
    }
}

impl Default for NttAlgorithm {
    fn default() -> Self {
        NttAlgorithm::Precomputed
    }
}

impl MsmAlgorithm {
    pub fn decide(deg: u64) -> Self {
        if deg <= 2u64.pow(20) {
            MsmAlgorithm::Batched
        } else {
            MsmAlgorithm::Sequential
        }
    }
    pub fn temporary_space_needed<F>(&self, deg: u64) -> u64 {
        // WARNING placeholder here, need complete parameters to decide the temporary space needed
        0
    }
}

impl Default for MsmAlgorithm {
    fn default() -> Self {
        MsmAlgorithm::Batched
    }
}


pub mod template {
    use super::*;

    #[derive(Debug, Clone)]
    pub enum VertexNode<I, A, C, E> {
        NewPoly(u64, PolyInit),
        Constant(C),
        Arith(A),
        Entry,
        Return,
        /// A small scalar. To accomodate big scalars, use global constants.
        LiteralScalar(usize),
        /// Convert a local from one representation to another
        Ntt {
            s: I,
            to: PolyType,
            from: PolyType,
            alg: NttAlgorithm
        },
        RotateIdx(I, i32),
        Interplote {
            xs: Vec<I>,
            ys: Vec<I>,
        },
        Blind(I, usize),
        Array(Vec<I>),
        AssmblePoly(u64, I),
        Msm {
            scalars: I,
            points: I,
            alg: MsmAlgorithm
        },
        HashTranscript {
            transcript: I,
            value: I,
            typ: transit::HashTyp,
        },
        /// Returns (transcript, scalar)
        SqueezeScalar(I),
        TupleGet(I, usize),
        ArrayGet(I, usize),
        UserFunction(E, Vec<I>),
    }
}

pub type VertexNode = template::VertexNode<ExprId, Arith, ConstantId, UFunctionId>;

pub type Vertex<'s, Rt: RuntimeType> = transit::Vertex<VertexNode, Typ<Rt>, SourceInfo<'s>>;

impl<'s, Rt: RuntimeType> digraph::internal::Predecessors<ExprId> for Vertex<'s, Rt> {
    #[allow(refining_impl_trait)]
    fn predecessors<'a>(&'a self) -> Box<dyn Iterator<Item = ExprId> + 'a> {
        self.uses()
    }
}

impl<'s, Rt: RuntimeType> Vertex<'s, Rt> {
    pub fn uses<'a>(&'a self) -> Box<dyn Iterator<Item = ExprId> + 'a> {
        use template::VertexNode::*;
        match self.node() {
            Arith(transit::Arith::Bin(_, lhs, rhs)) => Box::new([*lhs, *rhs].into_iter()),
            Arith(transit::Arith::Unr(_, x)) => Box::new([*x].into_iter()),
            Ntt { s, .. } => Box::new([*s].into_iter()),
            RotateIdx(x, _) => Box::new([*x].into_iter()),
            Interplote { xs, ys } => Box::new(xs.iter().copied().chain(ys.iter().copied())),
            Array(es) => Box::new(es.iter().copied()),
            AssmblePoly(_, es) => Box::new([*es].into_iter()),
            Msm { scalars, points , ..} => Box::new([*scalars, *points].into_iter()),
            HashTranscript {
                transcript, value, ..
            } => Box::new([*transcript, *value].into_iter()),
            SqueezeScalar(x) => Box::new([*x].into_iter()),
            TupleGet(x, _) => Box::new([*x].into_iter()),
            Blind(x, _) => Box::new([*x].into_iter()),
            ArrayGet(x, _) => Box::new([*x].into_iter()),
            UserFunction(_, es) => Box::new(es.iter().copied()),
            _ => Box::new([].into_iter()),
        }
    }
    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut ExprId> + 'a> {
        use template::VertexNode::*;
        match self.node_mut() {
            Arith(transit::Arith::Bin(_, lhs, rhs)) => Box::new([lhs, rhs].into_iter()),
            Arith(transit::Arith::Unr(_, x)) => Box::new([x].into_iter()),
            Ntt { s, .. } => Box::new([s].into_iter()),
            RotateIdx(x, _) => Box::new([x].into_iter()),
            Interplote { xs, ys } => Box::new(xs.iter_mut().chain(ys.iter_mut())),
            Array(es) => Box::new(es.iter_mut()),
            AssmblePoly(_, es) => Box::new([es].into_iter()),
            Msm { scalars, points, .. } => Box::new([scalars, points].into_iter()),
            HashTranscript {
                transcript, value, ..
            } => Box::new([transcript, value].into_iter()),
            SqueezeScalar(x) => Box::new([x].into_iter()),
            TupleGet(x, _) => Box::new([x].into_iter()),
            Blind(x, _) => Box::new([x].into_iter()),
            ArrayGet(x, _) => Box::new([x].into_iter()),
            UserFunction(_, es) => Box::new(es.iter_mut()),
            _ => Box::new([].into_iter()),
        }
    }
    pub fn modify_locals(&mut self, f: &mut impl FnMut(ExprId) -> ExprId) {
        self.uses_mut().for_each(|i| *i = f(*i));
    }
    pub fn scalar_size() -> usize {
        unimplemented!()
    }
    pub fn temporary_space_needed(&self) -> u64 {
        use template::VertexNode::*;
        match self.node() {
            Arith(..) => 0,
            NewPoly(..) => 0,
            Constant(..) => 0,
            Entry => 0,
            Return => 0,
            LiteralScalar(..) => 0,
            Ntt {alg, .. } => alg.temporary_space_needed::<Rt::Field>(self.typ().unwrap_poly().1),
            RotateIdx(..) => 0,
            Interplote {.. } => 0,
            Blind(..) => 0,
            Array(..) => 0,
            AssmblePoly(..) => 0,
            Msm {alg,.. } => alg.temporary_space_needed::<Rt::Field>(self.typ().unwrap_poly().1),
            HashTranscript {.. } => 0,
            SqueezeScalar(..) => 0,
            TupleGet(..) => 0,
            ArrayGet(..) => 0,
            UserFunction(..) => 0,
        }
    }
}

/// Invariants are same as those of [`tree::tree1::Cg`]
pub type Cg<'s, Rt: RuntimeType> = transit::Cg<ExprId, Vertex<'s, Rt>>;

pub mod typ;
pub mod graph_scheduling;
