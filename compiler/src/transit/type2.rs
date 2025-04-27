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
use zkpoly_common::arith::{self, ArithUnrOp, UnrOp};
use zkpoly_common::digraph;
use zkpoly_common::heap::UsizeId;
use zkpoly_common::load_dynamic::Libs;
pub use zkpoly_common::typ::PolyType;
use zkpoly_memory_pool::PinnedMemoryPool;
pub use zkpoly_runtime::args::{RuntimeType, Variable};
pub use zkpoly_runtime::error::RuntimeError;

zkpoly_common::define_usize_id!(VertexId);

pub type Arith = arith::ArithGraph<VertexId, arith::ExprId>;

#[derive(Debug, Clone)]
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
    I: Copy,
{
    pub fn uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        use NttAlgorithm::*;
        match self {
            Precomputed(x) => Box::new([*x].into_iter()),
            Standard { pq, omega: base } => Box::new([*pq, *base].into_iter()),
            Undecieded => Box::new(std::iter::empty()),
        }
    }

    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use NttAlgorithm::*;
        match self {
            Precomputed(x) => Box::new([x].into_iter()),
            Standard { pq, omega: base } => Box::new([pq, base].into_iter()),
            Undecieded => Box::new(std::iter::empty()),
        }
    }

    pub fn relabeled<I2>(&self, mapping: &mut impl FnMut(I) -> I2) -> NttAlgorithm<I2> {
        use NttAlgorithm::*;
        match self {
            Precomputed(x) => Precomputed(mapping(*x)),
            Standard { pq, omega: base } => Standard {
                pq: mapping(*pq),
                omega: mapping(*base),
            },
            Undecieded => Undecieded,
        }
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
    Gpu,
    Cpu,
    /// If it has any successor or predecesor on GPU, then it should be on GPU
    PreferGpu,
}

pub mod template {
    use zkpoly_common::msm_config::MsmConfig;
    use zkpoly_runtime::args::EntryId;

    use super::{arith, transit, Device, NttAlgorithm, PolyInit, PolyType};

    #[derive(Debug, Clone)]
    pub enum VertexNode<I, A, C, E> {
        NewPoly(u64, PolyInit, PolyType),
        Constant(C),
        Extend(I, u64),
        /// A single binary or unary arith expression, to be fused to an arith graph
        SingleArith(arith::Arith<I>),
        ScalarInvert {
            val: I,
        },
        /// .1 is size of each chunk, if chunking is enabled
        Arith {
            arith: A,
            chunking: Option<u64>,
        },
        Entry(EntryId),
        Return(I),
        /// Convert a local from one representation to another
        Ntt {
            s: I,
            to: PolyType,
            from: PolyType,
            alg: NttAlgorithm<I>,
        },
        RotateIdx(I, i32),
        Slice(I, u64, u64),
        Interpolate {
            xs: Vec<I>,
            ys: Vec<I>,
        },
        Blind(I, u64, u64),
        Array(Vec<I>),
        AssmblePoly(u64, Vec<I>),
        Msm {
            polys: Vec<I>,
            points: Vec<I>,
            alg: MsmConfig,
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
        KateDivision(I, I),
        EvaluatePoly {
            poly: I,
            at: I,
        },
        BatchedInvert(I),
        ScanMul {
            x0: I,
            poly: I,
        },
        DistributePowers {
            poly: I,
            powers: I,
        },
        IndexPoly(I, u64),
        AssertEq(I, I, Option<String>),
        Print(I, String),
    }

    impl<I, C, E> VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, E>
    where
        I: Copy,
    {
        pub fn unwrap_constant(&self) -> &C {
            match self {
                Self::Constant(constant) => constant,
                _ => panic!("this is not constant"),
            }
        }
        pub fn unwrap_ntt_alg_mut(&mut self) -> &mut NttAlgorithm<I> {
            match self {
                Self::Ntt { alg, .. } => alg,
                _ => panic!("this is not ntt"),
            }
        }

        pub fn unwrap_msm(&mut self) -> (&mut Vec<I>, &mut Vec<I>, &mut MsmConfig) {
            match self {
                Self::Msm { polys, points, alg } => (polys, points, alg),
                _ => panic!("this is not msm"),
            }
        }
        pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
            use VertexNode::*;
            match self {
                Extend(x, _) => Box::new([x].into_iter()),
                SingleArith(expr) => expr.uses_mut(),
                ScalarInvert { val } => Box::new([val].into_iter()),
                Arith { arith, .. } => arith.uses_mut(),
                Ntt { s, alg, .. } => Box::new([s].into_iter().chain(alg.uses_mut())),
                RotateIdx(x, _) => Box::new([x].into_iter()),
                Slice(x, ..) => Box::new([x].into_iter()),
                Interpolate { xs, ys } => Box::new(xs.iter_mut().chain(ys.iter_mut())),
                Array(es) => Box::new(es.iter_mut()),
                AssmblePoly(_, es) => Box::new(es.iter_mut()),
                Msm {
                    polys: scalars,
                    points,
                    ..
                } => Box::new(scalars.iter_mut().chain(points.iter_mut())),
                HashTranscript {
                    transcript, value, ..
                } => Box::new([transcript, value].into_iter()),
                SqueezeScalar(x) => Box::new([x].into_iter()),
                TupleGet(x, _) => Box::new([x].into_iter()),
                Blind(x, ..) => Box::new([x].into_iter()),
                ArrayGet(x, _) => Box::new([x].into_iter()),
                UserFunction(_, es) => Box::new(es.iter_mut()),
                KateDivision(lhs, rhs) => Box::new([lhs, rhs].into_iter()),
                EvaluatePoly { poly, at } => Box::new([poly, at].into_iter()),
                BatchedInvert(x) => Box::new([x].into_iter()),
                ScanMul { x0, poly } => Box::new([x0, poly].into_iter()),
                DistributePowers { powers, poly } => Box::new([poly, powers].into_iter()),
                Return(x) => Box::new([x].into_iter()),
                IndexPoly(x, _) => Box::new([x].into_iter()),
                AssertEq(x, y, _msg) => Box::new([x, y].into_iter()),
                Print(x, _) => Box::new([x].into_iter()),
                _ => Box::new(std::iter::empty()),
            }
        }

        pub fn uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
            use VertexNode::*;
            match self {
                Extend(x, _) => Box::new([*x].into_iter()),
                SingleArith(expr) => expr.uses(),
                ScalarInvert { val } => Box::new([*val].into_iter()),
                Arith { arith, .. } => Box::new(arith.uses()),
                Ntt { s, alg, .. } => Box::new([*s].into_iter().chain(alg.uses())),
                RotateIdx(x, _) => Box::new([*x].into_iter()),
                Slice(x, ..) => Box::new([*x].into_iter()),
                Interpolate { xs, ys } => Box::new(xs.iter().copied().chain(ys.iter().copied())),
                Array(es) => Box::new(es.iter().copied()),
                AssmblePoly(_, es) => Box::new(es.iter().copied()),
                Msm {
                    polys: scalars,
                    points,
                    ..
                } => Box::new(scalars.iter().copied().chain(points.iter().copied())),
                HashTranscript {
                    transcript, value, ..
                } => Box::new([*transcript, *value].into_iter()),
                SqueezeScalar(x) => Box::new([*x].into_iter()),
                TupleGet(x, _) => Box::new([*x].into_iter()),
                Blind(x, ..) => Box::new([*x].into_iter()),
                ArrayGet(x, _) => Box::new([*x].into_iter()),
                UserFunction(_, es) => Box::new(es.iter().copied()),
                KateDivision(lhs, rhs) => Box::new([*lhs, *rhs].into_iter()),
                EvaluatePoly { poly, at } => Box::new([*poly, *at].into_iter()),
                BatchedInvert(x) => Box::new([*x].into_iter()),
                ScanMul { x0, poly } => Box::new([*x0, *poly].into_iter()),
                DistributePowers { powers, poly } => Box::new([*poly, *powers].into_iter()),
                Return(x) => Box::new([*x].into_iter()),
                IndexPoly(x, _) => Box::new([*x].into_iter()),
                AssertEq(x, y, _msg) => Box::new([*x, *y].into_iter()),
                Print(x, _) => Box::new([*x].into_iter()),
                _ => Box::new(std::iter::empty()),
            }
        }

        pub fn device(&self) -> Device {
            use VertexNode::*;
            match self {
                NewPoly(..) | Extend(..) | ScalarInvert { .. } | IndexPoly { .. } => {
                    Device::PreferGpu
                }
                Arith { chunking, .. } => {
                    if chunking.is_some() {
                        Device::Cpu
                    } else {
                        Device::Gpu
                    }
                }
                Ntt { .. } => Device::Gpu,
                KateDivision(_, _) => Device::Gpu,
                EvaluatePoly { .. } => Device::Gpu,
                BatchedInvert(_) => Device::Gpu,
                ScanMul { .. } => Device::Gpu,
                DistributePowers { .. } => Device::Gpu,
                SingleArith(arith) => {
                    match arith {
                        zkpoly_common::arith::Arith::Bin(..) => Device::Gpu, // for add/sub with different len
                        zkpoly_common::arith::Arith::Unr(..) => Device::PreferGpu, // for pow
                    }
                }
                _ => Device::Cpu,
            }
        }

        pub fn is_virtual(&self) -> bool {
            use VertexNode::*;
            match self {
                Array(..) | ArrayGet(..) | TupleGet(..) | RotateIdx(..) | Slice(..) => true,
                _ => false,
            }
        }

        pub fn unexpcted_during_kernel_gen(&self) -> bool {
            use VertexNode::*;
            match self {
                Extend(..) | NewPoly(..) => true,
                _ => self.is_virtual(),
            }
        }

        pub fn is_return(&self) -> bool {
            use VertexNode::*;
            match self {
                Return(..) => true,
                _ => false,
            }
        }

        pub fn no_allocate_output(&self) -> bool {
            use VertexNode::*;
            match self {
                Entry(..) | Constant(..) | AssertEq(..) | Print(..) => true,
                _ => false,
            }
        }

        pub fn immortal_on_cpu(&self) -> bool {
            use VertexNode::*;
            match self {
                Constant(..) => true,
                _ => false,
            }
        }
    }
}

pub mod partial_typed {
    use super::*;
    pub type Vertex<'s, T> = transit::Vertex<VertexNode, T, SourceInfo<'s>>;
}

pub type VertexNode = template::VertexNode<VertexId, Arith, ConstantId, user_function::Id>;

pub type Vertex<'s, Rt> = transit::Vertex<VertexNode, Typ<Rt>, SourceInfo<'s>>;

impl<'s, I: UsizeId, C, E, T, S> digraph::internal::Predecessors<I>
    for transit::Vertex<template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, E>, T, S>
{
    #[allow(refining_impl_trait)]
    fn predecessors<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        self.node().uses()
    }
}

impl<'s, Rt: RuntimeType, I, C>
    transit::Vertex<
        template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, user_function::Id>,
        Typ<Rt>,
        SourceInfo<'s>,
    >
where
    I: Copy,
{
    pub fn uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        self.node().uses()
    }
    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        self.node_mut().uses_mut()
    }
    pub fn space(&self) -> u64 {
        self.typ().size().total()
    }
    pub fn device(&self) -> Device {
        self.node().device()
    }
    pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        use template::VertexNode::*;
        match self.node() {
            ScalarInvert { val } => Box::new([*val].into_iter()),
            Ntt { s, .. } => Box::new([*s].into_iter()),
            RotateIdx(s, ..) => Box::new([*s].into_iter()),
            HashTranscript { transcript, .. } => Box::new([*transcript].into_iter()),
            SqueezeScalar(transcript) => Box::new([*transcript].into_iter()),
            Arith { arith, .. } => arith.mutable_uses(),
            Blind(poly, ..) => Box::new([*poly].into_iter()),
            BatchedInvert(poly) => Box::new([*poly].into_iter()),
            DistributePowers { poly, .. } => Box::new([*poly].into_iter()),
            SingleArith(arith::Arith::Unr(UnrOp::S(ArithUnrOp::Pow(_)), scalar)) => {
                Box::new([*scalar].into_iter())
            }
            _ => Box::new([].into_iter()),
        }
    }
    pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use template::VertexNode::*;
        match self.node_mut() {
            ScalarInvert { val } => Box::new([val].into_iter()),
            Ntt { s, .. } => Box::new([s].into_iter()),
            RotateIdx(s, ..) => Box::new([s].into_iter()),
            HashTranscript { transcript, .. } => Box::new([transcript].into_iter()),
            SqueezeScalar(transcript) => Box::new([transcript].into_iter()),
            Arith { arith, .. } => arith.mutable_uses_mut(),
            Blind(poly, ..) => Box::new([poly].into_iter()),
            BatchedInvert(poly) => Box::new([poly].into_iter()),
            DistributePowers { poly, .. } => Box::new([poly].into_iter()),
            SingleArith(arith::Arith::Unr(UnrOp::S(ArithUnrOp::Pow(_)), scalar)) => {
                Box::new([scalar].into_iter())
            }
            _ => Box::new([].into_iter()),
        }
    }
    pub fn outputs_inplace<'a, 'b>(
        &'b self,
        uf_table: &'a user_function::Table<Rt>,
        device: Device,
    ) -> Box<dyn Iterator<Item = Option<I>> + 'b> {
        use template::VertexNode::*;
        match self.node() {
            ScalarInvert { val } => Box::new([Some(*val)].into_iter()),
            Ntt { s, .. } => Box::new([Some(*s)].into_iter()),
            RotateIdx(s, ..) => {
                use Device::*;
                match device {
                    Cpu => Box::new([Some(*s)].into_iter()),
                    Gpu => Box::new([None].into_iter()),
                    PreferGpu => panic!("device must be completely decided here"),
                }
            }
            Arith { arith, .. } => arith.outputs_inplace(),
            Blind(poly, ..) => Box::new([Some(*poly)].into_iter()),
            BatchedInvert(poly) => Box::new([Some(*poly)].into_iter()),
            DistributePowers { poly, .. } => Box::new([Some(*poly)].into_iter()),
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
            SingleArith(arith::Arith::Unr(UnrOp::S(ArithUnrOp::Pow(_)), scalar)) => {
                Box::new([Some(*scalar)].into_iter())
            }
            _ => {
                let len = self.typ().size().len();
                Box::new(std::iter::repeat(None).take(len))
            }
        }
    }

    pub fn is_virtual(&self) -> bool {
        self.node().is_virtual()
    }
}

impl<I, C, E> template::VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, E>
where
    I: Copy,
    C: Clone,
    E: Clone,
{
    pub fn relabeled<I2: Default + Ord + std::fmt::Debug + Clone>(
        &self,
        mut mapping: impl FnMut(I) -> I2,
    ) -> template::VertexNode<I2, arith::ArithGraph<I2, arith::ExprId>, C, E> {
        use template::VertexNode::*;

        match self {
            NewPoly(deg, init, typ) => NewPoly(*deg, init.clone(), typ.clone()),
            Constant(c) => Constant(c.clone()),
            Extend(s, deg) => Extend(mapping(*s), deg.clone()),
            SingleArith(expr) => SingleArith(expr.relabeled(&mut mapping)),
            Arith { arith, chunking } => Arith {
                arith: arith.relabeled(mapping),
                chunking: *chunking,
            },
            Entry(idx) => Entry(*idx),
            Return(x) => Return(mapping(*x)),
            Ntt { alg, s, to, from } => Ntt {
                alg: alg.relabeled(&mut mapping),
                s: mapping(*s),
                to: to.clone(),
                from: from.clone(),
            },
            RotateIdx(s, deg) => RotateIdx(mapping(*s), *deg),
            Slice(s, start, end) => Slice(mapping(*s), *start, *end),
            Interpolate { xs, ys } => Interpolate {
                xs: xs.iter().map(|x| mapping(*x)).collect(),
                ys: ys.iter().map(|x| mapping(*x)).collect(),
            },
            Blind(s, left, right) => Blind(mapping(*s), *left, *right),
            Array(es) => Array(es.iter().map(|x| mapping(*x)).collect()),
            AssmblePoly(s, es) => AssmblePoly(*s, es.iter().map(|x| mapping(*x)).collect()),
            Msm {
                alg,
                polys: scalars,
                points,
            } => Msm {
                alg: alg.clone(),
                polys: scalars.iter().map(|x| mapping(*x)).collect(),
                points: points.iter().map(|x| mapping(*x)).collect(),
            },
            HashTranscript {
                transcript,
                value,
                typ,
            } => HashTranscript {
                transcript: mapping(*transcript),
                value: mapping(*value),
                typ: typ.clone(),
            },
            SqueezeScalar(transcript) => SqueezeScalar(mapping(*transcript)),
            TupleGet(s, i) => TupleGet(mapping(*s), *i),
            ArrayGet(s, i) => ArrayGet(mapping(*s), *i),
            UserFunction(fid, args) => {
                UserFunction(fid.clone(), args.iter().map(|x| mapping(*x)).collect())
            }
            KateDivision(lhs, rhs) => KateDivision(mapping(*lhs), mapping(*rhs)),
            EvaluatePoly { poly, at } => EvaluatePoly {
                poly: mapping(*poly),
                at: mapping(*at),
            },
            BatchedInvert(s) => BatchedInvert(mapping(*s)),
            ScanMul { x0, poly } => ScanMul {
                x0: mapping(*x0),
                poly: mapping(*poly),
            },
            DistributePowers { poly, powers } => DistributePowers {
                poly: mapping(*poly),
                powers: mapping(*powers),
            },
            ScalarInvert { val } => ScalarInvert { val: mapping(*val) },
            IndexPoly(x, idx) => IndexPoly(mapping(*x), *idx),
            AssertEq(x, y, msg) => AssertEq(mapping(*x), mapping(*y), msg.clone()),
            Print(x, s) => Print(mapping(*x), s.clone()),
        }
    }

    pub fn track(&self, device: Device) -> super::type3::Track {
        use super::type3::Track::*;
        use template::VertexNode::*;

        let on_device = super::type3::Track::on_device;

        match self {
            NewPoly(..) => on_device(device),
            ScalarInvert { .. } => on_device(device),
            Constant(..) => Cpu,
            Extend(..) => on_device(device),
            SingleArith(..) => Gpu,
            Arith { chunking, .. } => {
                if let Some(..) = chunking {
                    CoProcess
                } else {
                    Gpu
                }
            }
            Entry(..) => Cpu,
            Return(..) => MemoryManagement,
            Ntt { .. } => Gpu,
            RotateIdx(..) => unreachable!(),
            Slice(..) => unreachable!(),
            Interpolate { .. } => Cpu,
            Blind(..) => Cpu,
            Array(..) => unreachable!(),
            AssmblePoly(..) => Cpu,
            Msm { .. } => CoProcess,
            HashTranscript { .. } => Cpu,
            SqueezeScalar(..) => Cpu,
            TupleGet(..) => unreachable!(),
            ArrayGet(..) => unreachable!(),
            UserFunction(..) => Cpu,
            KateDivision(..) => Gpu,
            EvaluatePoly { .. } => Gpu,
            BatchedInvert(..) => Gpu,
            ScanMul { .. } => Gpu,
            DistributePowers { .. } => Gpu,
            IndexPoly(..) => on_device(device),
            AssertEq(..) => Cpu,
            Print(..) => Cpu,
        }
    }
}

/// Invariants are same as those of [`tree::tree1::Cg`]
pub type Cg<'s, Rt> = transit::Cg<VertexId, Vertex<'s, Rt>>;

impl<'s, Rt: RuntimeType> Cg<'s, Rt> {
    pub fn temporary_space_needed(
        &self,
        vid: VertexId,
        libs: &mut Libs,
    ) -> Option<(Vec<u64>, super::type3::Device)> {
        use super::type3::Device::*;
        use template::VertexNode::*;
        match self.g.vertex(vid).node() {
            Extend(..) => None,
            ScalarInvert { .. } => None,
            SingleArith(..) => None,
            Arith { .. } => None,
            NewPoly(..) => None,
            Constant(..) => None,
            Return(..) => None,
            Entry(..) => None,
            Ntt { .. } => None,
            RotateIdx(..) => None,
            Slice(..) => None,
            Interpolate { .. } => None,
            Blind(..) => None,
            Array(..) => None,
            AssmblePoly(..) => None,
            Msm { polys, alg, .. } => {
                let (_, len) = self.g.vertex(polys[0]).typ().unwrap_poly();
                Some((temporary_space::msm::<Rt>(alg, *len as usize, libs), Gpu))
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
                    Gpu,
                ))
            }
            EvaluatePoly { poly, .. } => {
                let (_, len) = self.g.vertex(*poly).typ().unwrap_poly();
                Some((temporary_space::poly_eval::<Rt>(*len as usize, libs), Gpu))
            }
            BatchedInvert(poly) => {
                let (_, len) = self.g.vertex(*poly).typ().unwrap_poly();
                Some((temporary_space::poly_invert::<Rt>(*len as usize, libs), Gpu))
            }
            ScanMul { poly, .. } => {
                let (_, len) = self.g.vertex(*poly).typ().unwrap_poly();
                Some((temporary_space::poly_scan::<Rt>(*len as usize, libs), Gpu))
            }
            DistributePowers { .. } => None,
            IndexPoly(..) => None,
            AssertEq(..) => None,
            Print(..) => None,
        }
    }
}

pub struct Program<'s, Rt: RuntimeType> {
    pub(crate) cg: Cg<'s, Rt>,
    pub(crate) user_function_table: user_function::Table<Rt>,
    pub(crate) consant_table: ConstantTable<Rt>,
    pub(crate) memory_pool: PinnedMemoryPool,
}

pub mod decide_device;
pub mod graph_scheduling;
pub mod intt_mending;
pub mod kernel_fusion;
pub mod manage_inverse;
pub mod memory_planning;
pub mod object_analysis;
pub mod precompute;
pub mod pretty_print;
pub mod temporary_space;
pub mod typ;
pub mod user_function;
pub mod visualize;
