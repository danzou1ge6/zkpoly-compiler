//! Data structures for Stage 2 Tree Transit IR.
//! This type of IR has NTT automatically inserted,
//! therefore polynomial operations are transformed into vector expressions.
//! Also, vertices of the computation graph still contain expression trees.

use crate::transit::{self, PolyInit, SourceInfo};
pub use typ::Typ;
use zkpoly_common::arith;
use zkpoly_common::digraph;
use zkpoly_common::heap::UsizeId;
use zkpoly_common::load_dynamic::Libs;
pub use zkpoly_common::typ::PolyType;
pub use zkpoly_runtime::args::{Constant, ConstantId, RuntimeType, Variable};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
    /// If it has any successor or predecesor on GPU, then it should be on GPU
    PreferGpu,
}

pub mod template {
    use zkpoly_common::msm_config::MsmConfig;

    use super::{arith, transit, NttAlgorithm, PolyInit, PolyType};

    #[derive(Debug, Clone)]
    pub enum VertexNode<I, A, C, E> {
        NewPoly(u64, PolyInit, PolyType),
        Constant(C),
        Extend(I, u64),
        /// A single binary or unary arith expression, to be fused to an arith graph
        SingleArith(arith::Arith<I>),
        /// .1 is size of each chunk, if chunking is enabled
        Arith(A, Option<u64>),
        Entry,
        Return,
        /// A small scalar. To accomodate big scalars, use global constants.
        LiteralScalar(usize),
        /// Convert a local from one representation to another
        Ntt {
            s: I,
            to: PolyType,
            from: PolyType,
            alg: NttAlgorithm<I>,
        },
        RotateIdx(I, i32),
        Slice(I, u64, u64),
        Interplote {
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
            scalar: I,
            poly: I,
        },
    }

    impl<I, C, E> VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, E>
    where
        I: Copy,
    {
        pub fn uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
            use VertexNode::*;
            match self {
                SingleArith(expr) => expr.uses(),
                Arith(arith, ..) => Box::new(arith.uses()),
                Ntt { s, alg, .. } => Box::new([*s].into_iter().chain(alg.uses())),
                RotateIdx(x, _) => Box::new([*x].into_iter()),
                Slice(x, ..) => Box::new([*x].into_iter()),
                Interplote { xs, ys } => Box::new(xs.iter().copied().chain(ys.iter().copied())),
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
                DistributePowers { scalar, poly } => Box::new([*scalar, *poly].into_iter()),
                _ => Box::new(std::iter::empty()),
            }
        }

        pub fn is_virtual(&self) -> bool {
            use VertexNode::*;
            match self {
                Array(..) | ArrayGet(..) | TupleGet(..) | RotateIdx(..) | Slice(..) => true,
                _ => false,
            }
        }

        pub fn unexpcted_during_lowering(&self) -> bool {
            use VertexNode::*;
            match self {
                SingleArith(..) => true,
                x => x.is_virtual(),
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

impl<'s, Rt: RuntimeType> digraph::internal::Predecessors<VertexId> for Vertex<'s, Rt> {
    #[allow(refining_impl_trait)]
    fn predecessors<'a>(&'a self) -> Box<dyn Iterator<Item = VertexId> + 'a> {
        self.uses()
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
    pub fn space(&self) -> u64 {
        self.typ().size().total()
    }
    pub fn device(&self) -> Device {
        use template::VertexNode::*;
        match self.node() {
            NewPoly(..) | Extend(..) => Device::PreferGpu,
            Arith(_, chk) => {
                        if chk.is_some() {
                            Device::Cpu
                        } else {
                            Device::Gpu
                        }
                    }
            Ntt { .. } => Device::Gpu,
            KateDivision(_, _) => Device::Gpu,
            EvaluatePoly { .. } => Device::Gpu,
            BatchedInvert(_) => Device::Gpu,
            ScanMul(_) => Device::Gpu,
            DistributePowers { .. } => Device::Gpu,
            _ => Device::Cpu,
        }
    }
    pub fn mutable_uses<'a>(
        &'a self,
        uf_table: &'a user_function::Table<Rt>,
    ) -> Box<dyn Iterator<Item = I> + 'a> {
        use template::VertexNode::*;
        match self.node() {
            Ntt { s, .. } => Box::new([*s].into_iter()),
            RotateIdx(s, ..) => Box::new([*s].into_iter()),
            HashTranscript { transcript, .. } => Box::new([*transcript].into_iter()),
            SqueezeScalar(transcript) => Box::new([*transcript].into_iter()),
            Arith(_, _) => todo!(),
            Blind(poly, ..) => Box::new([*poly].into_iter()),
            BatchedInvert(poly) => Box::new([*poly].into_iter()),
            DistributePowers {  poly, .. } => Box::new([*poly].into_iter()),
            _ => Box::new([].into_iter()),
        }
    }
    pub fn mutable_uses_mut<'a>(
        &'a mut self,
        uf_table: &'a user_function::Table<Rt>,
    ) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use template::VertexNode::*;
        match self.node_mut() {
            Ntt { s, .. } => Box::new([s].into_iter()),
            RotateIdx(s, ..) => Box::new([s].into_iter()),
            HashTranscript { transcript, .. } => Box::new([transcript].into_iter()),
            SqueezeScalar(transcript) => Box::new([transcript].into_iter()),
            Arith(_, _) => todo!(),
            Blind(poly, ..) => Box::new([poly].into_iter()),
            BatchedInvert(poly) => Box::new([poly].into_iter()),
            DistributePowers {  poly, .. } => Box::new([poly].into_iter()),
            UserFunction(fid, args) => {
                let f_typ = &uf_table[*fid].typ;
                let r = f_typ
                    .args
                    .iter()
                    .zip(args.iter_mut())
                    .filter(|((_, arg_mutability), _)| {
                        arg_mutability == &user_function::Mutability::Mutable
                    })
                    .map(|(_, arg)| arg);
                Box::new(r)
            }
            _ => Box::new([].into_iter()),
        }
    }
    pub fn outputs_inplace<'a, 'b>(
        &'b self,
        uf_table: &'a user_function::Table<Rt>,
        device: super::type3::Device,
    ) -> Box<dyn Iterator<Item = Option<I>> + 'b> {
        use template::VertexNode::*;
        match self.node() {
            Ntt { s, .. } => Box::new([Some(*s)].into_iter()),
            RotateIdx(s, ..) => {
                use super::type3::Device::*;
                match device {
                    Cpu => Box::new([Some(*s)].into_iter()),
                    Gpu => Box::new([None].into_iter()),
                    Stack => panic!("RotateIdx output can't be on stack"),
                }
            }
            Arith(_, _) => todo!(),
            Blind(poly, ..) => Box::new([Some(*poly)].into_iter()),
            BatchedInvert(poly) => Box::new([Some(*poly)].into_iter()),
            DistributePowers {  poly, .. } => Box::new([Some(*poly)].into_iter()),
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
    pub fn relabeled<I2>(
        &self,
        mut mapping: impl FnMut(I) -> I2,
    ) -> template::VertexNode<I2, arith::ArithGraph<I2, arith::ExprId>, C, E> {
        use template::VertexNode::*;

        match self {
            NewPoly(deg, init, typ) => NewPoly(*deg, init.clone(), typ.clone()),
            Constant(c) => Constant(c.clone()),
            Extend(s, deg) => Extend(mapping(*s), deg.clone()),
            SingleArith(expr) => SingleArith(expr.relabeled(&mut mapping)),
            Arith(arith, chk) => Arith(arith.relabeled(mapping), *chk),
            Entry => Entry,
            Return => Return,
            LiteralScalar(s) => LiteralScalar(*s),
            Ntt { alg, s, to, from } => Ntt {
                alg: alg.relabeled(&mut mapping),
                s: mapping(*s),
                to: to.clone(),
                from: from.clone(),
            },
            RotateIdx(s, deg) => RotateIdx(mapping(*s), *deg),
            Slice(s, start, end) => Slice(mapping(*s), *start, *end),
            Interplote { xs, ys } => Interplote {
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
            DistributePowers { poly, scalar } => DistributePowers {
                poly: mapping(*poly),
                scalar: mapping(*scalar),
            },
        }
    }

    pub fn track(&self, device: super::type3::Device) -> super::type3::Track {
        use super::type3::Track::*;
        use template::VertexNode::*;

        let on_device = super::type3::Track::on_device;

        if self.unexpcted_during_lowering() {
            panic!("vertex is unexpected during lowering, it shouldn't be on any track")
        }

        match self {
            NewPoly(..) => on_device(device),
            Constant(..) => Cpu,
            Extend(..) => on_device(device),
            SingleArith(..) => unreachable!(),
            Arith(_, chk) => {
                if let Some(..) = chk {
                    CoProcess
                } else {
                    Gpu
                }
            }
            Entry => Cpu,
            Return => Cpu,
            LiteralScalar(..) => Cpu,
            Ntt { .. } => CoProcess,
            RotateIdx(..) => unreachable!(),
            Slice(..) => unreachable!(),
            Interplote { .. } => Cpu,
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
            SingleArith(..) => None,
            Arith(..) => None,
            NewPoly(..) => None,
            Constant(..) => None,
            Entry => None,
            Return => None,
            LiteralScalar(..) => None,
            Ntt { .. } => None,
            RotateIdx(..) => None,
            Slice(..) => None,
            Interplote { .. } => None,
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
            DistributePowers { .. } => todo!(),
        }
    }
}

pub mod graph_scheduling;
pub mod memory_planning;
pub mod temporary_space;
pub mod typ;
pub mod user_function;
