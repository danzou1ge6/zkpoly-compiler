//! Data structures for Stage 2 Tree Transit IR.
//! This type of IR has NTT automatically inserted,
//! therefore polynomial operations are transformed into vector expressions.
//! Also, vertices of the computation graph still contain expression trees.

use crate::transit::{self, PolyInit, SourceInfo};
pub use typ::Typ;
use zkpoly_common::arith;
use zkpoly_common::digraph;
use zkpoly_common::heap::UsizeId;
pub use zkpoly_common::typ::PolyType;
pub use zkpoly_runtime::args::{Constant, ConstantId, RuntimeType, Variable};
pub use zkpoly_runtime::error::RuntimeError;

zkpoly_common::define_usize_id!(VertexId);

pub type Arith = arith::ArithGraph<VertexId, arith::ExprId>;

#[derive(Debug, Clone)]
pub enum NttAlgorithm {
    Precomputed,
    Standard,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
    /// If it has any successor or predecesor on GPU, then it should be on GPU
    PreferGpu,
}

impl NttAlgorithm {
    pub fn decide(deg: u64) -> Self {
        if deg <= 2u64.pow(20) {
            NttAlgorithm::Precomputed
        } else {
            NttAlgorithm::Standard
        }
    }
}

impl Default for NttAlgorithm {
    fn default() -> Self {
        NttAlgorithm::Precomputed
    }
}

pub mod template {
    use zkpoly_common::msm_config::MsmConfig;

    use super::{arith, transit, NttAlgorithm, PolyInit, PolyType};

    #[derive(Debug, Clone)]
    pub enum VertexNode<I, A, C, E> {
        NewPoly(u64, PolyInit),
        Constant(C),
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
            alg: NttAlgorithm,
            twiddle_factors: I,
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
    }

    impl<I, C, E> VertexNode<I, arith::ArithGraph<I, arith::ExprId>, C, E>
    where
        I: Copy,
    {
        pub fn uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
            use VertexNode::*;
            match self {
                Arith(arith, ..) => Box::new(arith.uses()),
                Ntt {
                    s, twiddle_factors, ..
                } => Box::new([*s, *twiddle_factors].into_iter()),
                RotateIdx(x, _) => Box::new([*x].into_iter()),
                Interplote { xs, ys } => Box::new(xs.iter().copied().chain(ys.iter().copied())),
                Array(es) => Box::new(es.iter().copied()),
                AssmblePoly(_, es) => Box::new([*es].into_iter()),
                Msm {
                    scalars, points, ..
                } => Box::new([*scalars, *points].into_iter()),
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

        pub fn is_virtual(&self) -> bool {
            use VertexNode::*;
            match self {
                Array(..) | ArrayGet(..) | TupleGet(..) | RotateIdx(..) => true,
                _ => false,
            }
        }
    }
}

// TODO
// - Decide whether to use chunked Arith kernel based on k
// - Temporary space
// - Chunking of polynomial is done during compilation
// - Take inplace into consideration in scheduling and memory planning
// - Twiddle factor precomputing
// - Points precomputing

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
    pub fn temporary_space_needed(&self) -> Option<(u64, super::type3::Device)> {
        use super::type3::Device::*;
        use template::VertexNode::*;
        match self.node() {
            Arith(..) => None,
            NewPoly(..) => None,
            Constant(..) => None,
            Entry => None,
            Return => None,
            LiteralScalar(..) => None,
            Ntt { alg, .. } => None, // Some((temporary_space::ntt::<Rt>(alg), Gpu)),
            RotateIdx(..) => None,
            Interplote { .. } => None,
            Blind(..) => None,
            Array(..) => None,
            AssmblePoly(..) => None,
            Msm { alg, .. } => None, // Some((temporary_space::msm::<Rt>(alg), Gpu)),
            HashTranscript { .. } => None,
            SqueezeScalar(..) => None,
            TupleGet(..) => None,
            ArrayGet(..) => None,
            UserFunction(..) => None,
            KateDivision(..) => todo!(),
            EvaluatePoly { .. } => todo!(),
            BatchedInvert(..) => todo!(),
        }
    }
    pub fn space(&self) -> u64 {
        self.typ().size().total()
    }
    pub fn device(&self) -> Device {
        use template::VertexNode::*;
        match self.node() {
            NewPoly(..) | RotateIdx(..) => Device::PreferGpu,
            Arith(_, chk) => {
                if chk.is_some() {
                    Device::Cpu
                } else {
                    Device::Gpu
                }
            }
            Ntt { .. } => Device::Gpu,
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
            UserFunction(fid, args) => {
                let f_typ = &uf_table[*fid].typ;
                let r = f_typ
                    .args
                    .iter()
                    .zip(args.iter())
                    .filter(|((_, arg_mutability), _)| {
                        arg_mutability == &user_function::Mutability::Mutable
                    })
                    .map(|(_, arg)| *arg);
                Box::new(r)
            }
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
            NewPoly(deg, init) => NewPoly(*deg, init.clone()),
            Constant(c) => Constant(c.clone()),
            Arith(arith, chk) => Arith(arith.relabeled(mapping), *chk),
            Entry => Entry,
            Return => Return,
            LiteralScalar(s) => LiteralScalar(*s),
            Ntt {
                alg,
                s,
                to,
                from,
                twiddle_factors,
            } => Ntt {
                alg: alg.clone(),
                s: mapping(*s),
                to: to.clone(),
                from: from.clone(),
                twiddle_factors: mapping(*twiddle_factors),
            },
            RotateIdx(s, deg) => RotateIdx(mapping(*s), *deg),
            Interplote { xs, ys } => Interplote {
                xs: xs.iter().map(|x| mapping(*x)).collect(),
                ys: ys.iter().map(|x| mapping(*x)).collect(),
            },
            Blind(s, blind) => Blind(mapping(*s), *blind),
            Array(es) => Array(es.iter().map(|x| mapping(*x)).collect()),
            AssmblePoly(s, es) => AssmblePoly(*s, mapping(*es)),
            Msm {
                alg,
                scalars,
                points,
            } => Msm {
                alg: alg.clone(),
                scalars: mapping(*scalars),
                points: mapping(*points),
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
        }
    }

    pub fn track(&self, device: super::type3::Device) -> super::type3::Track {
        use super::type3::Track::*;
        use template::VertexNode::*;

        let on_device = super::type3::Track::on_device;

        match self {
            NewPoly(..) => on_device(device),
            Constant(..) => Cpu,
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
            RotateIdx(..) => on_device(device),
            Interplote { .. } => Cpu,
            Blind(..) => Cpu,
            Array(..) => Cpu,
            AssmblePoly(..) => Cpu,
            Msm { .. } => CoProcess,
            HashTranscript { .. } => Cpu,
            SqueezeScalar(..) => Cpu,
            TupleGet(..) => Cpu,
            ArrayGet(..) => Cpu,
            UserFunction(..) => Cpu,
            KateDivision(..) => Gpu,
            EvaluatePoly { .. } => Gpu,
            BatchedInvert(..) => Gpu,
        }
    }
}

/// Invariants are same as those of [`tree::tree1::Cg`]
pub type Cg<'s, Rt> = transit::Cg<VertexId, Vertex<'s, Rt>>;

pub mod graph_scheduling;
pub mod memory_planning;
pub mod temporary_space;
pub mod typ;
pub mod user_function;
