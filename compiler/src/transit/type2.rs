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

#[derive(Debug, Clone)]
pub enum MsmAlgorithm {
    Batched,
    Sequential,
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
            alg: MsmAlgorithm,
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

// TODO
// - A should be Ag
// - Evaluate should be out of Ag
// - Decide whether to use chunked Arith kernel based on k
// - Temporary space
// - Rotation merged to transfer, whenever possible
// - Chunking of polynomial is done in Ag kernels
// - After memory planning, for each rotation, track transfer of value before and after the rotation,
//   if a transfer can be found, then use latest transfer to rotate the value
// - Problem: How is rotation performed when it cannot be merged to transfer?
//     + Use a rotation offset
//     + Rotate it physically
//   Anyway, they are GPU or CPU kernels
// - Inplace correction before scheduling
// - Take inplace into consideration in scheduling and memory planning
// - Twiddle factor precomputing

pub type VertexNode = template::VertexNode<VertexId, Arith, ConstantId, user_function::Id>;

pub type Vertex<'s, Rt> = transit::Vertex<VertexNode, Typ<Rt>, SourceInfo<'s>>;

impl<'s, Rt: RuntimeType> digraph::internal::Predecessors<VertexId> for Vertex<'s, Rt> {
    #[allow(refining_impl_trait)]
    fn predecessors<'a>(&'a self) -> Box<dyn Iterator<Item = VertexId> + 'a> {
        self.uses()
    }
}

impl<'s, Rt: RuntimeType> Vertex<'s, Rt> {
    pub fn uses<'a>(&'a self) -> Box<dyn Iterator<Item = VertexId> + 'a> {
        use template::VertexNode::*;
        match self.node() {
            Arith(arith, ..) => Box::new(arith.uses()),
            Ntt { s, .. } => Box::new([*s].into_iter()),
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
    pub fn temporary_space_needed(&self) -> u64 {
        use template::VertexNode::*;
        match self.node() {
            Arith(..) => 0,
            NewPoly(..) => 0,
            Constant(..) => 0,
            Entry => 0,
            Return => 0,
            LiteralScalar(..) => 0,
            Ntt { alg, .. } => alg.temporary_space_needed::<Rt::Field>(self.typ().unwrap_poly().1),
            RotateIdx(..) => 0,
            Interplote { .. } => 0,
            Blind(..) => 0,
            Array(..) => 0,
            AssmblePoly(..) => 0,
            Msm { alg, .. } => alg.temporary_space_needed::<Rt::Field>(self.typ().unwrap_poly().1),
            HashTranscript { .. } => 0,
            SqueezeScalar(..) => 0,
            TupleGet(..) => 0,
            ArrayGet(..) => 0,
            UserFunction(..) => 0,
        }
    }
    pub fn space(&self) -> u64 {
        self.typ().size().total()
    }
    pub fn device(&self) -> Device {
        use template::VertexNode::*;
        match self.node() {
            NewPoly(..) | RotateIdx(..) => Device::PreferGpu,
            Arith(_, chk) => if chk.is_some() {
                Device::Cpu
            } else {
                Device::Gpu
            },
            Ntt { .. } => Device::Gpu,
            _ => Device::Cpu,
        }
    }
    pub fn mutable_uses<'a>(
        &'a self,
        uf_table: &'a user_function::Table<Rt>,
    ) -> Box<dyn Iterator<Item = VertexId> + 'a> {
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
    ) -> Box<dyn Iterator<Item = &'a mut VertexId> + 'a> {
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
    pub fn outputs_inplace<'a>(
        &self,
        uf_table: &'a user_function::Table<Rt>,
    ) -> Box<dyn Iterator<Item = Option<VertexId>>> {
        use template::VertexNode::*;
        match self.node() {
            Ntt { s, .. } => Box::new([Some(*s)].into_iter()),
            RotateIdx(s, ..) => Box::new([Some(*s)].into_iter()),
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
        use template::VertexNode::*;
        match self.node() {
            Array(..) | ArrayGet(..) | TupleGet(..) => true,
            _ => false,
        }
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
            Ntt { alg, s, to, from } => Ntt {
                alg: alg.clone(),
                s: mapping(*s),
                to: to.clone(),
                from: from.clone(),
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
        }
    }
}

/// Invariants are same as those of [`tree::tree1::Cg`]
pub type Cg<'s, Rt> = transit::Cg<VertexId, Vertex<'s, Rt>>;

pub mod graph_scheduling;
pub mod memory_planning;
pub mod typ;
pub mod user_function;
