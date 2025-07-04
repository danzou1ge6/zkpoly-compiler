use zkpoly_common::msm_config::MsmConfig;
use zkpoly_runtime::args::EntryId;

use super::{arith, transit, DevicePreference, NttAlgorithm, PolyInit, PolyType};

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize)]
/// An operation that can be performed on a sliced polynomial,
/// and result of which can be simply concated to obtain the full result
pub enum SliceableNode<I, A, C> {
    NewPoly(u64, PolyInit, PolyType),
    Constant(C),
    SingleArith(arith::Arith<I>),
    ScalarInvert(I),
    Arith { arith: A, chunking: Option<u64> },
    RotateIdx(I, i32),
    Blind(I, u64, u64),
    BatchedInvert(I),
    ScanMul { x0: I, poly: I },
    DistributePowers { poly: I, powers: I },
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize)]
/// These operations can be sliced, but they only gives partial result.
/// So vertices following this node cannot group into a sliceable subgraph
pub enum LastSliceableNode<I> {
    /// Partial MSM result is the inner product between the polynomial slice
    /// and the currespounding points.
    /// To get the full result, we only need to sum those points.
    Msm {
        polys: Vec<I>,
        points: Vec<I>,
        alg: MsmConfig,
    },
    /// Partial polynomial evaluation result of EvaluatePoly([a_0, a_1, ..., a_(n - 1)], x)
    /// is y_i = a_0 + a_1 x + ... + a_(n - 1) x^(n - 1).
    /// Therefore to get the full result, we need to evaluate
    /// y = y_0 + y_1 x^n + ... + y_(m - 1) x^(n (m - 1))
    EvaluatePoly { poly: I, at: I },
}

pub trait SubgraphNode<I>
where
    I: 'static,
{
    fn inputs(&self) -> impl Iterator<Item = &'_ I>;
    fn inputs_mut(&mut self) -> impl Iterator<Item = &'_ mut I>;
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize)]
pub enum VertexNode<I, A, C, E, S> {
    Sliceable(SliceableNode<I, A, C>),
    LastSliceable(LastSliceableNode<I>),
    Subgraph(S),
    Extend(I, u64),
    Entry(EntryId),
    Return(I),
    /// A constant that can not be sliced, e.g. twiddle factors
    UnsliceableConstant(C),
    /// Convert a local from one representation to another
    Ntt {
        s: I,
        to: PolyType,
        from: PolyType,
        alg: NttAlgorithm<I>,
    },
    Slice(I, u64, u64),
    Interpolate {
        xs: Vec<I>,
        ys: Vec<I>,
    },
    Array(Vec<I>),
    AssmblePoly(u64, Vec<I>),
    HashTranscript {
        transcript: I,
        value: I,
        typ: transit::HashTyp,
    },
    /// Returns (transcript, scalar)
    SqueezeScalar(I),
    PolyPermute(I, I, usize),
    TupleGet(I, usize),
    ArrayGet(I, usize),
    UserFunction(E, Vec<I>),
    KateDivision(I, I),
    IndexPoly(I, u64),
    AssertEq(I, I, Option<String>),
    Print(I, String),
}

impl<I, C> SliceableNode<I, super::Arith<I>, C>
where
    I: Clone,
{
    pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use SliceableNode::*;
        match self {
            NewPoly(_, _, _) => Box::new(std::iter::empty()),
            Constant(_) => Box::new(std::iter::empty()),
            SingleArith(arith) => Box::new(arith.uses_ref()),
            ScalarInvert(x) => Box::new([x].into_iter()),
            Arith { arith, .. } => Box::new(arith.uses_ref()),
            RotateIdx(x, _) => Box::new([x].into_iter()),
            Blind(x, _, _) => Box::new([x].into_iter()),
            BatchedInvert(x) => Box::new([x].into_iter()),
            ScanMul { x0, poly } => Box::new([x0, poly].into_iter()),
            DistributePowers { poly, powers } => Box::new([poly, powers].into_iter()),
        }
    }

    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use SliceableNode::*;
        match self {
            NewPoly(_, _, _) => Box::new(std::iter::empty()),
            Constant(_) => Box::new(std::iter::empty()),
            SingleArith(arith) => Box::new(arith.uses_mut()),
            ScalarInvert(x) => Box::new([x].into_iter()),
            Arith { arith, .. } => Box::new(arith.uses_mut()),
            RotateIdx(x, _) => Box::new([x].into_iter()),
            Blind(x, _, _) => Box::new([x].into_iter()),
            BatchedInvert(x) => Box::new([x].into_iter()),
            ScanMul { x0, poly } => Box::new([x0, poly].into_iter()),
            DistributePowers { poly, powers } => Box::new([poly, powers].into_iter()),
        }
    }

    pub fn device(&self) -> DevicePreference {
        use DevicePreference::*;
        use SliceableNode::*;
        match self {
            NewPoly(..) => PreferGpu,
            Constant(..) => Cpu,
            SingleArith(arith) => match arith {
                arith::Arith::Unr(arith::UnrOp::P(arith::ArithUnrOp::Pow(..)), _)
                | arith::Arith::Unr(arith::UnrOp::S(arith::ArithUnrOp::Pow(..)), _) => PreferGpu,
                _ => Gpu,
            },
            ScalarInvert(..) => PreferGpu,
            Arith { chunking, .. } => {
                if chunking.is_some() {
                    Cpu
                } else {
                    Gpu
                }
            }
            RotateIdx(..) => Cpu,
            Blind(..) => Cpu,
            BatchedInvert(..) => Gpu,
            ScanMul { .. } => Gpu,
            DistributePowers { .. } => Gpu,
        }
    }

    pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        use SliceableNode::*;
        match self {
            Arith { arith, .. } => arith.mutable_uses(),
            Blind(poly, ..) => Box::new([poly.clone()].into_iter()),
            BatchedInvert(x) => Box::new([x.clone()].into_iter()),
            DistributePowers { poly, .. } => Box::new([poly.clone()].into_iter()),
            SingleArith(arith::Arith::Unr(arith::UnrOp::S(arith::ArithUnrOp::Pow(_)), scalar)) => {
                Box::new([scalar.clone()].into_iter())
            }
            ScalarInvert(val) => Box::new([val.clone()].into_iter()),
            _ => Box::new(std::iter::empty()),
        }
    }

    pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use SliceableNode::*;
        match self {
            Arith { arith, .. } => arith.mutable_uses_mut(),
            Blind(poly, ..) => Box::new([poly].into_iter()),
            BatchedInvert(x) => Box::new([x].into_iter()),
            DistributePowers { poly, .. } => Box::new([poly].into_iter()),
            SingleArith(arith::Arith::Unr(arith::UnrOp::S(arith::ArithUnrOp::Pow(_)), scalar)) => {
                Box::new([scalar].into_iter())
            }
            ScalarInvert(val) => Box::new([val].into_iter()),
            _ => Box::new(std::iter::empty()),
        }
    }

    pub fn outputs_inplace<'a>(&'a self) -> Box<dyn Iterator<Item = Option<I>> + 'a> {
        use SliceableNode::*;
        match self {
            SingleArith(arith::Arith::Unr(arith::UnrOp::S(arith::ArithUnrOp::Pow(_)), scalar)) => {
                Box::new(std::iter::once(Some(scalar.clone())))
            }
            ScalarInvert(val) => Box::new(std::iter::once(Some(val.clone()))),
            Arith { arith, .. } => arith.outputs_inplace(),
            Blind(poly, ..) => Box::new(std::iter::once(Some(poly.clone()))),
            BatchedInvert(poly) => Box::new(std::iter::once(Some(poly.clone()))),
            DistributePowers { poly, .. } => Box::new(std::iter::once(Some(poly.clone()))),
            _ => Box::new(std::iter::repeat(None)),
        }
    }

    pub fn try_relabeled<I2, Er>(
        &self,
        mut mapping: impl FnMut(I) -> Result<I2, Er>,
    ) -> Result<SliceableNode<I2, super::Arith<I2>, C>, Er>
    where
        C: Clone,
        I2: Default + Ord + std::fmt::Debug + Clone,
    {
        use SliceableNode::*;
        Ok(match self {
            NewPoly(deg, init, typ) => NewPoly(deg.clone(), init.clone(), typ.clone()),
            Constant(c) => Constant(c.clone()),
            SingleArith(a) => SingleArith(a.try_relabeled(&mut mapping)?),
            ScalarInvert(x) => ScalarInvert(mapping(x.clone())?),
            Arith { arith, chunking } => Arith {
                arith: arith.try_relabeled(&mut mapping)?,
                chunking: chunking.clone(),
            },
            RotateIdx(x, n) => RotateIdx(mapping(x.clone())?, n.clone()),
            Blind(poly, n, m) => Blind(mapping(poly.clone())?, n.clone(), m.clone()),
            BatchedInvert(poly) => BatchedInvert(mapping(poly.clone())?),
            ScanMul { x0, poly } => ScanMul {
                x0: mapping(x0.clone())?,
                poly: mapping(poly.clone())?,
            },
            DistributePowers { poly, powers } => DistributePowers {
                poly: mapping(poly.clone())?,
                powers: mapping(powers.clone())?,
            },
        })
    }

    pub fn relabeled<I2>(
        &self,
        mut mapping: impl FnMut(I) -> I2,
    ) -> SliceableNode<I2, super::Arith<I2>, C>
    where
        I2: Default + Ord + std::fmt::Debug + Clone,
        C: Clone,
    {
        self.try_relabeled::<I2, ()>(|x| Ok(mapping(x))).unwrap()
    }

    pub fn track(&self, device: super::Device) -> super::super::type3::Track {
        use super::super::type3::Track::*;
        use SliceableNode::*;

        let on_device = super::super::type3::Track::on_device;
        let currespounding_gpu = |dev: super::Device| {
            if !device.is_gpu() {
                panic!("this vertex needs to be executed on some GPU");
            }
            on_device(dev)
        };

        match self {
            NewPoly(..) => on_device(device),
            Constant(..) => Cpu,
            SingleArith(..) => currespounding_gpu(device),
            ScalarInvert(..) => on_device(device),
            Arith { chunking, .. } => {
                if chunking.is_some() {
                    CoProcess
                } else {
                    currespounding_gpu(device)
                }
            }
            RotateIdx(..) => panic!("this vertex is not actually executed"),
            Blind(..) => Cpu,
            BatchedInvert(..) => currespounding_gpu(device),
            ScanMul { .. } => currespounding_gpu(device),
            DistributePowers { .. } => currespounding_gpu(device),
        }
    }
}

impl<I> LastSliceableNode<I>
where
    I: Clone,
{
    pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use LastSliceableNode::*;
        match self {
            Msm { polys, points, .. } => Box::new(polys.iter().chain(points.iter())),
            EvaluatePoly { poly, at } => Box::new([poly, at].into_iter()),
        }
    }

    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use LastSliceableNode::*;
        match self {
            Msm { polys, points, .. } => Box::new(polys.iter_mut().chain(points.iter_mut())),
            EvaluatePoly { poly, at } => Box::new([poly, at].into_iter()),
        }
    }

    pub fn device(&self) -> DevicePreference {
        use DevicePreference::*;
        use LastSliceableNode::*;
        match self {
            Msm { .. } => Gpu,
            EvaluatePoly { .. } => Gpu,
        }
    }

    pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
        Box::new(std::iter::empty())
    }

    pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        Box::new(std::iter::empty())
    }

    pub fn outputs_inplace<'a>(&'a self) -> Box<dyn Iterator<Item = Option<I>> + 'a> {
        Box::new(std::iter::repeat(None))
    }

    pub fn try_relabeld<I2, Er>(
        &self,
        mut mapping: impl FnMut(I) -> Result<I2, Er>,
    ) -> Result<LastSliceableNode<I2>, Er>
    where
        I2: Default + Ord + std::fmt::Debug + Clone,
    {
        use LastSliceableNode::*;
        Ok(match self {
            Msm { polys, points, alg } => Msm {
                polys: polys
                    .iter()
                    .map(|x| mapping(x.clone()))
                    .collect::<Result<_, _>>()?,
                points: points
                    .iter()
                    .map(|x| mapping(x.clone()))
                    .collect::<Result<_, _>>()?,
                alg: alg.clone(),
            },
            EvaluatePoly { poly, at } => EvaluatePoly {
                poly: mapping(poly.clone())?,
                at: mapping(at.clone())?,
            },
        })
    }

    pub fn relabeled<I2>(&self, mut mapping: impl FnMut(I) -> I2) -> LastSliceableNode<I2>
    where
        I2: Default + Ord + std::fmt::Debug + Clone,
    {
        self.try_relabeld::<I2, ()>(|x| Ok(mapping(x))).unwrap()
    }

    pub fn track(&self, device: super::Device) -> super::super::type3::Track {
        use super::super::type3::Track::*;
        use LastSliceableNode::*;

        let on_device = super::super::type3::Track::on_device;
        let currespounding_gpu = |dev: super::Device| {
            if !device.is_gpu() {
                panic!("this vertex needs to be executed on some GPU");
            }
            on_device(dev)
        };

        match self {
            Msm { .. } => CoProcess,
            EvaluatePoly { .. } => currespounding_gpu(device),
        }
    }
}

impl<I, C, E, S> VertexNode<I, super::Arith<I>, C, E, S>
where
    I: Clone + 'static,
    S: SubgraphNode<I>,
{
    pub fn is_sliceable(&self) -> bool {
        matches!(self, Self::Sliceable(_))
    }

    pub fn is_last_sliceable(&self) -> bool {
        matches!(self, Self::LastSliceable(_))
    }

    pub fn unwrap_constant(&self) -> &C {
        match self {
            Self::UnsliceableConstant(constant) => constant,
            Self::Sliceable(SliceableNode::Constant(c)) => c,
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
            Self::LastSliceable(LastSliceableNode::Msm { polys, points, alg }) => {
                (polys, points, alg)
            }
            _ => panic!("this is not msm"),
        }
    }
    pub fn uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
        use VertexNode::*;
        match self {
            Sliceable(sn) => sn.uses_mut(),
            LastSliceable(lsn) => lsn.uses_mut(),
            Subgraph(s) => Box::new(s.inputs_mut()),
            Extend(x, _) => Box::new([x].into_iter()),
            Ntt { s, alg, .. } => Box::new([s].into_iter().chain(alg.uses_mut())),
            Slice(x, ..) => Box::new([x].into_iter()),
            Interpolate { xs, ys } => Box::new(xs.iter_mut().chain(ys.iter_mut())),
            Array(es) => Box::new(es.iter_mut()),
            HashTranscript {
                transcript, value, ..
            } => Box::new([transcript, value].into_iter()),
            SqueezeScalar(x) => Box::new([x].into_iter()),
            TupleGet(x, _) => Box::new([x].into_iter()),
            ArrayGet(x, _) => Box::new([x].into_iter()),
            UserFunction(_, es) => Box::new(es.iter_mut()),
            KateDivision(lhs, rhs) => Box::new([lhs, rhs].into_iter()),
            Return(x) => Box::new([x].into_iter()),
            IndexPoly(x, _) => Box::new([x].into_iter()),
            AssertEq(x, y, _msg) => Box::new([x, y].into_iter()),
            Print(x, _) => Box::new([x].into_iter()),
            PolyPermute(input, table, _) => Box::new([input, table].into_iter()),
            AssmblePoly(_, xs) => Box::new(xs.iter_mut()),
            Entry(_) | UnsliceableConstant(_) => Box::new(std::iter::empty()),
        }
    }

    pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
        use VertexNode::*;
        match self {
            Sliceable(sn) => sn.uses_ref(),
            LastSliceable(lsn) => lsn.uses_ref(),
            Subgraph(s) => Box::new(s.inputs()),
            Extend(x, _) => Box::new([x].into_iter()),
            Ntt { s, alg, .. } => Box::new([s].into_iter().chain(alg.uses_ref())),
            Slice(x, ..) => Box::new([x].into_iter()),
            Interpolate { xs, ys } => Box::new(xs.iter().chain(ys.iter())),
            Array(es) => Box::new(es.iter()),
            AssmblePoly(_, es) => Box::new(es.iter()),
            HashTranscript {
                transcript, value, ..
            } => Box::new([transcript, value].into_iter()),
            SqueezeScalar(x) => Box::new([x].into_iter()),
            TupleGet(x, _) => Box::new([x].into_iter()),
            ArrayGet(x, _) => Box::new([x].into_iter()),
            UserFunction(_, es) => Box::new(es.iter()),
            KateDivision(lhs, rhs) => Box::new([lhs, rhs].into_iter()),
            Return(x) => Box::new([x].into_iter()),
            IndexPoly(x, _) => Box::new([x].into_iter()),
            AssertEq(x, y, _msg) => Box::new([x, y].into_iter()),
            Print(x, _) => Box::new([x].into_iter()),
            PolyPermute(input, table, _) => Box::new([input, table].into_iter()),
            Entry(_) | UnsliceableConstant(_) => Box::new(std::iter::empty()),
        }
    }

    pub fn uses<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.uses_ref().cloned()
    }

    pub fn device(&self) -> DevicePreference {
        use DevicePreference::*;
        use VertexNode::*;
        match self {
            Sliceable(sn) => sn.device(),
            LastSliceable(lsn) => lsn.device(),
            IndexPoly { .. } => PreferGpu,
            Ntt { .. } => Gpu,
            KateDivision(_, _) => Gpu,
            PolyPermute(_, _, _) => Gpu,
            _ => Cpu,
        }
    }

    pub fn no_supports_sliced_inputs(&self) -> bool {
        use VertexNode::*;
        match self {
            LastSliceable(LastSliceableNode::Msm { .. }) => true,
            Ntt { .. } => true,
            Sliceable(SliceableNode::Arith { chunking, .. }) => chunking.is_some(),
            Sliceable(SliceableNode::ScanMul { .. }) => true,
            _ => false,
        }
    }

    pub fn is_virtual(&self) -> bool {
        use VertexNode::*;
        match self {
            Array(..)
            | ArrayGet(..)
            | TupleGet(..)
            | Sliceable(SliceableNode::RotateIdx(..))
            | Slice(..) => true,
            _ => false,
        }
    }

    pub fn unexpcted_during_kernel_gen(&self) -> bool {
        use VertexNode::*;
        match self {
            Extend(..) | Sliceable(SliceableNode::NewPoly(..)) => true,
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

    pub fn immortal(&self) -> bool {
        use VertexNode::*;
        match self {
            Sliceable(SliceableNode::Constant(..)) => true,
            UnsliceableConstant(..) => true,
            Entry(..) => true,
            _ => false,
        }
    }
}
