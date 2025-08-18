use super::super::Arith;
use super::{LastSliceableNode, SliceableNode, SubgraphNode, VertexNode};
use std::fmt::{Debug, Display};
use zkpoly_common::arith;

pub enum EdgeLabel {
    Enumerate(&'static str, usize),
    Plain(&'static str),
}

impl std::fmt::Display for EdgeLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use EdgeLabel::*;
        match self {
            Enumerate(x, i) => write!(f, "{x}{i}"),
            Plain(x) => write!(f, "{x}"),
        }
    }
}

pub struct LabelT<'a, T>(pub &'a T);

impl<'a, I, C> Display for LabelT<'a, SliceableNode<I, Arith<I>, C>>
where
    C: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use SliceableNode::*;
        match &self.0 {
            NewPoly(deg, ..) => write!(f, "NewPoly(deg={})", deg),
            Constant(cid) => write!(f, "Constant({:?})", cid),
            SingleArith(arith::Arith::Bin(op, ..)) => {
                write!(f, "SingleArith({:?})", op)
            }
            SingleArith(arith::Arith::Unr(op, ..)) => {
                write!(f, "SingleArith({:?})", op)
            }
            Arith {
                chunking: Some(chunking),
                ..
            } => write!(f, "Arith({})", chunking),
            Arith { chunking: None, .. } => write!(f, "Arith",),
            RotateIdx(_, idx) => write!(f, "Rotate({})", idx),
            Blind(_, left, right) => write!(f, "Blind[{}:{}]", left, right),
            BatchedInvert(_) => write!(f, "BatchInvert"),
            ScanMul { .. } => write!(f, "ScanMul"),
            DistributePowers { .. } => write!(f, "DistPowers"),
            ScalarInvert { .. } => write!(f, "ScalarInvert"),
        }
    }
}

impl<I, C> SliceableNode<I, Arith<I>, C> {
    pub fn labeled_uses<'a>(&'a self) -> Box<dyn Iterator<Item = (I, Option<EdgeLabel>)> + 'a>
    where
        I: Clone,
    {
        use SliceableNode::*;
        match self {
            ScanMul { x0, poly } => Box::new(
                [
                    (x0.clone(), Some(EdgeLabel::Plain("x0"))),
                    (poly.clone(), Some(EdgeLabel::Plain("poly"))),
                ]
                .into_iter(),
            ),
            DistributePowers { poly, powers } => Box::new(
                [
                    (poly.clone(), Some(EdgeLabel::Plain("poly"))),
                    (powers.clone(), Some(EdgeLabel::Plain("powers"))),
                ]
                .into_iter(),
            ),
            _ => Box::new(self.uses_ref().map(|u| (u.clone(), None))),
        }
    }
}

impl<'a, I> Display for LabelT<'a, LastSliceableNode<I>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use LastSliceableNode::*;
        match &self.0 {
            Msm { .. } => write!(f, "Msm"),
            EvaluatePoly { .. } => write!(f, "EvaluatePoly"),
        }
    }
}

impl<I> LastSliceableNode<I> {
    pub fn labeled_uses<'a>(&'a self) -> Box<dyn Iterator<Item = (I, Option<EdgeLabel>)> + 'a>
    where
        I: Clone,
    {
        use LastSliceableNode::*;
        match self {
            Msm { polys, points, .. } => Box::new(
                polys
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (p.clone(), Some(EdgeLabel::Enumerate("poly", i))))
                    .chain(
                        points
                            .iter()
                            .enumerate()
                            .map(|(i, p)| (p.clone(), Some(EdgeLabel::Enumerate("point", i)))),
                    ),
            ),
            _ => Box::new(self.uses_ref().map(|u| (u.clone(), None))),
        }
    }
}

impl<'a, I, C, E, S> Display for LabelT<'a, VertexNode<I, Arith<I>, C, E, S>>
where
    C: Debug,
    E: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use VertexNode::*;
        match &self.0 {
            Sliceable(sn) => write!(f, "{}", LabelT(sn)),
            LastSliceable(lsn) => write!(f, "{}", LabelT(lsn)),
            Subgraph(_) => write!(f, "Subgraph"),
            UnsliceableConstant(c) => write!(f, "Constant({:?})", c),
            Extend(_, to) => write!(f, "Extend({})", to),
            PolyPermute(_input, _table, usable_len) => {
                write!(f, "PolyPermute({})", usable_len)
            }
            Entry(id) => write!(f, "Entry({:?})", id),
            Return(_) => write!(f, "Return"),
            Ntt { from, to, .. } => write!(f, "NTT({:?} -> {:?})", from, to),
            Slice(_, start, end) => write!(f, "Slice[{}:{}]", start, end),
            Interpolate { .. } => write!(f, "Interpolate"),
            Array(_) => write!(f, "Array"),
            AssmblePoly(deg, _) => write!(f, "AssemblePoly(deg={})", deg),
            HashTranscript { typ, .. } => write!(f, "Hash({:?})", typ),
            SqueezeScalar(_) => write!(f, "SqueezeScalar"),
            TupleGet(_, idx) => write!(f, "TupleGet({})", idx),
            ArrayGet(_, idx) => write!(f, "ArrayGet({})", idx),
            UserFunction(id, _) => write!(f, "UserFunc({:?})", id),
            KateDivision(..) => write!(f, "KateDivision"),
            IndexPoly(_, idx) => write!(f, "IndexPoly({})", idx),
            AssertEq(..) => write!(f, "AssertEq"),
            Print(_, label) => write!(f, "Print({})", label),
        }
    }
}

impl<I, C, E, S> VertexNode<I, Arith<I>, C, E, S> {
    pub fn labeled_uses<'a>(&'a self) -> Box<dyn Iterator<Item = (I, Option<EdgeLabel>)> + 'a>
    where
        I: Clone + 'static,
        S: SubgraphNode<I>,
    {
        use VertexNode::*;
        match self {
            Interpolate { xs, ys } => Box::new(
                xs.iter()
                    .cloned()
                    .enumerate()
                    .map(|(i, x)| (x, Some(EdgeLabel::Enumerate("x", i))))
                    .chain(
                        ys.iter()
                            .cloned()
                            .enumerate()
                            .map(|(i, y)| (y, Some(EdgeLabel::Enumerate("y", i)))),
                    ),
            ),
            Array(xs) | AssmblePoly(_, xs) => Box::new(
                xs.iter()
                    .cloned()
                    .enumerate()
                    .map(|(i, x)| (x, Some(EdgeLabel::Enumerate("x", i)))),
            ),
            _ => Box::new(self.uses_ref().map(|u| (u.clone(), None))),
        }
    }
}
