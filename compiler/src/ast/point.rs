use zkpoly_memory_pool::PinnedMemoryPool;

use super::*;
use crate::utils::log2;

#[derive(Debug, Clone)]
pub struct PrecomputedPointsData<Rt: RuntimeType>(rt::point::PointArray<Rt::PointAffine>, u32);

pub type PrecomputedPoints<Rt: RuntimeType> = Outer<PrecomputedPointsData<Rt>>;

impl<Rt: RuntimeType> PrecomputedPoints<Rt> {
    #[track_caller]
    pub fn construct(points: &[Rt::PointAffine], allocator: &mut PinnedMemoryPool) -> Self {
        if let Some(log_n) = log2(points.len() as u64) {
            let src = SourceInfo::new(Location::caller().clone(), None);
            Self::new(
                PrecomputedPointsData(rt::point::PointArray::from_vec(points, allocator), log_n),
                src,
            )
        } else {
            panic!("points length must be power of 2");
        }
    }
}

impl<Rt: RuntimeType> TypeEraseable<Rt> for PrecomputedPoints<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        cg.lookup_or_insert_with(self.as_ptr(), |cg| {
            let constant = cg.add_constant(
                PrecomputedPoints::to_variable(self.inner.t.0.clone()),
                None,
                zkpoly_common::typ::Typ::PointBase {
                    len: 2usize.pow(self.inner.t.1),
                },
            );
            Vertex::new(
                VertexNode::Constant(constant),
                Some(Typ::PointBase {
                    log_n: self.inner.t.1,
                }),
                self.src_lowered(),
            )
        })
    }
}

impl<Rt: RuntimeType> RuntimeCorrespondance<Rt> for PrecomputedPoints<Rt> {
    type Rtc = rt::point::PointArray<Rt::PointAffine>;
    type RtcBorrowed<'a> = &'a Self::Rtc;
    type RtcBorrowedMut<'a> = &'a mut Self::Rtc;

    fn to_variable(x: Self::Rtc) -> Variable<Rt> {
        Variable::PointArray(x)
    }

    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
        match var {
            Variable::PointArray(x) => Some(x),
            _ => {
                eprintln!("expected PointArray, got {:?}", var);
                None
            }
        }
    }

    fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
        match var {
            Variable::PointArray(x) => Some(x),
            _ => {
                eprintln!("expected PointArray, got {:?}", var);
                None
            }
        }
    }
}

#[derive(Debug)]
pub enum PointNode<Rt: RuntimeType> {
    AssertEq(Point<Rt>, Point<Rt>, Option<String>),
    Constant(Rt::PointAffine),
    Common(CommonNode<Rt>),
}

pub type Point<Rt: RuntimeType> = Outer<PointNode<Rt>>;

impl<Rt: RuntimeType> From<(CommonNode<Rt>, SourceInfo)> for Point<Rt> {
    fn from((node, src): (CommonNode<Rt>, SourceInfo)) -> Self {
        Self::new(PointNode::Common(node), src)
    }
}

impl<Rt: RuntimeType> TypeEraseable<Rt> for Point<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        use PointNode::*;
        cg.lookup_or_insert_with(self.as_ptr(), |cg| match &self.inner.t {
            AssertEq(a, b, msg) => {
                let a = a.erase(cg);
                let b = b.erase(cg);
                Vertex::new(
                    VertexNode::AssertEq(a, b, msg.clone()),
                    Some(Typ::Point),
                    self.src_lowered(),
                )
            }
            Constant(c) => {
                let constant =
                    cg.add_constant(Point::to_variable(c.clone()), None, zkpoly_common::typ::Typ::Point);
                Vertex::new(
                    VertexNode::Constant(constant),
                    Some(Typ::Point),
                    self.src_lowered(),
                )
            }
            Common(node) => node.vertex(cg, self.src_lowered()),
        })
    }
}

impl<Rt: RuntimeType> Point<Rt> {
    #[track_caller]
    pub fn assert_eq(&self, b: &Point<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Self::new(PointNode::AssertEq(self.clone(), b.clone(), None), src)
    }

    #[track_caller]
    pub fn assert_eq_with_msg(&self, b: &Point<Rt>, msg: String) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Self::new(PointNode::AssertEq(self.clone(), b.clone(), Some(msg)), src)
    }

    #[track_caller]
    pub fn constant(data: Rt::PointAffine) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Self::new(PointNode::Constant(data), src)
    }
}

impl<Rt: RuntimeType> RuntimeCorrespondance<Rt> for Point<Rt> {
    type Rtc = Rt::PointAffine;
    type RtcBorrowed<'a> = &'a Self::Rtc;
    type RtcBorrowedMut<'a> = &'a mut Self::Rtc;

    fn to_variable(x: Self::Rtc) -> Variable<Rt> {
        Variable::Point(rt::point::Point::new(x))
    }
    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
        match var {
            Variable::Point(x) => Some(x.as_ref()),
            _ => {
                eprintln!("expected Point, got {:?}", var);
                None
            }
        }
    }

    fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
        match var {
            Variable::Point(x) => Some(x.as_mut()),
            _ => {
                eprintln!("expected Point, got {:?}", var);
                None
            }
        }
    }
}

#[track_caller]
pub fn msm_coef<Rt: RuntimeType>(
    polys: impl Iterator<Item = PolyCoef<Rt>>,
    points: &PrecomputedPoints<Rt>,
) -> Array<Rt, Point<Rt>> {
    let src = SourceInfo::new(Location::caller().clone(), None);
    Array::wrap(ArrayUntyped::new(
        ArrayNode::MsmCoef(polys.collect(), points.clone()),
        src,
    ))
}

#[track_caller]
pub fn msm_lagrange<Rt: RuntimeType>(
    polys: impl Iterator<Item = PolyLagrange<Rt>>,
    points: &PrecomputedPoints<Rt>,
) -> Array<Rt, Point<Rt>> {
    let src = SourceInfo::new(Location::caller().clone(), None);
    Array::wrap(ArrayUntyped::new(
        ArrayNode::MsmLagrange(polys.collect(), points.clone()),
        src,
    ))
}
