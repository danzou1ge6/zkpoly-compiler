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
            let constant =
                cg.add_constant(PrecomputedPoints::to_variable(self.inner.t.0.clone()), None);
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
            _ => None,
        }
    }

    fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
        match var {
            Variable::PointArray(x) => Some(x),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum PointNode<Rt: RuntimeType> {
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
        cg.lookup_or_insert_with(self.as_ptr(), |cg| match &self.inner.t {
            PointNode::Common(node) => node.vertex(cg, self.src_lowered()),
        })
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
            Variable::Point(x) => Some(&x.value),
            _ => None,
        }
    }

    fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
        match var {
            Variable::Point(x) => Some(&mut x.value),
            _ => None,
        }
    }
}

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
