use super::*;

#[derive(Debug, Clone)]
pub struct PrecomputedPointsData<Rt: RuntimeType>(Vec<Rt::PointAffine>);

pub type PrecomputedPoints<Rt: RuntimeType> = Outer<PrecomputedPointsData<Rt>>;

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

pub fn msm_coef<Rt: RuntimeType>(
    polys: &[PolyCoef<Rt>],
    points: &[PrecomputedPoints<Rt>],
) -> Array<Rt, Scalar<Rt>> {
    let src = SourceInfo::new(Location::caller().clone(), None);
    Array::wrap(ArrayUntyped::new(
        ArrayNode::MsmCoef(polys.to_vec(), points.to_vec()),
        src,
    ))
}

pub fn msm_lagrange<Rt: RuntimeType>(
    polys: &[PolyLagrange<Rt>],
    points: &[PrecomputedPoints<Rt>],
) -> Array<Rt, Scalar<Rt>> {
    let src = SourceInfo::new(Location::caller().clone(), None);
    Array::wrap(ArrayUntyped::new(
        ArrayNode::MsmLagrange(polys.to_vec(), points.to_vec()),
        src,
    ))
}
