use std::ops::{Add, Div, Mul, Neg, Sub};

use zkpoly_common::arith::{ArithBinOp, ArithUnrOp};

use super::*;

#[derive(Debug)]
pub enum ScalarNode<Rt: RuntimeType> {
    Arith(ScalarArith<Scalar<Rt>>),
    Constant(Rt::Field),
    Entry,
    EvaluatePoly(PolyCoef<Rt>, Scalar<Rt>),
    Common(CommonNode<Rt>),
}

pub type Scalar<Rt: RuntimeType> = Outer<ScalarNode<Rt>>;

impl<Rt: RuntimeType> From<(CommonNode<Rt>, SourceInfo)> for Scalar<Rt> {
    fn from(value: (CommonNode<Rt>, SourceInfo)) -> Self {
        Scalar::new(ScalarNode::Common(value.0), value.1)
    }
}

impl<Rt: RuntimeType> TypeEraseable<Rt> for Scalar<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        cg.lookup_or_insert_with(self.as_ptr(), |cg| {
            use ScalarNode::*;
            let new_vertex = |node, typ| Vertex::new(node, typ, self.src_lowered());

            match &self.inner.t {
                Arith(sa) => {
                    let arith = sa.to_arith(cg);
                    new_vertex(VertexNode::SingleArith(arith), Some(Typ::Scalar))
                }
                Constant(x) => {
                    let constant = unimplemented!();
                    new_vertex(VertexNode::Constant(constant), Some(Typ::Scalar))
                }
                Entry => todo!("track entry ID"),
                EvaluatePoly(poly, scalar) => {
                    let poly = poly.erase(cg);
                    let at = scalar.erase(cg);
                    new_vertex(VertexNode::EvaluatePoly { poly, at }, Some(Typ::Scalar))
                }
                Common(cn) => cn.vertex(cg, self.src_lowered()),
            }
        })
    }
}

impl<Rt: RuntimeType> Scalar<Rt> {
    fn ss_op(&self, rhs: &Scalar<Rt>, op: ArithBinOp, src: SourceInfo) -> Self {
        Scalar::new(
            ScalarNode::Arith(ScalarArith::Bin(op, self.clone(), rhs.clone())),
            src,
        )
    }

    fn sp_op(&self, rhs: &PolyLagrange<Rt>, op: ArithBinOp, src: SourceInfo) -> PolyLagrange<Rt> {
        PolyLagrange::new(
            PolyLagrangeNode::Arith(LagrangeArith::Sp(op, self.clone(), rhs.clone())),
            src,
        )
    }

    fn scp_op(
        &self,
        rhs: &PolyCoef<Rt>,
        op: fn(Scalar<Rt>, PolyCoef<Rt>) -> CoefArith<PolyCoef<Rt>, Scalar<Rt>>,
        src: SourceInfo,
    ) -> PolyCoef<Rt> {
        PolyCoef::new(PolyCoefNode::Arith(op(self.clone(), rhs.clone())), src)
    }

    fn unr_op(&self, op: ArithUnrOp, src: SourceInfo) -> Self {
        Scalar::new(ScalarNode::Arith(ScalarArith::Unr(op, self.clone())), src)
    }

    #[track_caller]
    pub fn invert(&self) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.unr_op(ArithUnrOp::Inv, src)
    }

    #[track_caller]
    pub fn constant(data: Rt::Field) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(ScalarNode::Constant(data), src)
    }

    #[track_caller]
    pub fn entry() -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(ScalarNode::Entry, src)
    }
}

impl<Rt: RuntimeType> Add<&Scalar<Rt>> for &Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn add(self, rhs: &Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ss_op(rhs, ArithBinOp::Add, src)
    }
}

impl<Rt: RuntimeType> Sub<&Scalar<Rt>> for &Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn sub(self, rhs: &Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ss_op(rhs, ArithBinOp::Sub, src)
    }
}

impl<Rt: RuntimeType> Mul<&Scalar<Rt>> for &Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn mul(self, rhs: &Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ss_op(rhs, ArithBinOp::Mul, src)
    }
}

impl<Rt: RuntimeType> Div<&Scalar<Rt>> for &Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn div(self, rhs: &Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ss_op(rhs, ArithBinOp::Div, src)
    }
}

impl<Rt: RuntimeType> Neg for &Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn neg(self) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.unr_op(ArithUnrOp::Neg, src)
    }
}

impl<Rt: RuntimeType> Add<&PolyLagrange<Rt>> for &Scalar<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn add(self, rhs: &PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.sp_op(rhs, ArithBinOp::Add, src)
    }
}

impl<Rt: RuntimeType> Sub<&PolyLagrange<Rt>> for &Scalar<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn sub(self, rhs: &PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.sp_op(rhs, ArithBinOp::Sub, src)
    }
}

impl<Rt: RuntimeType> Mul<&PolyLagrange<Rt>> for &Scalar<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn mul(self, rhs: &PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.sp_op(rhs, ArithBinOp::Mul, src)
    }
}

impl<Rt: RuntimeType> Div<&PolyLagrange<Rt>> for &Scalar<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn div(self, rhs: &PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.sp_op(rhs, ArithBinOp::Div, src)
    }
}

impl<Rt: RuntimeType> Add<&PolyCoef<Rt>> for &Scalar<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn add(self, rhs: &PolyCoef<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        rhs.ps_op(self, CoefArith::AddPs, src)
    }
}

impl<Rt: RuntimeType> Sub<&PolyCoef<Rt>> for &Scalar<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn sub(self, rhs: &PolyCoef<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.scp_op(rhs, CoefArith::SubSp, src)
    }
}
