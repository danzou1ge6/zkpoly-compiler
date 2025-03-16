use std::ops::{Add, Div, Mul, Neg, Sub};

use zkpoly_common::arith::{ArithBinOp, ArithUnrOp};

use super::*;

#[derive(Debug)]
pub enum ScalarNode<Rt: RuntimeType> {
    Arith(ScalarArith<Scalar<Rt>>),
    Constant(Rt::Field),
    One,
    Zero,
    EvaluatePoly(PolyCoef<Rt>, Scalar<Rt>),
    IndexLagrange(PolyLagrange<Rt>, u64),
    IndexCoef(PolyCoef<Rt>, u64),
    AssertEq(Scalar<Rt>, Scalar<Rt>),
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
                    let constant =
                        cg.add_constant(Scalar::to_variable(rt::scalar::Scalar::from_ff(x)), None);
                    new_vertex(VertexNode::Constant(constant), Some(Typ::Scalar))
                }
                One => new_vertex(VertexNode::Constant(cg.one), Some(Typ::Scalar)),
                Zero => new_vertex(VertexNode::Constant(cg.zero), Some(Typ::Scalar)),
                EvaluatePoly(poly, scalar) => {
                    let poly = poly.erase(cg);
                    let at = scalar.erase(cg);
                    new_vertex(VertexNode::EvaluatePoly { poly, at }, Some(Typ::Scalar))
                }
                IndexCoef(poly, idx) => {
                    let poly = poly.erase(cg);
                    new_vertex(VertexNode::IndexPoly(poly, *idx), Some(Typ::Scalar))
                }
                IndexLagrange(poly, idx) => {
                    let poly = poly.erase(cg);
                    new_vertex(VertexNode::IndexPoly(poly, *idx), Some(Typ::Scalar))
                }
                AssertEq(a, b) => {
                    let a = a.erase(cg);
                    let b = b.erase(cg);
                    new_vertex(VertexNode::AssertEq(a, b), Some(Typ::Scalar))
                }
                Common(cn) => cn.vertex(cg, self.src_lowered()),
            }
        })
    }
}

impl<Rt: RuntimeType> RuntimeCorrespondance<Rt> for Scalar<Rt> {
    type Rtc = rt::scalar::Scalar<Rt::Field>;
    type RtcBorrowed<'a> = &'a Self::Rtc;
    type RtcBorrowedMut<'a> = &'a mut Self::Rtc;

    fn to_variable(x: Self::Rtc) -> Variable<Rt> {
        Variable::Scalar(x)
    }
    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
        match var {
            Variable::Scalar(x) => Some(x),
            _ => {
                eprintln!("expected scalar, got {:?}", var);
                None
            }
        }
    }
    fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
        match var {
            Variable::Scalar(x) => Some(x),
            _ => {
                eprintln!("expected scalar, got {:?}", var);
                None
            }
        }
    }
}

impl<Rt: RuntimeType> Scalar<Rt> {
    fn ss_op(self, rhs: Scalar<Rt>, op: ArithBinOp, src: SourceInfo) -> Self {
        Scalar::new(ScalarNode::Arith(ScalarArith::Bin(op, self, rhs)), src)
    }

    fn sp_op(self, rhs: PolyLagrange<Rt>, op: ArithBinOp, src: SourceInfo) -> PolyLagrange<Rt> {
        PolyLagrange::new(
            PolyLagrangeNode::Arith(LagrangeArith::Sp(op, self, rhs)),
            src,
        )
    }

    fn scp_op(
        self,
        rhs: PolyCoef<Rt>,
        op: fn(Scalar<Rt>, PolyCoef<Rt>) -> CoefArith<PolyCoef<Rt>, Scalar<Rt>>,
        src: SourceInfo,
    ) -> PolyCoef<Rt> {
        PolyCoef::new(PolyCoefNode::Arith(op(self, rhs)), src)
    }

    fn unr_op(self, op: ArithUnrOp, src: SourceInfo) -> Self {
        Scalar::new(ScalarNode::Arith(ScalarArith::Unr(op, self)), src)
    }

    #[track_caller]
    pub fn invert(&self) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.clone().unr_op(ArithUnrOp::Inv, src)
    }

    #[track_caller]
    pub fn constant(data: Rt::Field) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(ScalarNode::Constant(data), src)
    }

    #[track_caller]
    pub fn one() -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(ScalarNode::One, src)
    }

    #[track_caller]
    pub fn zero() -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(ScalarNode::Zero, src)
    }

    #[track_caller]
    pub fn pow(&self, power: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.clone().unr_op(ArithUnrOp::Pow(power), src)
    }

    #[track_caller]
    pub fn assert_eq(&self, rhs: &Scalar<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(ScalarNode::AssertEq(self.clone(), rhs.clone()), src)
    }
}

impl<Rt: RuntimeType> Add<Scalar<Rt>> for Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn add(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ss_op(rhs, ArithBinOp::Add, src)
    }
}

impl<Rt: RuntimeType> Sub<Scalar<Rt>> for Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn sub(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ss_op(rhs, ArithBinOp::Sub, src)
    }
}

impl<Rt: RuntimeType> Mul<Scalar<Rt>> for Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn mul(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ss_op(rhs, ArithBinOp::Mul, src)
    }
}

impl<Rt: RuntimeType> Div<Scalar<Rt>> for Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn div(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ss_op(rhs, ArithBinOp::Div, src)
    }
}

impl<Rt: RuntimeType> Neg for Scalar<Rt> {
    type Output = Scalar<Rt>;

    #[track_caller]
    fn neg(self) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.unr_op(ArithUnrOp::Neg, src)
    }
}

impl<Rt: RuntimeType> Add<PolyLagrange<Rt>> for Scalar<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn add(self, rhs: PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.sp_op(rhs, ArithBinOp::Add, src)
    }
}

impl<Rt: RuntimeType> Sub<PolyLagrange<Rt>> for Scalar<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn sub(self, rhs: PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.sp_op(rhs, ArithBinOp::Sub, src)
    }
}

impl<Rt: RuntimeType> Mul<PolyLagrange<Rt>> for Scalar<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn mul(self, rhs: PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.sp_op(rhs, ArithBinOp::Mul, src)
    }
}

impl<Rt: RuntimeType> Div<PolyLagrange<Rt>> for Scalar<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn div(self, rhs: PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.sp_op(rhs, ArithBinOp::Div, src)
    }
}

impl<Rt: RuntimeType> Add<PolyCoef<Rt>> for Scalar<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn add(self, rhs: PolyCoef<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        rhs.ps_op(self, CoefArith::AddPs, src)
    }
}

impl<Rt: RuntimeType> Sub<PolyCoef<Rt>> for Scalar<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn sub(self, rhs: PolyCoef<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.scp_op(rhs, CoefArith::SubSp, src)
    }
}
