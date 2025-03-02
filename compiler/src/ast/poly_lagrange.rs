use arith::{ArithBinOp, ArithUnrOp};
use std::ops::{Add, Div, Mul, Neg, Sub};
use zkpoly_common::typ::PolyType;
use zkpoly_memory_pool::PinnedMemoryPool;

use crate::{transit::type2::template::VertexNode, transit::type2::NttAlgorithm};

use super::*;

#[derive(Debug)]
pub enum PolyLagrangeNode<Rt: RuntimeType> {
    Arith(LagrangeArith<PolyLagrange<Rt>, Scalar<Rt>>),
    New(PolyInit, u64),
    Constant(rt::scalar::ScalarArray<Rt::Field>, u64),
    FromCoef(PolyCoef<Rt>),
    RotateIdx(PolyLagrange<Rt>, i32),
    Blind(PolyLagrange<Rt>, u64, u64),
    Slice(PolyLagrange<Rt>, u64, u64),
    DistributePowers(PolyLagrange<Rt>, PolyLagrange<Rt>),
    ScanMul(PolyLagrange<Rt>, Scalar<Rt>),
    Extend(PolyLagrange<Rt>, u64),
    Common(CommonNode<Rt>),
}

pub type PolyLagrange<Rt: RuntimeType> = Outer<PolyLagrangeNode<Rt>>;

impl<'c, Rt: RuntimeType> From<(CommonNode<Rt>, SourceInfo)> for PolyLagrange<Rt> {
    fn from(value: (CommonNode<Rt>, SourceInfo)) -> Self {
        Self::new(PolyLagrangeNode::Common(value.0), value.1)
    }
}

impl<'c, Rt: RuntimeType> TypeEraseable<Rt> for PolyLagrange<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        cg.lookup_or_insert_with(self.as_ptr(), |cg| {
            use PolyLagrangeNode::*;
            match &self.inner.t {
                Arith(la) => {
                    let arith = la.to_arith(cg);
                    Vertex::new(
                        VertexNode::SingleArith(arith),
                        Some(Typ::lagrange()),
                        self.src_lowered(),
                    )
                }
                New(init, deg) => Vertex::new(
                    VertexNode::NewPoly(*deg, init.clone(), PolyType::Lagrange),
                    Some(Typ::Poly((PolyType::Lagrange, Some(*deg)))),
                    self.src_lowered(),
                ),
                Constant(data, len) => {
                    let constant_id = cg.add_constant(
                        PolyLagrange::to_variable(data.clone()),
                        self.src().name.clone(),
                    );
                    Vertex::new(
                        VertexNode::Constant(constant_id),
                        Some(Typ::lagrange_with_deg(*len)),
                        self.src_lowered(),
                    )
                }
                FromCoef(coefs) => {
                    let coefs = coefs.erase(cg);
                    Vertex::new(
                        VertexNode::Ntt {
                            s: coefs,
                            to: PolyType::Lagrange,
                            from: PolyType::Coef,
                            alg: NttAlgorithm::default(),
                        },
                        Some(Typ::lagrange()),
                        self.src_lowered(),
                    )
                }
                RotateIdx(operand, x) => {
                    let operand = operand.erase(cg);
                    Vertex::new(
                        VertexNode::RotateIdx(operand, *x),
                        Some(Typ::lagrange()),
                        self.src_lowered(),
                    )
                }
                Blind(operand, start, end) => {
                    let operand = operand.erase(cg);
                    Vertex::new(
                        VertexNode::Blind(operand, *start, *end),
                        Some(Typ::lagrange()),
                        self.src_lowered(),
                    )
                }
                Slice(operand, start, end) => {
                    let operand = operand.erase(cg);
                    Vertex::new(
                        VertexNode::Slice(operand, *start, *end),
                        Some(Typ::lagrange()),
                        self.src_lowered(),
                    )
                }
                DistributePowers(poly, powers) => {
                    let poly = poly.erase(cg);
                    let powers = powers.erase(cg);
                    Vertex::new(
                        VertexNode::DistributePowers { poly, powers },
                        Some(Typ::lagrange()),
                        self.src_lowered(),
                    )
                }
                ScanMul(poly, x0) => {
                    let poly = poly.erase(cg);
                    let x0 = x0.erase(cg);
                    Vertex::new(
                        VertexNode::ScanMul { poly, x0 },
                        Some(Typ::lagrange()),
                        self.src_lowered(),
                    )
                }
                Extend(poly, deg) => {
                    let poly = poly.erase(cg);
                    Vertex::new(
                        VertexNode::Extend(poly, *deg),
                        Some(Typ::lagrange_with_deg(*deg)),
                        self.src_lowered(),
                    )
                }
                Common(cn) => cn.vertex(cg, self.src_lowered()),
            }
        })
    }
}

impl<'c, Rt: RuntimeType> RuntimeCorrespondance<Rt> for PolyLagrange<Rt> {
    type Rtc = rt::scalar::ScalarArray<Rt::Field>;
    type RtcBorrowed<'a> = &'a Self::Rtc;
    type RtcBorrowedMut<'a> = &'a mut Self::Rtc;

    fn to_variable(x: Self::Rtc) -> Variable<Rt> {
        Variable::ScalarArray(x)
    }
    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
        match var {
            Variable::ScalarArray(arr) => Some(arr),
            _ => None,
        }
    }

    fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
        match var {
            Variable::ScalarArray(arr) => Some(arr),
            _ => None,
        }
    }
}

impl<'c, Rt: RuntimeType> PolyLagrange<Rt> {
    fn pp_op(self, rhs: PolyLagrange<Rt>, op: ArithBinOp, src: SourceInfo) -> Self {
        PolyLagrange::new(
            PolyLagrangeNode::Arith(LagrangeArith::Bin(op, self, rhs)),
            src,
        )
    }

    fn ps_op(self, rhs: Scalar<Rt>, op: ArithBinOp, src: SourceInfo) -> Self {
        PolyLagrange::new(
            PolyLagrangeNode::Arith(LagrangeArith::Ps(op, self, rhs)),
            src,
        )
    }

    fn unr_op(self, op: ArithUnrOp, src: SourceInfo) -> Self {
        PolyLagrange::new(PolyLagrangeNode::Arith(LagrangeArith::Unr(op, self)), src)
    }

    #[track_caller]
    pub fn invert(&self) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::unr_op(self.clone(), ArithUnrOp::Inv, src)
    }

    #[track_caller]
    pub fn ones(deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(PolyLagrangeNode::New(PolyInit::Ones, deg), src)
    }

    #[track_caller]
    pub fn zeros(deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(PolyLagrangeNode::New(PolyInit::Zeros, deg), src)
    }

    #[track_caller]
    pub fn constant(data: &[Rt::Field], allocator: &mut PinnedMemoryPool) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(
            PolyLagrangeNode::Constant(
                rt::scalar::ScalarArray::from_vec(data, allocator),
                data.len() as u64,
            ),
            src,
        )
    }

    #[track_caller]
    pub fn constant_from_iter(
        data: impl Iterator<Item = Rt::Field>,
        len: u64,
        allocator: &mut PinnedMemoryPool,
    ) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(
            PolyLagrangeNode::Constant(
                rt::scalar::ScalarArray::from_iter(data, len as usize, allocator),
                len,
            ),
            src,
        )
    }

    #[track_caller]
    pub fn rotate(&self, rot: i32) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(PolyLagrangeNode::RotateIdx(self.clone(), rot), src)
    }

    #[track_caller]
    pub fn slice(&self, start: u64, end: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(PolyLagrangeNode::Slice(self.clone(), start, end), src)
    }

    #[track_caller]
    pub fn blind(&self, start: u64, end: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(PolyLagrangeNode::Blind(self.clone(), start, end), src)
    }

    #[track_caller]
    pub fn distribute_powers(&self, power: &PolyLagrange<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(
            PolyLagrangeNode::DistributePowers(self.clone(), power.clone()),
            src,
        )
    }

    #[track_caller]
    pub fn scan_mul(&self, x0: &Scalar<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(PolyLagrangeNode::ScanMul(self.clone(), x0.clone()), src)
    }

    #[track_caller]
    pub fn to_coef(&self) -> PolyCoef<Rt> {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::FromLagrange(self.clone()), src)
    }

    #[track_caller]
    pub fn index(&self, idx: u64) -> Scalar<Rt> {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(scalar::ScalarNode::IndexLagrange(self.clone(), idx), src)
    }

    #[track_caller]
    pub fn random(deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let zeros = PolyLagrange::new(PolyLagrangeNode::New(PolyInit::Zeros, deg), src.clone());
        PolyLagrange::new(PolyLagrangeNode::Blind(zeros, 0, deg), src.clone())
    }

    #[track_caller]
    pub fn extend(&self, deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(PolyLagrangeNode::Extend(self.clone(), deg), src.clone())
    }
}

impl<'c, Rt: RuntimeType> Add<PolyLagrange<Rt>> for PolyLagrange<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn add(self, rhs: PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::pp_op(self, rhs, ArithBinOp::Add, src)
    }
}

impl<'c, Rt: RuntimeType> Sub<PolyLagrange<Rt>> for PolyLagrange<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn sub(self, rhs: PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::pp_op(self, rhs, ArithBinOp::Sub, src)
    }
}

impl<'c, Rt: RuntimeType> Mul<PolyLagrange<Rt>> for PolyLagrange<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn mul(self, rhs: PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::pp_op(self, rhs, ArithBinOp::Mul, src)
    }
}

impl<'c, Rt: RuntimeType> Div<PolyLagrange<Rt>> for PolyLagrange<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn div(self, rhs: PolyLagrange<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::pp_op(self, rhs, ArithBinOp::Div, src)
    }
}

impl<'c, Rt: RuntimeType> Neg for PolyLagrange<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn neg(self) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::unr_op(self, ArithUnrOp::Neg, src)
    }
}

impl<'c, Rt: RuntimeType> Add<Scalar<Rt>> for PolyLagrange<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn add(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::ps_op(self, rhs, ArithBinOp::Add, src)
    }
}

impl<'c, Rt: RuntimeType> Sub<Scalar<Rt>> for PolyLagrange<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn sub(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::ps_op(self, rhs, ArithBinOp::Sub, src)
    }
}

impl<'c, Rt: RuntimeType> Mul<Scalar<Rt>> for PolyLagrange<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn mul(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::ps_op(self, rhs, ArithBinOp::Mul, src)
    }
}

impl<'c, Rt: RuntimeType> Div<Scalar<Rt>> for PolyLagrange<Rt> {
    type Output = PolyLagrange<Rt>;

    #[track_caller]
    fn div(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::ps_op(self, rhs, ArithBinOp::Div, src)
    }
}
