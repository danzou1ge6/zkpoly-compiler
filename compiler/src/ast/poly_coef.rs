use std::ops::{Add, Neg, Sub};

use super::*;

#[derive(Debug)]
pub enum PolyCoefNode<Rt: RuntimeType> {
    Arith(CoefArith<PolyCoef<Rt>, Scalar<Rt>>),
    New(PolyInit, u64),
    Constant(Vec<Rt::Field>),
    Entry(u64),
    FromLagrange(PolyLagrange<Rt>),
    Extend(PolyCoef<Rt>, u64),
    Assemble(Vec<Scalar<Rt>>),
    Interplote {
        xs: Vec<Scalar<Rt>>,
        ys: Vec<Scalar<Rt>>,
    },
}

pub type PolyCoef<Rt: RuntimeType> = Outer<PolyCoefNode<Rt>>;

impl<Rt: RuntimeType> PolyCoef<Rt> {
    fn pp_op(
        &self,
        rhs: &PolyCoef<Rt>,
        op: fn(PolyCoef<Rt>, PolyCoef<Rt>) -> CoefArith<PolyCoef<Rt>, Scalar<Rt>>,
        src: SourceInfo,
    ) -> Self {
        PolyCoef::new(PolyCoefNode::Arith(op(self.clone(), rhs.clone())), src)
    }

    pub(super) fn ps_op(
        &self,
        rhs: &Scalar<Rt>,
        op: fn(PolyCoef<Rt>, Scalar<Rt>) -> CoefArith<PolyCoef<Rt>, Scalar<Rt>>,
        src: SourceInfo,
    ) -> Self {
        PolyCoef::new(PolyCoefNode::Arith(op(self.clone(), rhs.clone())), src)
    }

    fn unr_op(
        &self,
        op: fn(PolyCoef<Rt>) -> CoefArith<PolyCoef<Rt>, Scalar<Rt>>,
        src: SourceInfo,
    ) -> Self {
        PolyCoef::new(PolyCoefNode::Arith(op(self.clone())), src)
    }

    #[track_caller]
    pub fn one(deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::New(PolyInit::Ones, deg), src)
    }

    #[track_caller]
    pub fn zero(deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::New(PolyInit::Zeros, deg), src)
    }

    #[track_caller]
    pub fn constant(values: Vec<Rt::Field>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::Constant(values), src)
    }

    #[track_caller]
    pub fn entry(deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::Entry(deg), src)
    }

    #[track_caller]
    pub fn to_lagrange(&self) -> PolyLagrange<Rt> {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyLagrange::new(PolyLagrangeNode::FromCoef(self.clone()), src)
    }

    #[track_caller]
    pub fn extend(&self, deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::Extend(self.clone(), deg), src)
    }

    #[track_caller]
    pub fn assemble(&self, xs: Vec<Scalar<Rt>>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::Assemble(xs), src)
    }

    #[track_caller]
    pub fn interplote(xs: Vec<Scalar<Rt>>, ys: Vec<Scalar<Rt>>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::Interplote { xs, ys }, src)
    }

    #[track_caller]
    pub fn evaluate(&self, x: &Scalar<Rt>) -> Scalar<Rt> {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(scalar::ScalarNode::EvaluatePoly(self.clone(), x.clone()), src)
    }
}

impl<Rt: RuntimeType> Add<&PolyCoef<Rt>> for &PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn add(self, rhs: &PolyCoef<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.pp_op(rhs, CoefArith::AddPp, src)
    }
}

impl<Rt: RuntimeType> Sub<&PolyCoef<Rt>> for &PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn sub(self, rhs: &PolyCoef<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.pp_op(rhs, CoefArith::SubPp, src)
    }
}

impl<Rt: RuntimeType> Add<&Scalar<Rt>> for &PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn add(self, rhs: &Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ps_op(rhs, CoefArith::AddPs, src)
    }
}

impl<Rt: RuntimeType> Sub<&Scalar<Rt>> for &PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn sub(self, rhs: &Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ps_op(rhs, CoefArith::SubPs, src)
    }
}

impl<Rt: RuntimeType> Neg for &PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn neg(self) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.unr_op(CoefArith::Neg, src)
    }
}
