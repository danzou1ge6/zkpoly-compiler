use std::ops::{Add, Neg, Sub};

use zkpoly_common::typ::PolyType;

use type2::NttAlgorithm;

use super::*;

#[derive(Debug)]
pub enum PolyCoefNode<Rt: RuntimeType> {
    Arith(CoefArith<PolyCoef<Rt>, Scalar<Rt>>),
    New(PolyInit, u64),
    Constant(Vec<Rt::Field>),
    Entry(u64),
    FromLagrange(PolyLagrange<Rt>),
    Extend(PolyCoef<Rt>, u64),
    Assemble(Vec<Scalar<Rt>>, u64),
    Interplote {
        xs: Vec<Scalar<Rt>>,
        ys: Vec<Scalar<Rt>>,
    },
    Common(CommonNode<Rt>),
}

pub type PolyCoef<Rt: RuntimeType> = Outer<PolyCoefNode<Rt>>;

impl<Rt: RuntimeType> From<(CommonNode<Rt>, SourceInfo)> for PolyCoef<Rt> {
    fn from(value: (CommonNode<Rt>, SourceInfo)) -> Self {
        Self::new(PolyCoefNode::Common(value.0), value.1)
    }
}

impl<Rt: RuntimeType> TypeEraseable<Rt> for PolyCoef<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        use PolyCoefNode::*;
        let new_vertex = |node, typ| Vertex::new(node, typ, self.src_lowered());

        cg.lookup_or_insert_with(self.as_ptr(), |cg| match &self.inner.t {
            Arith(arith) => {
                let arith = arith.to_arith(cg);
                new_vertex(
                    VertexNode::SingleArith(arith),
                    Some(Typ::coef()),
                )
            }
            New(init, deg) => new_vertex(
                VertexNode::NewPoly(*deg, init.clone(), PolyType::Coef),
                Some(Typ::coef()),
            ),
            Constant(data) => {
                let value = unimplemented!();
                let constant_id = cg.add_constant(value, self.src().name.clone());
                new_vertex(
                    VertexNode::Constant(constant_id),
                    Some(Typ::coef_with_deg(data.len() as u64)),
                )
            }
            Entry(deg) => todo!("track entry ID"),
            FromLagrange(values) => {
                let values = values.erase(cg);
                new_vertex(
                    VertexNode::Ntt {
                        s: values,
                        to: PolyType::Coef,
                        from: PolyType::Lagrange,
                        alg: NttAlgorithm::default(),
                    },
                    Some(Typ::coef()),
                )
            }
            Extend(poly, deg) => {
                let poly = poly.erase(cg);
                new_vertex(
                    VertexNode::Extend(poly, *deg),
                    Some(Typ::coef_with_deg(*deg)),
                )
            }
            Assemble(values, deg) => {
                let values = values.iter().cloned().map(|x| x.erase(cg)).collect();
                new_vertex(
                    VertexNode::AssmblePoly(*deg, values),
                    Some(Typ::coef_with_deg(*deg)),
                )
            }
            Interplote { xs, ys } => {
                let deg = xs.len() as u64;
                let xs = xs.iter().cloned().map(|x| x.erase(cg)).collect();
                let ys = ys.iter().cloned().map(|y| y.erase(cg)).collect();
                new_vertex(
                    VertexNode::Interplote { xs, ys },
                    Some(Typ::coef_with_deg(deg)),
                )
            }
            Common(cn) => cn.vertex(cg, self.src_lowered()),
        })
    }
}

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
    pub fn assemble(&self, xs: Vec<Scalar<Rt>>, deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::Assemble(xs, deg), src)
    }

    #[track_caller]
    pub fn interplote(xs: Vec<Scalar<Rt>>, ys: Vec<Scalar<Rt>>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::Interplote { xs, ys }, src)
    }

    #[track_caller]
    pub fn evaluate(&self, x: &Scalar<Rt>) -> Scalar<Rt> {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(
            scalar::ScalarNode::EvaluatePoly(self.clone(), x.clone()),
            src,
        )
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
