use std::ops::{Add, Mul, Neg, Sub};

use zkpoly_common::typ::PolyType;

use type2::NttAlgorithm;
use zkpoly_runtime::runtime::transfer::Transfer;

use super::*;

#[derive(Debug)]
pub enum PolyCoefNode<Rt: RuntimeType> {
    Arith(CoefArith<PolyCoef<Rt>, Scalar<Rt>>),
    New(PolyInit, u64),
    Constant(rt::scalar::ScalarArray<Rt::Field>),
    FromLagrange(PolyLagrange<Rt>),
    Extend(PolyCoef<Rt>, u64),
    Assemble(Vec<Scalar<Rt>>, u64),
    Interplote {
        xs: Vec<Scalar<Rt>>,
        ys: Vec<Scalar<Rt>>,
    },
    Slice(PolyCoef<Rt>, u64, u64),
    DistributePowers(PolyCoef<Rt>, PolyLagrange<Rt>),
    Blind(PolyCoef<Rt>, u64, u64),
    KateDivision(PolyCoef<Rt>, Scalar<Rt>),
    AssertEq(PolyCoef<Rt>, PolyCoef<Rt>, Option<String>),
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
                    VertexNode::Sliceable(SliceableNode::SingleArith(arith)),
                    Some(Typ::coef()),
                )
            }
            New(init, deg) => new_vertex(
                VertexNode::Sliceable(SliceableNode::NewPoly(*deg, init.clone(), PolyType::Coef)),
                Some(Typ::coef()),
            ),
            Constant(data) => {
                let constant_id = cg.add_constant(
                    PolyCoef::to_variable(data.clone()),
                    self.src().name.clone(),
                    zkpoly_common::typ::Typ::scalar_array(data.len),
                    data.device.clone(),
                );
                new_vertex(
                    VertexNode::Sliceable(SliceableNode::Constant(constant_id)),
                    Some(Typ::coef_with_deg(data.len() as u64)),
                )
            }
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
                    VertexNode::Interpolate { xs, ys },
                    Some(Typ::coef_with_deg(deg)),
                )
            }
            Slice(operand, start, end) => {
                let operand = operand.erase(cg);
                Vertex::new(
                    VertexNode::Slice(operand, *start, *end),
                    Some(Typ::coef()),
                    self.src_lowered(),
                )
            }
            DistributePowers(poly, powers) => {
                let poly = poly.erase(cg);
                let powers = powers.erase(cg);
                Vertex::new(
                    VertexNode::Sliceable(SliceableNode::DistributePowers { poly, powers }),
                    Some(Typ::coef()),
                    self.src_lowered(),
                )
            }
            Blind(operand, begin, end) => {
                let operand = operand.erase(cg);
                new_vertex(
                    VertexNode::Sliceable(SliceableNode::Blind(operand, *begin, *end)),
                    Some(Typ::coef()),
                )
            }
            KateDivision(lhs, b) => {
                let lhs = lhs.erase(cg);
                let b = b.erase(cg);
                new_vertex(VertexNode::KateDivision(lhs, b), Some(Typ::coef()))
            }
            AssertEq(a, b, msg) => {
                let a = a.erase(cg);
                let b = b.erase(cg);
                new_vertex(VertexNode::AssertEq(a, b, msg.clone()), Some(Typ::coef()))
            }
            Common(cn) => cn.vertex(cg, self.src_lowered()),
        })
    }
}

impl<Rt: RuntimeType> RuntimeCorrespondance<Rt> for PolyCoef<Rt> {
    type Rtc = rt::scalar::ScalarArray<Rt::Field>;
    type RtcBorrowed<'a> = &'a Self::Rtc;
    type RtcBorrowedMut<'a> = &'a mut Self::Rtc;

    fn to_variable(x: Self::Rtc) -> Variable<Rt> {
        Variable::ScalarArray(x)
    }
    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
        match var {
            Variable::ScalarArray(arr) => Some(arr),
            _ => {
                eprintln!("expected ScalarArray, got {:?}", var);
                None
            }
        }
    }

    fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
        match var {
            Variable::ScalarArray(arr) => Some(arr),
            _ => {
                eprintln!("expected ScalarArray, got {:?}", var);
                None
            }
        }
    }
}

impl<Rt: RuntimeType> PolyCoef<Rt> {
    fn pp_op(
        self,
        rhs: PolyCoef<Rt>,
        op: fn(PolyCoef<Rt>, PolyCoef<Rt>) -> CoefArith<PolyCoef<Rt>, Scalar<Rt>>,
        src: SourceInfo,
    ) -> Self {
        PolyCoef::new(PolyCoefNode::Arith(op(self, rhs)), src)
    }

    pub(super) fn ps_op(
        self,
        rhs: Scalar<Rt>,
        op: fn(PolyCoef<Rt>, Scalar<Rt>) -> CoefArith<PolyCoef<Rt>, Scalar<Rt>>,
        src: SourceInfo,
    ) -> Self {
        PolyCoef::new(PolyCoefNode::Arith(op(self, rhs)), src)
    }

    fn unr_op(
        self,
        op: fn(PolyCoef<Rt>) -> CoefArith<PolyCoef<Rt>, Scalar<Rt>>,
        src: SourceInfo,
    ) -> Self {
        PolyCoef::new(PolyCoefNode::Arith(op(self)), src)
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
    pub fn constant(data: &[Rt::Field], allocator: &mut ConstantPool) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        if zkpoly_common::typ::Typ::scalar_array(data.len())
            .can_on_disk::<Rt::Field, Rt::PointAffine>()
            && allocator.has_disk()
        {
            let temp_cpu = rt::scalar::ScalarArray::borrow_vec(data);
            let mut disk_poly =
                rt::scalar::ScalarArray::alloc_disk(data.len(), allocator.unwrap_disk());
            temp_cpu.cpu2disk(&mut disk_poly);
            PolyCoef::new(PolyCoefNode::Constant(disk_poly), src)
        } else {
            PolyCoef::new(
                PolyCoefNode::Constant(rt::scalar::ScalarArray::from_vec(data, &mut allocator.cpu)),
                src,
            )
        }
    }

    #[track_caller]
    pub fn constant_from_iter(
        data: impl Iterator<Item = Rt::Field>,
        len: u64,
        allocator: &mut ConstantPool,
    ) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let temp_cpu = rt::scalar::ScalarArray::from_iter(data, len as usize, &mut allocator.cpu);
        if zkpoly_common::typ::Typ::scalar_array(len as usize)
            .can_on_disk::<Rt::Field, Rt::PointAffine>()
            && allocator.has_disk()
        {
            let mut disk_poly =
                rt::scalar::ScalarArray::alloc_disk(len as usize, allocator.unwrap_disk());
            temp_cpu.cpu2disk(&mut disk_poly);
            allocator.cpu.free(temp_cpu.values);
            PolyCoef::new(PolyCoefNode::Constant(disk_poly), src)
        } else {
            PolyCoef::new(PolyCoefNode::Constant(temp_cpu), src)
        }
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
    pub fn slice(&self, start: u64, end: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::Slice(self.clone(), start, end), src)
    }

    #[track_caller]
    pub fn distribute_powers(&self, power: &PolyLagrange<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(
            PolyCoefNode::DistributePowers(self.clone(), power.clone()),
            src,
        )
    }

    #[track_caller]
    pub fn evaluate(&self, x: &Scalar<Rt>) -> Scalar<Rt> {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(
            scalar::ScalarNode::EvaluatePoly(self.clone(), x.clone()),
            src,
        )
    }

    #[track_caller]
    pub fn index(&self, idx: u64) -> Scalar<Rt> {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Scalar::new(scalar::ScalarNode::IndexCoef(self.clone(), idx), src)
    }

    #[track_caller]
    pub fn blind(&self, begin: u64, end: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::Blind(self.clone(), begin, end), src)
    }

    #[track_caller]
    pub fn random(deg: u64) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let zeros = PolyCoef::new(PolyCoefNode::New(PolyInit::Zeros, deg), src.clone());
        PolyCoef::new(PolyCoefNode::Blind(zeros, 0, deg), src.clone())
    }

    #[track_caller]
    pub fn kate_div(&self, b: &Scalar<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::KateDivision(self.clone(), b.clone()), src)
    }

    #[track_caller]
    pub fn assert_eq(&self, b: &PolyCoef<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(PolyCoefNode::AssertEq(self.clone(), b.clone(), None), src)
    }

    #[track_caller]
    pub fn assert_eq_with_msg(&self, b: &PolyCoef<Rt>, msg: String) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        PolyCoef::new(
            PolyCoefNode::AssertEq(self.clone(), b.clone(), Some(msg)),
            src,
        )
    }
}

impl<Rt: RuntimeType> Add<PolyCoef<Rt>> for PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn add(self, rhs: PolyCoef<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.pp_op(rhs, CoefArith::AddPp, src)
    }
}

impl<Rt: RuntimeType> Sub<PolyCoef<Rt>> for PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn sub(self, rhs: PolyCoef<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.pp_op(rhs, CoefArith::SubPp, src)
    }
}

impl<Rt: RuntimeType> Add<Scalar<Rt>> for PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn add(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ps_op(rhs, CoefArith::AddPs, src)
    }
}

impl<Rt: RuntimeType> Sub<Scalar<Rt>> for PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn sub(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ps_op(rhs, CoefArith::SubPs, src)
    }
}

impl<Rt: RuntimeType> Mul<Scalar<Rt>> for PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn mul(self, rhs: Scalar<Rt>) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.ps_op(rhs, CoefArith::MulPs, src)
    }
}

impl<Rt: RuntimeType> Neg for PolyCoef<Rt> {
    type Output = PolyCoef<Rt>;

    #[track_caller]
    fn neg(self) -> Self::Output {
        let src = SourceInfo::new(Location::caller().clone(), None);
        self.unr_op(CoefArith::Neg, src)
    }
}
