pub use crate::transit::{self, PolyInit};
use std::{fmt::Debug, marker::PhantomData, panic::Location, rc::Rc};
use zkpoly_common::arith;
pub use zkpoly_runtime::args::{Constant, ConstantId};
use zkpoly_runtime::{
    self as rt,
    args::{RuntimeType, Variable},
};

use self::transit::type2::{self, EntryId, VertexId, VertexNode};

pub mod lowering;

use lowering::{Cg, Typ, Vertex};

pub trait TypeEraseable<Rt: RuntimeType>: std::fmt::Debug + 'static {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId;
}

pub trait CommonConstructors<Rt: RuntimeType> {
    fn from_tuple_get(tuple: TupleUntyped<Rt>, idx: usize, src: SourceInfo) -> Self;
    fn from_function_call(
        f: FunctionUntyped<Rt>,
        args: Vec<AstVertex<Rt>>,
        src: SourceInfo,
    ) -> Self;
    fn from_array_get(array: ArrayUntyped<Rt>, idx: usize, src: SourceInfo) -> Self;
    fn from_entry(idx: EntryId, src: SourceInfo) -> Self;
}

#[derive(Debug)]
pub enum CommonNode<Rt: RuntimeType> {
    TupleGet(TupleUntyped<Rt>, usize),
    ArrayGet(ArrayUntyped<Rt>, usize),
    FunctionCall(FunctionUntyped<Rt>, Vec<AstVertex<Rt>>),
    Entry(EntryId)
}

impl<Rt: RuntimeType> CommonNode<Rt> {
    fn from_array_get(array: ArrayUntyped<Rt>, idx: usize) -> Self {
        Self::ArrayGet(array, idx)
    }
    fn from_tuple_get(tuple: TupleUntyped<Rt>, idx: usize) -> Self {
        Self::TupleGet(tuple, idx)
    }
    fn from_function_call(f: FunctionUntyped<Rt>, args: Vec<AstVertex<Rt>>) -> Self {
        Self::FunctionCall(f, args)
    }
    fn from_entry(id: EntryId) -> Self {
        Self::Entry(id)
    }
}

impl<T, Rt: RuntimeType> CommonConstructors<Rt> for T
where
    T: From<(CommonNode<Rt>, SourceInfo)>,
{
    fn from_tuple_get(tuple: TupleUntyped<Rt>, idx: usize, src: SourceInfo) -> Self {
        (CommonNode::from_tuple_get(tuple, idx), src).into()
    }

    fn from_function_call(
        f: FunctionUntyped<Rt>,
        args: Vec<AstVertex<Rt>>,
        src: SourceInfo,
    ) -> Self {
        (CommonNode::from_function_call(f, args), src).into()
    }

    fn from_array_get(array: ArrayUntyped<Rt>, idx: usize, src: SourceInfo) -> Self {
        (CommonNode::from_array_get(array, idx), src).into()
    }

    fn from_entry(id: EntryId, src: SourceInfo) -> Self {
        (CommonNode::from_entry(id), src).into()
    }
}

impl<Rt: RuntimeType> CommonNode<Rt> {
    fn vertex<'s>(&self, cg: &mut Cg<'s, Rt>, src: transit::SourceInfo<'static>) -> Vertex<'s, Rt> {
        match self {
            CommonNode::TupleGet(tuple, idx) => {
                let tuple = tuple.erase(cg);
                Vertex::new(VertexNode::TupleGet(tuple, *idx), None, src)
            }
            CommonNode::ArrayGet(array, idx) => {
                let array = array.erase(cg);
                Vertex::new(VertexNode::ArrayGet(array, *idx), None, src)
            }
            CommonNode::FunctionCall(f, args) => {
                let fid = cg.add_function(f.clone());
                let args = args.iter().map(|x| x.erase(cg)).collect();
                Vertex::new(VertexNode::UserFunction(fid, args), None, src)
            },
            CommonNode::Entry(id) => {
                Vertex::new(VertexNode::Entry(*id), None, src)
            }
        }
    }
}

pub trait RuntimeCorrespondance<Rt: RuntimeType> {
    type Rtc;
    type RtcBorrowed<'a>;

    fn to_variable(x: Self::Rtc) -> Variable<Rt>;
    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>>;
}

#[derive(Debug)]
pub struct AstVertex<Rt: RuntimeType>(Box<dyn TypeEraseable<Rt>>);

impl<Rt: RuntimeType> AstVertex<Rt> {
    fn new(t: impl TypeEraseable<Rt>) -> Self {
        Self(Box::new(t))
    }
}

impl<Rt: RuntimeType> TypeEraseable<Rt> for AstVertex<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        self.0.erase(cg)
    }
}

zkpoly_common::define_usize_id!(ExprId);

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub loc: Location<'static>,
    pub name: Option<String>,
}

impl SourceInfo {
    pub fn new(loc: Location<'static>, name: Option<String>) -> Self {
        Self { loc, name }
    }
}

#[derive(Debug, Clone)]
pub struct Inner<T> {
    t: T,
    src: SourceInfo,
}

pub struct Outer<T> {
    inner: Rc<Inner<T>>,
}

impl<T> Outer<T> {
    pub fn as_ptr(&self) -> *const u8 {
        Rc::as_ptr(&self.inner) as *const _
    }

    pub fn new(t: T, src: SourceInfo) -> Self {
        Self {
            inner: Rc::new(Inner { t, src }),
        }
    }

    pub fn src(&self) -> &SourceInfo {
        &self.inner.src
    }

    pub fn src_lowered(&self) -> transit::SourceInfo<'static> {
        self.inner.src.clone().into()
    }
}

impl<T> Clone for Outer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Debug for Outer<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Outer({:?})", self.inner.t)
    }
}

pub struct Phantomed<T, P> {
    t: T,
    p: PhantomData<P>,
}

impl<T, P> Phantomed<T, P> {
    pub fn wrap(t: T) -> Self {
        Self { t, p: PhantomData }
    }
}

impl<T, P> Clone for Phantomed<T, P>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            t: self.t.clone(),
            p: PhantomData,
        }
    }
}

impl<T, P> Debug for Phantomed<T, P>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Phantomed({:?})", self.t)
    }
}

#[derive(Debug, Clone)]
pub enum LagrangeArith<P, S> {
    Bin(arith::ArithBinOp, P, P),
    Sp(arith::ArithBinOp, S, P),
    Ps(arith::ArithBinOp, P, S),
    Unr(arith::ArithUnrOp, P),
}

impl<P, S> LagrangeArith<P, S> {
    pub fn to_arith<'s, Rt: RuntimeType>(&self, cg: &mut Cg<'s, Rt>) -> arith::Arith<VertexId>
    where
        P: TypeEraseable<Rt>,
        S: TypeEraseable<Rt>,
    {
        use arith::{Arith, BinOp, SpOp, UnrOp};
        use LagrangeArith::*;
        match self {
            Bin(op, lhs, rhs) => {
                let lhs = lhs.erase(cg);
                let rhs = rhs.erase(cg);
                Arith::Bin(BinOp::Pp(op.clone()), lhs, rhs)
            }
            Sp(op, lhs, rhs) => {
                let lhs = lhs.erase(cg);
                let rhs = rhs.erase(cg);
                Arith::Bin(BinOp::Sp(SpOp::for_4ops(op.clone(), false)), lhs, rhs)
            }
            Ps(op, lhs, rhs) => {
                let lhs = lhs.erase(cg);
                let rhs = rhs.erase(cg);
                Arith::Bin(BinOp::Sp(SpOp::for_4ops(op.clone(), true)), rhs, lhs)
            }
            Unr(op, p) => {
                let p = p.erase(cg);
                Arith::Unr(UnrOp::P(op.clone()), p)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum CoefArith<P, S> {
    AddPp(P, P),
    AddPs(P, S),
    SubPp(P, P),
    SubPs(P, S),
    SubSp(S, P),
    Neg(P),
}

impl<P, S> CoefArith<P, S> {
    pub fn to_arith<'s, Rt: RuntimeType>(&self, cg: &mut Cg<'s, Rt>) -> arith::Arith<VertexId>
    where
        P: TypeEraseable<Rt>,
        S: TypeEraseable<Rt>,
    {
        use arith::{Arith, BinOp, SpOp, UnrOp};
        use CoefArith::*;
        match self {
            AddPp(lhs, rhs) => {
                let lhs = lhs.erase(cg);
                let rhs = rhs.erase(cg);
                Arith::Bin(BinOp::Pp(arith::ArithBinOp::Add), lhs, rhs)
            }
            AddPs(lhs, rhs) => {
                let lhs = lhs.erase(cg);
                let rhs = rhs.erase(cg);
                Arith::Bin(BinOp::Sp(SpOp::Add), rhs, lhs)
            }
            SubPp(lhs, rhs) => {
                let lhs = lhs.erase(cg);
                let rhs = rhs.erase(cg);
                Arith::Bin(BinOp::Pp(arith::ArithBinOp::Sub), lhs, rhs)
            }
            SubPs(lhs, rhs) => {
                let lhs = lhs.erase(cg);
                let rhs = rhs.erase(cg);
                Arith::Bin(BinOp::Sp(SpOp::SubBy), rhs, lhs)
            }
            SubSp(lhs, rhs) => {
                let lhs = lhs.erase(cg);
                let rhs = rhs.erase(cg);
                Arith::Bin(BinOp::Sp(SpOp::Sub), lhs, rhs)
            }
            Neg(p) => {
                let p = p.erase(cg);
                Arith::Unr(UnrOp::P(arith::ArithUnrOp::Neg), p)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ScalarArith<S> {
    Bin(arith::ArithBinOp, S, S),
    Unr(arith::ArithUnrOp, S),
}

impl<S> ScalarArith<S> {
    pub fn to_arith<'s, Rt: RuntimeType>(&self, cg: &mut Cg<'s, Rt>) -> arith::Arith<VertexId>
    where
        S: TypeEraseable<Rt>,
    {
        use arith::{Arith, BinOp, SpOp, UnrOp};
        use ScalarArith::*;
        match self {
            Bin(op, lhs, rhs) => {
                let lhs = lhs.erase(cg);
                let rhs = rhs.erase(cg);
                Arith::Bin(BinOp::Ss(op.clone()), lhs, rhs)
            }
            Unr(op, p) => {
                let p = p.erase(cg);
                Arith::Unr(UnrOp::S(op.clone()), p)
            }
        }
    }
}

pub mod array;
pub mod point;
pub mod poly_coef;
pub mod poly_lagrange;
pub mod scalar;
pub mod transcript;
pub mod tuple;
pub mod user_function;
pub mod whatever;

use array::ArrayUntyped;
use tuple::TupleUntyped;
use user_function::FunctionUntyped;
use whatever::WhateverUntyped;

pub use array::{Array, ArrayNode};
pub use point::{Point, PointNode, PrecomputedPoints, PrecomputedPointsData};
pub use poly_coef::{PolyCoef, PolyCoefNode};
pub use poly_lagrange::{PolyLagrange, PolyLagrangeNode};
pub use scalar::Scalar;
pub use transcript::{Transcript, TranscriptNode};
pub use tuple::{
    Tuple10, Tuple2, Tuple3, Tuple4, Tuple5, Tuple6, Tuple7, Tuple8, Tuple9, TupleNode,
};
pub use user_function::Function;
pub use whatever::{Whatever, WhateverNode};
