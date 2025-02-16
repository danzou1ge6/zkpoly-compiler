pub use crate::transit::{
    self,
    type2::{template, Typ},
    PolyInit,
};
use std::{fmt::Debug, marker::PhantomData, panic::Location, rc::Rc};
use zkpoly_common::{arith, digraph::internal::Digraph};
use zkpoly_runtime::args::{RuntimeType, TryBorrowVariable, Variable};
pub use zkpoly_runtime::args::{Constant, ConstantId};

use self::transit::type2::{partial_typed::Vertex, VertexId, VertexNode};

pub mod lowering;

use lowering::Cg;

pub trait TypeEraseable<Rt: RuntimeType>: std::fmt::Debug + 'static {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId;
}

pub trait CommonConstructors<Rt: RuntimeType> {
    fn from_tuple_get(tuple: TupleUntyped<Rt>, idx: usize, src: SourceInfo) -> Self;
    fn from_function_call(f: FunctionUntyped<Rt>, args: Vec<AstVertex<Rt>>, src: SourceInfo) -> Self;
    fn from_array_get(array: ArrayUntyped<Rt>, idx: usize, src: SourceInfo) -> Self;
}

#[derive(Debug)]
pub enum CommonNode<Rt: RuntimeType> {
    TupleGet(TupleUntyped<Rt>, usize),
    ArrayGet(ArrayUntyped<Rt>, usize),
    FunctionCall(FunctionUntyped<Rt>, Vec<AstVertex<Rt>>),
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
}

impl<T, Rt: RuntimeType> CommonConstructors<Rt> for T where T: From<(CommonNode<Rt>, SourceInfo)> {
    fn from_tuple_get(tuple: TupleUntyped<Rt>, idx: usize, src: SourceInfo) -> Self {
        (CommonNode::from_tuple_get(tuple, idx), src).into()
    }

    fn from_function_call(f: FunctionUntyped<Rt>, args: Vec<AstVertex<Rt>>, src: SourceInfo) -> Self {
        (CommonNode::from_function_call(f, args), src).into()
    }

    fn from_array_get(array: ArrayUntyped<Rt>, idx: usize, src: SourceInfo) -> Self {
        (CommonNode::from_array_get(array, idx), src).into()
    }
}

impl<Rt: RuntimeType> TypeEraseable<Rt> for CommonNode<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        unimplemented!()
    }
}

pub trait RuntimeCorrespondance<Rt: RuntimeType> {
    type Rtc: Into<Variable<Rt>> + TryBorrowVariable<Rt>;
}

#[derive(Debug)]
pub struct AstVertex<Rt: RuntimeType>(Box<dyn TypeEraseable<Rt>>);

impl<Rt: RuntimeType> AstVertex<Rt> {
    fn new(t: impl TypeEraseable<Rt>) -> Self {
        Self(Box::new(t))
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

#[derive(Debug, Clone)]
pub enum CoefArith<P, S> {
    AddPp(P, P),
    AddPs(P, S),
    SubPp(P, P),
    SubPs(P, S),
    SubSp(S, P),
    Neg(P),
}

#[derive(Debug, Clone)]
pub enum ScalarArith<S> {
    Bin(arith::ArithBinOp, S, S),
    Unr(arith::ArithUnrOp, S),
}


pub mod array;
pub mod point;
pub mod poly_coef;
pub mod poly_lagrange;
pub mod scalar;
pub mod tuple;
pub mod user_function;
pub mod transcript;
pub mod whatever;

use array::ArrayUntyped;
use tuple::TupleUntyped;
use user_function::FunctionUntyped;
use whatever::WhateverUntyped;

pub use array::{Array, ArrayNode};
pub use point::{PrecomputedPoints, PrecomputedPointsData, Point, PointNode};
pub use poly_coef::{PolyCoef, PolyCoefNode};
pub use poly_lagrange::{PolyLagrange, PolyLagrangeNode};
pub use scalar::Scalar;
pub use tuple::{TupleNode, Tuple2, Tuple3, Tuple4, Tuple5, Tuple6, Tuple7, Tuple8, Tuple9, Tuple10};
pub use user_function::FunctionData;
pub use transcript::{Transcript, TranscriptNode};
pub use whatever::{Whatever, WhateverNode};
