pub use crate::transit::{
    self,
    type2::{template, PolyInit, Typ},
};
use std::{panic::Location, rc::Rc};
use zkpoly_runtime::args::RuntimeType;
pub use zkpoly_runtime::args::{Constant, ConstantId};
pub use zkpoly_runtime::functions::{Function, FunctionId as UFunctionId};

zkpoly_common::define_usize_id!(ExprId);

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub loc: Location<'static>,
    pub name: String,
}

impl SourceInfo {
    pub fn new(loc: Location<'static>, name: String) -> Self {
        Self { loc, name }
    }
}

#[derive(Debug, Clone)]
pub enum Arith<Rt: RuntimeType> {
    Bin(transit::arith::ArithBinOp, Vertex<Rt>, Vertex<Rt>),
    Unr(transit::arith::ArithUnrOp, Vertex<Rt>),
}

pub type VertexNode<Rt: RuntimeType> =
    template::VertexNode<Vertex<Rt>, Arith<Rt>, ConstantId, UFunctionId>;
pub use transit::HashTyp;

#[derive(Debug, Clone)]
pub struct VertexInner<Rt: RuntimeType> {
    node: VertexNode<Rt>,
    typ: Option<Typ<Rt>>,
    src: SourceInfo,
}

#[derive(Debug, Clone)]
pub struct Vertex<Rt: RuntimeType>(Rc<VertexInner<Rt>>);

impl<Rt: RuntimeType> From<VertexInner<Rt>> for Vertex<Rt> {
    fn from(inner: VertexInner<Rt>) -> Self {
        Self(Rc::new(inner))
    }
}

impl<Rt: RuntimeType> Vertex<Rt> {
    pub fn inner(&self) -> &VertexInner<Rt> {
        &self.0
    }
}

// pub mod builder;
pub mod typing;
