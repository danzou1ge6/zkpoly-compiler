pub use crate::transit::{
    self,
    type2::{template, PolyInit, Typ},
};
use std::{panic::Location, rc::Rc};
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
pub enum Arith {
    Bin(transit::ArithBinOp, Vertex, Vertex),
    Unr(transit::ArithUnrOp, Vertex),
}

pub type VertexNode = template::VertexNode<Vertex, Arith, ConstantId, UFunctionId>;
pub use transit::HashTyp;

#[derive(Debug, Clone)]
pub struct VertexInner {
    node: VertexNode,
    typ: Option<Typ>,
    src: SourceInfo,
}

#[derive(Debug, Clone)]
pub struct Vertex(Rc<VertexInner>);

impl From<VertexInner> for Vertex {
    fn from(inner: VertexInner) -> Self {
        Self(Rc::new(inner))
    }
}

impl Vertex {
    pub fn inner(&self) -> &VertexInner {
        &self.0
    }
}

pub mod builder;
