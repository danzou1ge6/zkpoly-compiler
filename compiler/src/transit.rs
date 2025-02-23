//! Common data structures for Transit IR's

use std::marker::PhantomData;
use std::{any, panic::Location};

use zkpoly_common::{digraph::internal::Digraph, heap::Heap};
use zkpoly_runtime::args::RuntimeType;

use crate::ast;

#[derive(Debug, Clone)]
pub struct SourceInfo<'s> {
    location: Locations<'s>,
    name: Option<String>,
}

impl<'s> SourceInfo<'s> {
    pub fn unwrap_location_single(&self) -> &Location<'s> {
        match &self.location {
            Locations::Single(loc) => loc,
            Locations::Multi(_) => panic!("unwrap_location_single: expected single location"),
        }
    }

    pub fn new(location: Locations<'s>, name: Option<String>) -> Self {
        Self { location, name }
    }
}

#[derive(Debug, Clone)]
pub enum Locations<'s> {
    Single(Location<'s>),
    Multi(Vec<Location<'s>>),
}

impl From<ast::SourceInfo> for SourceInfo<'_> {
    fn from(value: ast::SourceInfo) -> Self {
        Self {
            location: Locations::Single(value.loc),
            name: value.name,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PolyInit {
    Zeros,
    Ones,
}

/// Computation Graph of a Transit IR function.
/// [`V`]: vertex
/// [`I`]: vertex ID
#[derive(Debug, Clone)]
pub struct Cg<I, V> {
    pub(crate) output: I,
    pub(crate) g: Digraph<I, V>,
}

#[derive(Debug, Clone, Copy)]
pub enum HashTyp {
    WriteProof,
    NoWriteProof,
}

#[derive(Debug, Clone)]
pub struct Vertex<N, T, S>(N, T, S);

impl<N, T, S> Vertex<N, T, S> {
    pub fn new(v_node: N, v_typ: T, v_src: S) -> Self {
        Self(v_node, v_typ, v_src)
    }
    pub fn node(&self) -> &N {
        &self.0
    }
    pub fn typ(&self) -> &T {
        &self.1
    }
    pub fn src(&self) -> &S {
        &self.2
    }
    pub fn node_mut(&mut self) -> &mut N {
        &mut self.0
    }
    pub fn typ_mut(&mut self) -> &mut T {
        &mut self.1
    }
    pub fn src_mut(&mut self) -> &mut S {
        &mut self.2
    }

    pub fn map_typ<T2>(self, f: impl FnOnce(T) -> T2) -> Vertex<N, T2, S> {
        Vertex(self.0, f(self.1), self.2)
    }
}

// pub mod type1;
pub mod type2;
pub mod type3;
