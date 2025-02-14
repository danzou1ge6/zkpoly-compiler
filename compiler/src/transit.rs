//! Common data structures for Transit IR's

use std::any;
use std::marker::PhantomData;

use zkpoly_common::{digraph::internal::Digraph, heap::Heap};
use zkpoly_runtime::args::RuntimeType;

#[derive(Debug, Clone)]
pub struct SourceInfo<'s> {
    _marker: PhantomData<&'s str>,
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
    pub(crate) inputs: Vec<I>,
    pub(crate) outputs: Vec<I>,
    pub(crate) g: Digraph<I, V>,
}

impl<I, V> Cg<I, V> {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            g: Digraph::new(),
        }
    }
}

impl<I, V> Default for Cg<I, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum HashTyp {
    HashPoint,
    HashScalar,
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
}

// pub mod type1;
pub mod type2;
pub mod type3;
