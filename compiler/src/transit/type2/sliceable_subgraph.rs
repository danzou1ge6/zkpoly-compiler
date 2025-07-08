use zkpoly_common::digraph;
use zkpoly_runtime::args::RuntimeType;

use super::template::{LastSliceableNode, SliceableNode, VertexNode as OuterVertexNode};
use super::transit::{self, SourceInfo};
use super::{Arith, ConstantId, Typ};

zkpoly_common::define_usize_id!(VertexId);

pub mod template {
    use zkpoly_common::{digraph::internal::Digraph, heap::UsizeId};

    use super::transit::{self, type2};
    use super::{Arith, LastSliceableNode, OuterVertexNode, SliceableNode};

    #[derive(
        Debug, Clone, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
    )]
    pub enum VertexNode<I, Ie, C> {
        Input(Ie),
        Inner(SliceableNode<I, Arith<I>, C>),
        Last(LastSliceableNode<I>),
        TupleGet(I, usize),
        Return(Vec<I>),
    }

    impl<I, Ie, C> VertexNode<I, Ie, C>
    where
        I: Clone,
    {
        pub fn unwrap_input(&self) -> &Ie {
            match self {
                VertexNode::Input(x) => x,
                _ => panic!("called unwrap_input on non-input vertex"),
            }
        }

        pub fn unwrap_input_mut(&mut self) -> &mut Ie {
            match self {
                VertexNode::Input(x) => x,
                _ => panic!("called unwrap_input_mut on non-input vertex"),
            }
        }

        pub fn uses_ref<'a>(&'a self) -> Box<dyn Iterator<Item = &'a I> + 'a> {
            use VertexNode::*;
            match self {
                Input(..) => Box::new(std::iter::empty()),
                Inner(sn) => sn.uses_ref(),
                Last(lsn) => lsn.uses_ref(),
                TupleGet(t, _) => Box::new(std::iter::once(t)),
                Return(r) => Box::new(r.iter()),
            }
        }

        pub fn uses<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
            self.uses_ref().cloned()
        }

        pub fn device(&self) -> type2::DevicePreference {
            use VertexNode::*;
            match self {
                Input(..) => type2::DevicePreference::Cpu,
                Inner(sn) => sn.device(),
                Last(lsn) => lsn.device(),
                TupleGet(..) => type2::DevicePreference::Cpu,
                Return(..) => type2::DevicePreference::Cpu,
            }
        }

        pub fn mutable_uses<'a>(&'a self) -> Box<dyn Iterator<Item = I> + 'a> {
            use VertexNode::*;
            match self {
                Input(..) => Box::new(std::iter::empty()),
                Inner(sn) => sn.mutable_uses(),
                Last(lsn) => lsn.mutable_uses(),
                TupleGet(..) => Box::new(std::iter::empty()),
                Return(..) => Box::new(std::iter::empty()),
            }
        }

        pub fn mutable_uses_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut I> + 'a> {
            use VertexNode::*;
            match self {
                Input(..) => Box::new(std::iter::empty()),
                Inner(sn) => sn.mutable_uses_mut(),
                Last(lsn) => lsn.mutable_uses_mut(),
                TupleGet(..) => Box::new(std::iter::empty()),
                Return(..) => Box::new(std::iter::empty()),
            }
        }

        pub fn outputs_inplace<'a>(&'a self) -> Box<dyn Iterator<Item = Option<I>> + 'a> {
            use VertexNode::*;
            match self {
                Input(..) => Box::new(std::iter::repeat(None)),
                Inner(sn) => sn.outputs_inplace(),
                Last(lsn) => lsn.outputs_inplace(),
                TupleGet(..) => Box::new(std::iter::repeat(None)),
                Return(..) => Box::new(std::iter::repeat(None)),
            }
        }

        /// Provided a vertex in the big graph and returns a vertex in sliceable subgraph.
        /// If the given vertex is not sliceable nor last-sliceable, it is considered to be
        /// input to the subgraph.
        pub fn relabeled_external_node<E, Sg>(
            external_vid: Ie,
            node: &OuterVertexNode<Ie, Arith<Ie>, C, E, Sg>,
            mut f: impl FnMut(Ie) -> I,
        ) -> Self
        where
            I: Clone + Default + Ord + std::fmt::Debug,
            C: Clone,
            Ie: Clone,
        {
            use OuterVertexNode::*;
            match node {
                Sliceable(sn) => Self::Inner(sn.relabeled(f)),
                LastSliceable(sn) => Self::Last(sn.relabeled(f)),
                TupleGet(t, i) => Self::TupleGet(f(t.clone()), *i),
                _ => Self::Input(external_vid),
            }
        }

        pub fn try_relabeled<Ie2, I2, Er>(
            &self,
            mut f: impl FnMut(I) -> Result<I2, Er>,
            mut fe: impl FnMut(Ie) -> Result<Ie2, Er>,
        ) -> Result<VertexNode<I2, Ie2, C>, Er>
        where
            I: Clone,
            Ie: Clone,
            C: Clone,
            I2: Default + Ord + std::fmt::Debug + Clone,
        {
            use VertexNode::*;
            Ok(match self {
                Input(ie) => Input(fe(ie.clone())?),
                Inner(sn) => Inner(sn.try_relabeled(&mut f)?),
                Last(lsn) => Last(lsn.try_relabeld(&mut f)?),
                TupleGet(t, i) => TupleGet(f(t.clone())?, *i),
                Return(r) => Return(r.iter().cloned().map(f).collect::<Result<Vec<_>, _>>()?),
            })
        }

        pub fn relabeled<Ie2, I2>(
            &self,
            mut f: impl FnMut(I) -> I2,
            mut fe: impl FnMut(Ie) -> Ie2,
        ) -> VertexNode<I2, Ie2, C>
        where
            I: Clone,
            Ie: Clone,
            C: Clone,
            I2: Default + Ord + std::fmt::Debug + Clone,
            Ie2: Default + Ord + std::fmt::Debug + Clone,
        {
            self.try_relabeled(|x| Ok::<_, ()>(f(x)), |x| Ok(fe(x)))
                .unwrap()
        }
    }

    pub struct Cg<I, V> {
        pub(crate) inputs: Vec<I>,
        pub(crate) loop_inputs: Vec<I>,
        pub(crate) g: Digraph<I, V>,
        pub(crate) output: I,
        pub(crate) loop_output: Vec<I>,
    }

    impl<I, V> Cg<I, V> {
        pub fn new(inputs: Vec<I>, g: Digraph<I, V>, output: I) -> Self {
            Self {
                inputs,
                g,
                output,
                loop_inputs: Vec::new(),
                loop_output: Vec::new(),
            }
        }
    }

    impl<I, Ie, C, T, S> super::super::template::SubgraphNode<Ie>
        for Cg<I, transit::Vertex<VertexNode<I, Ie, C>, T, S>>
    where
        I: 'static + UsizeId,
        Ie: 'static,
    {
        fn inputs(&self) -> impl Iterator<Item = &'_ Ie> {
            self.inputs
                .iter()
                .map(|i| self.g.vertex(*i).node().unwrap_input())
        }

        fn inputs_mut(&mut self) -> impl Iterator<Item = &'_ mut Ie> {
            self.inputs
                .iter_mut()
                .map(|i| -> *mut Ie {
                    self.g.vertex_mut(*i).node_mut().unwrap_input_mut() as *mut _
                })
                .map(|p: *mut _| unsafe { p.as_mut().unwrap() })
        }
    }
}

pub mod alt_label {
    use super::*;

    pub type VertexNode<I, Ie> = template::VertexNode<I, Ie, ConstantId>;
    pub type Vertex<'s, I, Ie, T> = transit::Vertex<VertexNode<I, Ie>, T, SourceInfo<'s>>;
    pub type Cg<'s, I, Ie, Rt> = template::Cg<VertexId, Vertex<'s, I, Ie, super::Typ<Rt>>>;
}

pub type VertexNode = alt_label::VertexNode<VertexId, super::VertexId>;
pub type Vertex<'s, Rt> = alt_label::Vertex<'s, VertexId, super::VertexId, super::Typ<Rt>>;
pub type Cg<'s, Rt> = alt_label::Cg<'s, VertexId, super::VertexId, Rt>;

pub type CgPartialTyped<'s, T> =
    template::Cg<VertexId, Vertex<'s, alt_label::Vertex<'s, VertexId, super::VertexId, T>>>;

impl<'s, I, Ie, Rt: RuntimeType> alt_label::Cg<'s, I, Ie, Rt> {
    pub fn try_relabeled<Ie2, Er>(
        &self,
        mut f: impl FnMut(Ie) -> Result<Ie2, Er>,
    ) -> Result<alt_label::Cg<'s, I, Ie2, Rt>, Er>
    where
        Ie: Clone,
        I: Clone,
    {
        let g =
            self.g.map_by_ref_result(
                &mut |_, v: &alt_label::Vertex<'s, I, Ie, super::Typ<Rt>>| {
                    use template::VertexNode::*;
                    let node = match v.node() {
                        Input(ie) => Input(f(ie.clone())?),
                        Inner(inner) => Inner(inner.clone()),
                        Last(last) => Last(last.clone()),
                        TupleGet(t, i) => TupleGet(t.clone(), *i),
                        Return(r) => Return(r.clone()),
                    };

                    Ok(alt_label::Vertex::new(
                        node,
                        v.typ().clone(),
                        v.src().clone(),
                    ))
                },
            )?;

        Ok(alt_label::Cg {
            inputs: self.inputs.clone(),
            output: self.output.clone(),
            g,
            loop_inputs: self.loop_inputs.clone(),
            loop_output: self.loop_output.clone(),
        })
    }
}

impl<I, Ie, T, C, S> digraph::internal::Predecessors<I>
    for transit::Vertex<template::VertexNode<I, Ie, C>, T, S>
where
    I: Clone,
{
    fn predecessors(&self) -> impl Iterator<Item = I> {
        self.node().uses()
    }
}
