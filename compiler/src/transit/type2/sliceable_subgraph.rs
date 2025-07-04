use zkpoly_runtime::args::RuntimeType;

use super::template::{LastSliceableNode, SliceableNode, VertexNode as OuterVertexNode};
use super::transit::{self, SourceInfo};
use super::{Arith, ConstantId, Typ};

zkpoly_common::define_usize_id!(VertexId);

pub mod template {
    use zkpoly_common::heap::{Heap, UsizeId};

    use super::*;

    #[derive(
        Debug, Clone, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
    )]
    pub enum VertexNode<I, Ie, C> {
        Input(Ie),
        Inner(SliceableNode<I, Arith<I>, C>),
        Last(LastSliceableNode<I>),
        Return(Vec<I>),
    }

    impl<I, Ie, C> VertexNode<I, Ie, C> {
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

        /// Provided a vertex in the big graph and returns a vertex in sliceable subgraph.
        /// If the given vertex is not sliceable nor last-sliceable, it is considered to be
        /// input to the subgraph.
        pub fn relabeled_external_node<E, Sg>(
            external_vid: Ie,
            node: &OuterVertexNode<Ie, Arith<Ie>, C, E, Sg>,
            f: impl FnMut(Ie) -> I,
        ) -> Self
        where
            I: Clone + Default + Ord + std::fmt::Debug,
            C: Clone,
            Ie: Clone
        {
            use OuterVertexNode::*;
            match node {
                Sliceable(sn) => Self::Inner(sn.relabeled(f)),
                LastSliceable(sn) => Self::Last(sn.relabeled(f)),
                _ => Self::Input(external_vid),
            }
        }
    }

    pub struct Cg<I, V> {
        pub(crate) inputs: Vec<I>,
        pub(crate) vertices: Heap<I, V>,
        pub(crate) output: I,
    }

    impl<I, V> Cg<I, V> {
        pub fn new(inputs: Vec<I>, vertices: Heap<I, V>, output: I) -> Self {
            Self {
                inputs,
                vertices,
                output,
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
                .map(|i| self.vertices[*i].node().unwrap_input())
        }

        fn inputs_mut(&mut self) -> impl Iterator<Item = &'_ mut Ie> {
            self.inputs
                .iter_mut()
                .map(|i| -> *mut Ie { self.vertices[*i].node_mut().unwrap_input_mut() as *mut _ })
                .map(|p: *mut _| unsafe { p.as_mut().unwrap() })
        }
    }
}

pub mod alt_label {
    use super::*;

    pub type VertexNode<I> = template::VertexNode<VertexId, I, ConstantId>;
    pub type Vertex<'s, I, T> = transit::Vertex<VertexNode<I>, T, SourceInfo<'s>>;
    pub type Cg<'s, I, Rt> = template::Cg<VertexId, Vertex<'s, I, super::Typ<Rt>>>;
}

pub type VertexNode = alt_label::VertexNode<super::VertexId>;
pub type Vertex<'s, T> = alt_label::Vertex<'s, super::VertexId, T>;
pub type Cg<'s, Rt> = alt_label::Cg<'s, super::VertexId, Rt>;

pub type CgPartialTyped<'s, T> = template::Cg<VertexId, Vertex<'s, T>>;

impl<'s, I, Rt: RuntimeType> alt_label::Cg<'s, I, Rt> {
    pub fn try_relabeled<I2, Er>(
        &self,
        mut f: impl FnMut(I) -> Result<I2, Er>,
    ) -> Result<alt_label::Cg<'s, I2, Rt>, Er>
    where
        I: Clone,
    {
        let vertices = self.vertices.map_by_ref_result(&mut |_, v| {
            use template::VertexNode::*;
            let node = match v.node() {
                Input(ie) => Input(f(ie.clone())?),
                Inner(inner) => Inner(inner.clone()),
                Last(last) => Last(last.clone()),
                Return(r) => Return(r.clone()),
            };

            Ok(alt_label::Vertex::new(
                node,
                v.typ().clone(),
                v.src().clone(),
            ))
        })?;

        Ok(alt_label::Cg {
            inputs: self.inputs.clone(),
            vertices,
            output: self.output.clone(),
        })
    }
}
