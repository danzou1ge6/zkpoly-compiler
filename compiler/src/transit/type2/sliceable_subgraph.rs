use zkpoly_common::digraph;

use super::template::{LastSliceableNode, SliceableNode, VertexNode as OuterVertexNode};
use super::transit::{self, SourceInfo};
use super::{Arith, ConstantId, Typ};

zkpoly_common::define_usize_id!(VertexId);

impl std::fmt::Display for VertexId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", usize::from(*self))
    }
}

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

        pub fn unwrap_output(&self) -> &[I] {
            use VertexNode::*;
            match self {
                Return(r) => r,
                _ => panic!("called unwrap_output on non-output vertex"),
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
                Last(lsn) => Last(lsn.try_relabeled(&mut f)?),
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

    #[derive(
        Debug, Clone, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
    )]
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

    impl<I, Ie, C, T, S> Cg<I, transit::Vertex<VertexNode<I, Ie, C>, T, S>> {
        pub fn try_relabeled<Ie2, Er>(
            &self,
            mut f: impl FnMut(Ie) -> Result<Ie2, Er>,
        ) -> Result<Cg<I, transit::Vertex<VertexNode<I, Ie2, C>, T, S>>, Er>
        where
            Ie: Clone,
            I: UsizeId,
            C: Clone,
            T: Clone,
            S: Clone,
        {
            let g = self.g.map_by_ref_result(&mut |_,
                                                    v: &transit::Vertex<
                VertexNode<I, Ie, C>,
                T,
                S,
            >| {
                use VertexNode::*;
                let node = match v.node() {
                    Input(ie) => Input(f(ie.clone())?),
                    Inner(inner) => Inner(inner.clone()),
                    Last(last) => Last(last.clone()),
                    TupleGet(t, i) => TupleGet(t.clone(), *i),
                    Return(r) => Return(r.clone()),
                };

                Ok(transit::Vertex::new(node, v.typ().clone(), v.src().clone()))
            })?;

            Ok(Cg {
                inputs: self.inputs.clone(),
                output: self.output.clone(),
                g,
                loop_inputs: self.loop_inputs.clone(),
                loop_output: self.loop_output.clone(),
            })
        }

        pub fn nth_output(&self, i: usize) -> I
        where
            I: UsizeId,
        {
            self.g.vertex(self.output).node().unwrap_output()[i]
        }
    }

    impl<I, Ie, C, T, S> super::super::template::SubgraphNode<Ie>
        for Cg<I, transit::Vertex<VertexNode<I, Ie, C>, T, S>>
    where
        I: 'static + UsizeId,
        Ie: 'static + Clone,
        C: Clone,
        T: Clone,
        S: Clone,
    {
        type AltLabeled<Ie2> = Cg<I, transit::Vertex<VertexNode<I, Ie2, C>, T, S>>;

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

        fn try_relabeled<I2, Er>(
            &self,
            f: impl FnMut(Ie) -> Result<I2, Er>,
        ) -> Result<Self::AltLabeled<I2>, Er>
        where
            I2: 'static,
        {
            self.try_relabeled(f)
        }
    }

    mod labeling {
        use super::super::super::template::labelling::{EdgeLabel, LabelT};
        use super::VertexNode;
        use std::fmt::{Debug, Display};

        impl<'a, I, Ie, C> Display for LabelT<'a, VertexNode<I, Ie, C>>
        where
            C: Debug,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use VertexNode::*;
                match &self.0 {
                    Input(..) => write!(f, "Input"),
                    Inner(sn) => write!(f, "{}", LabelT(sn)),
                    Last(lsn) => write!(f, "{}", LabelT(lsn)),
                    TupleGet(_, i) => write!(f, "TupleGet({i})"),
                    Return(..) => write!(f, "Return"),
                }
            }
        }

        impl<I, Ie, C> VertexNode<I, Ie, C> {
            pub fn labeled_uses<'a>(
                &'a self,
            ) -> Box<dyn Iterator<Item = (I, Option<EdgeLabel>)> + 'a>
            where
                I: Clone,
            {
                use VertexNode::*;
                match self {
                    Return(xs) => Box::new(
                        xs.iter()
                            .enumerate()
                            .map(|(i, x)| (x.clone(), Some(EdgeLabel::Enumerate("x", i)))),
                    ),
                    _ => Box::new(self.uses().map(|x| (x, None))),
                }
            }
        }
    }

    mod pretty {
        use super::super::super::template::labelling::LabelT;
        use super::*;
        use crate::transit;
        use cytoscape_visualizer::{self as vis, pretty::*};
        use zkpoly_common::arith;

        use std::fmt::{Debug, Display};

        type Vertex<Ii, I, C, T, S> = transit::Vertex<VertexNode<Ii, I, C>, T, S>;

        impl<I, Ii, T, C, S> IdentifyVertex<Ii> for Vertex<Ii, I, C, T, S> {}

        impl<I, Ii, T, C, S> PrettyVertex<Ii, I, ()> for transit::Vertex<VertexNode<Ii, I, C>, T, S>
        where
            Ii: Display + Clone,
            I: Display + Clone,
            T: Debug,
            S: Debug,
            C: Debug,
        {
            fn pretty_vertex<'a>(
                &self,
                vid: Ii,
                ctx: Context<'a>,
                _aux: &(),
                builder: &mut vis::Builder,
            ) {
                let id = Self::identifier(vid, &ctx);
                let label = LabelT(self.node()).to_string();

                if let VertexNode::Inner(SliceableNode::Arith { arith, .. }) = self.node() {
                    let mut cluster_builder = vis::ClusterBuilder::new(vis::Vertex::new(label));
                    pretty_vertices(arith, &(), ctx.push(&id), &mut cluster_builder);
                    builder.cluster(id, cluster_builder);
                } else {
                    builder.vertex(
                        id,
                        vis::Vertex::new(label).with_info(format!(
                            "{:?}\\n@{:?}",
                            self.typ(),
                            self.src()
                        )),
                    );
                }
            }

            fn labeled_predecessors(
                &self,
            ) -> impl Iterator<Item = (PredecessorId<Ii, I>, vis::EdgeStyle)> {
                let r: Box<dyn Iterator<Item = _>> =
                    if let VertexNode::Input(outer_id) = self.node() {
                        Box::new(std::iter::once((
                            PredecessorId::External(outer_id.clone()),
                            vis::EdgeStyle::default(),
                        )))
                    } else {
                        Box::new(self.node().labeled_uses().map(|(u, el)| {
                            (
                                PredecessorId::Internal(u),
                                vis::EdgeStyle::default()
                                    .with_optional_label(el.map(|el| el.to_string())),
                            )
                        }))
                    };
                r
            }

            fn pretty_vertex_internal_edges<'a, G>(
                &self,
                vid: Ii,
                ctx: Context<'a>,
                _aux: &(),
                g: &G,
                builder: &mut vis::Builder,
            ) where
                G: PrettyGraph<Self, Ii>,
                Self: Sized,
            {
                let id = Self::identifier(vid, &ctx);
                if let VertexNode::Inner(SliceableNode::Arith { arith, .. }) = self.node() {
                    pretty_edges(arith, &(), ctx.push(&id), g, builder)
                }
            }
        }

        impl<I, Ii, T, C, S> PrettyGraph<Vertex<Ii, I, C, T, S>, Ii> for Cg<Ii, Vertex<Ii, I, C, T, S>>
        where
            Ii: UsizeId + Display,
            I: Display + Clone,
            T: Debug,
            C: Debug,
            S: Debug,
        {
            fn vertices(&self) -> impl Iterator<Item = Ii> {
                use VertexNode::*;
                self.g
                    .vertices()
                    .filter(|i| match self.g.vertex(*i).node() {
                        TupleGet(..) => false,
                        _ => true,
                    })
            }

            fn vertex(&self, vid: Ii) -> &Vertex<Ii, I, C, T, S> {
                self.g.vertex(vid)
            }

            fn indirect_from<'a>(&self, vid: Ii, ctx: &Context<'a>) -> vis::Id {
                match self.g.vertex(vid).node() {
                    VertexNode::TupleGet(from, i) => match self.g.vertex(*from).node() {
                        VertexNode::Inner(SliceableNode::Arith { arith, .. }) => {
                            let id = Vertex::<Ii, I, C, T, S>::identifier(*from, ctx);
                            arith::ArithVertex::<Ii, _>::identifier(
                                arith.outputs[*i],
                                &ctx.push(&id),
                            )
                        }
                        _ => panic!("tuple get only expected to get from Arith"),
                    },
                    _ => Vertex::<Ii, I, C, T, S>::identifier(vid, ctx),
                }
            }
        }
    }
}

pub mod alt_label {
    use super::*;

    pub type VertexNode<I, Ie> = template::VertexNode<I, Ie, ConstantId>;
    pub type Vertex<'s, I, Ie, T> = transit::Vertex<VertexNode<I, Ie>, T, SourceInfo<'s>>;
    pub type Cg<'s, I, Ie, Rt> = template::Cg<I, Vertex<'s, I, Ie, super::Typ<Rt>>>;
    pub type CgPartilTyped<'s, I, Ie, T> = template::Cg<I, Vertex<'s, I, Ie, T>>;
}

pub type VertexNode = alt_label::VertexNode<VertexId, super::VertexId>;
pub type Vertex<'s, Rt> = alt_label::Vertex<'s, VertexId, super::VertexId, super::Typ<Rt>>;
pub type Cg<'s, Rt> = alt_label::Cg<'s, VertexId, super::VertexId, Rt>;

pub type CgPartialTyped<'s, T> =
    template::Cg<VertexId, alt_label::Vertex<'s, VertexId, super::VertexId, T>>;

impl<I, Ie, T, C, S> digraph::internal::Predecessors<I>
    for transit::Vertex<template::VertexNode<I, Ie, C>, T, S>
where
    I: Clone,
{
    fn predecessors(&self) -> impl Iterator<Item = I> {
        self.node().uses()
    }
}
