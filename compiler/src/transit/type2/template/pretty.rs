use super::super::Arith;
use super::labelling::LabelT;
use super::*;
use crate::transit;
use cytoscape_visualizer::{self as vis, pretty::*};
use zkpoly_common::{
    digraph::internal::{Digraph, SubDigraph},
    heap::UsizeId,
};

use std::fmt::{Debug, Display};

type Vertex<I, C, E, S, T, Sr> = transit::Vertex<VertexNode<I, Arith<I>, C, E, S>, T, Sr>;

use super::super::sliceable_subgraph::template as ss_template;

type SsCg<Ii, I, C, T, Sr> =
    ss_template::Cg<Ii, transit::Vertex<ss_template::VertexNode<Ii, I, C>, T, Sr>>;

impl<I, C, E, S, T, Sr> IdentifyVertex<I> for Vertex<I, C, E, S, T, Sr> {}

pub struct StyledVertex<F>(pub F);

impl<I, C, E, T, Sr, F> PrettyVertex<I, (), StyledVertex<F>> for Vertex<I, C, E, (), T, Sr>
where
    I: Display + Copy + 'static,
    Sr: Debug,
    T: Debug,
    C: Debug,
    E: Debug,
    F: Fn(I, &Vertex<I, C, E, (), T, Sr>) -> vis::Vertex,
{
    fn pretty_vertex<'a>(
        &self,
        vid: I,
        ctx: Context<'a>,
        aux: &StyledVertex<F>,
        builder: &mut vis::Builder,
    ) {
        let label = LabelT(self.node()).to_string();
        let id = Self::identifier(vid, &ctx);

        match self.node() {
            VertexNode::Sliceable(SliceableNode::Arith { arith, .. }) => {
                let mut cluster_builder = vis::ClusterBuilder::new(vis::Vertex::new(label));
                pretty_vertices(arith, &(), ctx.push(&id), &mut cluster_builder);
                builder.cluster(id, cluster_builder);
            }
            _ => {
                builder.vertex(id, aux.0(vid, self));
            }
        }
    }

    fn labeled_predecessors(
        &self,
        _vid: I,
        _aux: &StyledVertex<F>,
    ) -> impl Iterator<Item = (PredecessorId<I, ()>, vis::EdgeStyle)> {
        self.node().labeled_uses().map(|(u, l)| {
            (
                PredecessorId::Internal(u),
                vis::EdgeStyle::default().with_optional_label(l.map(|l| l.to_string())),
            )
        })
    }

    fn pretty_vertex_internal_edges<'a, G>(
        &self,
        vid: I,
        ctx: Context<'a>,
        _aux: &StyledVertex<F>,
        g: &G,
        builder: &mut vis::Builder,
    ) where
        G: PrettyGraph<Self, I>,
        Self: Sized,
    {
        let id = Self::identifier(vid, &ctx);
        match self.node() {
            VertexNode::Sliceable(SliceableNode::Arith { arith, .. }) => {
                pretty_edges(arith, &(), ctx.push(&id), g, builder)
            }
            _ => {}
        }
    }
}

impl<I, C, E, T, Sr, Ii, F> PrettyVertex<I, (), StyledVertex<F>>
    for Vertex<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr>
where
    I: Display + Copy + 'static,
    Sr: Debug + Clone,
    T: Debug + Clone,
    C: Debug + Clone,
    E: Debug,
    Ii: Copy + Display + UsizeId + 'static,
    F: Fn(I, &Vertex<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr>) -> vis::Vertex,
{
    fn pretty_vertex<'a>(
        &self,
        vid: I,
        ctx: Context<'a>,
        aux: &StyledVertex<F>,
        builder: &mut vis::Builder,
    ) {
        let label = LabelT(self.node()).to_string();
        let id = Self::identifier(vid, &ctx);

        match self.node() {
            VertexNode::Sliceable(SliceableNode::Arith { arith, .. }) => {
                let mut cluster_builder = vis::ClusterBuilder::new(vis::Vertex::new(label));
                pretty_vertices(arith, &(), ctx.push(&id), &mut cluster_builder);
                builder.cluster(id, cluster_builder);
            }
            VertexNode::Subgraph(s) => {
                let mut cluster_builder = vis::ClusterBuilder::new(vis::Vertex::new(label));
                pretty_vertices(s, &(), ctx.push(&id), &mut cluster_builder);
                builder.cluster(id, cluster_builder);
            }
            _ => {
                builder.vertex(id, aux.0(vid, self));
            }
        }
    }

    fn labeled_predecessors(
        &self,
        _vid: I,
        _aux: &StyledVertex<F>,
    ) -> impl Iterator<Item = (PredecessorId<I, ()>, vis::EdgeStyle)> {
        self.node().labeled_uses().map(|(u, l)| {
            (
                PredecessorId::Internal(u),
                vis::EdgeStyle::default().with_optional_label(l.map(|l| l.to_string())),
            )
        })
    }

    fn pretty_vertex_internal_edges<'a, G>(
        &self,
        vid: I,
        ctx: Context<'a>,
        _aux: &StyledVertex<F>,
        g: &G,
        builder: &mut vis::Builder,
    ) where
        G: PrettyGraph<Self, I>,
        Self: Sized,
    {
        let id = Self::identifier(vid, &ctx);
        match self.node() {
            VertexNode::Sliceable(SliceableNode::Arith { arith, .. }) => {
                pretty_edges(arith, &(), ctx.push(&id), g, builder)
            }
            VertexNode::Subgraph(s) => pretty_edges(s, &(), ctx.push(&id), g, builder),
            _ => {}
        }
    }
}

impl<'g, I, C, E, T, Sr, Ii> PrettyGraph<Vertex<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr>, I>
    for SubDigraph<'g, I, Vertex<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr>>
where
    I: UsizeId + Display + 'static,
    Sr: Debug + Clone,
    T: Debug + Clone,
    C: Debug + Clone,
    E: Debug + Clone,
    Ii: UsizeId + Display + 'static,
{
    fn vertices(&self) -> impl Iterator<Item = I> {
        SubDigraph::vertices(self)
    }

    fn vertex(&self, vid: I) -> &Vertex<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr> {
        SubDigraph::<_, _>::vertex(self, vid)
    }

    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> Id {
        use VertexNode::*;

        let internal_id = |i| Vertex::<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr>::identifier(i, ctx);

        match self.vertex(vid).node() {
            TupleGet(from, i) | ArrayGet(from, i) => match self.vertex(*from).node() {
                Array(xs) => internal_id(xs[*i]),
                Sliceable(SliceableNode::Arith { arith, .. }) => internal_id(
                    arith
                        .g
                        .vertex(arith.outputs[*i])
                        .op
                        .unwrap_input_outerid()
                        .clone(),
                ),
                Subgraph(s) => {
                    let from_id = internal_id(vid);
                    s.indirect_from(s.nth_output(*i), &ctx.push(&from_id))
                }
                _ => internal_id(vid),
            },
            _ => internal_id(vid),
        }
    }
}

impl<I, C, E, T, Sr, Ii> PrettyGraph<Vertex<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr>, I>
    for Digraph<I, Vertex<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr>>
where
    I: UsizeId + Display + 'static,
    Sr: Debug + Clone,
    T: Debug + Clone,
    C: Debug + Clone,
    E: Debug + Clone,
    Ii: UsizeId + Display + 'static,
{
    fn vertices(&self) -> impl Iterator<Item = I> {
        Digraph::vertices(self)
    }

    fn vertex(&self, vid: I) -> &Vertex<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr> {
        Digraph::<_, _>::vertex(self, vid)
    }

    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> Id {
        use VertexNode::*;

        let internal_id = |i| Vertex::<I, C, E, SsCg<Ii, I, C, T, Sr>, T, Sr>::identifier(i, ctx);

        match self.vertex(vid).node() {
            TupleGet(from, i) | ArrayGet(from, i) => match self.vertex(*from).node() {
                Array(xs) => internal_id(xs[*i]),
                Sliceable(SliceableNode::Arith { arith, .. }) => internal_id(
                    arith
                        .g
                        .vertex(arith.outputs[*i])
                        .op
                        .unwrap_input_outerid()
                        .clone(),
                ),
                Subgraph(s) => {
                    let from_id = internal_id(vid);
                    s.indirect_from(s.nth_output(*i), &ctx.push(&from_id))
                }
                _ => internal_id(vid),
            },
            _ => internal_id(vid),
        }
    }
}

impl<'g, I, C, E, T, Sr> PrettyGraph<Vertex<I, C, E, (), T, Sr>, I>
    for SubDigraph<'g, I, Vertex<I, C, E, (), T, Sr>>
where
    I: UsizeId + Display + 'static,
    Sr: Debug + Clone,
    T: Debug + Clone,
    C: Debug + Clone,
    E: Debug + Clone,
{
    fn vertices(&self) -> impl Iterator<Item = I> {
        SubDigraph::vertices(self)
    }

    fn vertex(&self, vid: I) -> &Vertex<I, C, E, (), T, Sr> {
        SubDigraph::<_, _>::vertex(self, vid)
    }

    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> Id {
        use VertexNode::*;
        match self.vertex(vid).node() {
            TupleGet(from, i) | ArrayGet(from, i) => {
                let indirect_vid = match self.vertex(*from).node() {
                    Array(xs) => xs[*i],
                    Sliceable(SliceableNode::Arith { arith, .. }) => arith
                        .g
                        .vertex(arith.outputs[*i])
                        .op
                        .unwrap_input_outerid()
                        .clone(),
                    Subgraph(..) => panic!("this variant is not expected"),
                    _ => vid,
                };

                Vertex::<I, C, E, (), T, Sr>::identifier(indirect_vid, ctx)
            }
            _ => Vertex::<I, C, E, (), T, Sr>::identifier(vid, ctx),
        }
    }
}

impl<I, C, E, T, Sr> PrettyGraph<Vertex<I, C, E, (), T, Sr>, I>
    for Digraph<I, Vertex<I, C, E, (), T, Sr>>
where
    I: UsizeId + Display + 'static,
    Sr: Debug + Clone,
    T: Debug + Clone,
    C: Debug + Clone,
    E: Debug + Clone,
{
    fn vertices(&self) -> impl Iterator<Item = I> {
        Digraph::vertices(self)
    }

    fn vertex(&self, vid: I) -> &Vertex<I, C, E, (), T, Sr> {
        Digraph::<_, _>::vertex(self, vid)
    }

    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> Id {
        use VertexNode::*;
        match self.vertex(vid).node() {
            TupleGet(from, i) | ArrayGet(from, i) => {
                let indirect_vid = match self.vertex(*from).node() {
                    Array(xs) => xs[*i],
                    Sliceable(SliceableNode::Arith { arith, .. }) => arith
                        .g
                        .vertex(arith.outputs[*i])
                        .op
                        .unwrap_input_outerid()
                        .clone(),
                    Subgraph(..) => panic!("this variant is not expected"),
                    _ => vid,
                };

                Vertex::<I, C, E, (), T, Sr>::identifier(indirect_vid, ctx)
            }
            _ => Vertex::<I, C, E, (), T, Sr>::identifier(vid, ctx),
        }
    }
}
