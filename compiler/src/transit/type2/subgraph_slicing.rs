use std::collections::{HashMap, HashSet, VecDeque};

use zkpoly_common::{digraph::internal::Digraph, heap::Heap};
use zkpoly_runtime::args::RuntimeType;

use super::{sliceable_subgraph, template::SliceableProperty, Typ, VertexId};

struct DenseSet(Heap<VertexId, bool>);

impl std::ops::Deref for DenseSet {
    type Target = Heap<VertexId, bool>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for DenseSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl DenseSet {
    pub fn empty(order: usize) -> Self {
        Self(Heap::repeat(false, order))
    }

    pub fn add(&mut self, v: VertexId) {
        self.0[v] = true;
    }

    pub fn remove(&mut self, v: VertexId) {
        self.0[v] = false;
    }
}

struct Subgraph<'g, 's, Rt: RuntimeType> {
    vertices: DenseSet,
    forward_reachable: DenseSet,
    backward_reachable: DenseSet,
    g: super::CgSubgraph<'g, 's, Rt>,
    order: usize,
}

fn dfs<'s, 'g, It>(
    v: VertexId,
    next_vertices: impl Fn(VertexId) -> It,
    reachable: &mut DenseSet,
    subgraph: &DenseSet,
) where
    It: Iterator<Item = VertexId>,
{
    if reachable[v] || subgraph[v] {
        return;
    }

    reachable.add(v);

    for w in next_vertices(v) {
        dfs(w, &next_vertices, reachable, subgraph);
    }
}

impl<'s, 'g, Rt: RuntimeType> Subgraph<'g, 's, Rt> {
    fn start_with(v: VertexId, cg: super::CgSubgraph<'g, 's, Rt>) -> Self {
        let mut r = Self {
            vertices: DenseSet::empty(cg.full_order()),
            forward_reachable: DenseSet::empty(cg.full_order()),
            backward_reachable: DenseSet::empty(cg.full_order()),
            g: cg,
            order: 0,
        };
        let _ = r.add_vertex(v);

        r
    }

    fn add_vertex(&mut self, v: VertexId) {
        self.vertices.add(v);
        dfs(
            v,
            |u| self.g.predecessors_of(u),
            &mut self.backward_reachable,
            &self.vertices,
        );
        dfs(
            v,
            |u| self.g.successors_of(u),
            &mut self.forward_reachable,
            &self.vertices,
        );
        self.order += 1;
    }

    fn is_forward_reachable(&self, v: VertexId) -> bool {
        self.forward_reachable[v]
    }

    fn is_backward_reachable(&self, v: VertexId) -> bool {
        self.backward_reachable[v]
    }

    fn contains(&self, v: VertexId) -> bool {
        self.vertices[v]
    }
}

struct GrowingSubgraph<'g, 's, Rt: RuntimeType> {
    subgraph: Subgraph<'g, 's, Rt>,
    frontier: VecDeque<VertexId>,
}

impl<'g, 's, Rt: RuntimeType> GrowingSubgraph<'g, 's, Rt> {
    fn start_with(v: VertexId, cg: super::CgSubgraph<'g, 's, Rt>) -> Self {
        Self {
            subgraph: Subgraph::start_with(v, cg),
            frontier: [v].into(),
        }
    }

    fn grow(&mut self, v: VertexId) {
        self.subgraph.add_vertex(v);
        self.frontier.push_back(v);
    }

    fn try_grow(&mut self) -> bool {
        use super::template::SliceableProperty::*;
        if let Some(v) = self.frontier.pop_front() {
            for u in self
                .subgraph
                .g
                .successors_of(v)
                .collect::<Vec<_>>()
                .into_iter()
            {
                if !self.subgraph.contains(u)
                    && !self.subgraph.is_backward_reachable(u)
                    && self.subgraph.g.vertex(v).node().sliceable_property() != Last
                    && matches!(
                        self.subgraph.g.vertex(u).node().sliceable_property(),
                        Always | NonFirst | Last
                    )
                    && self.subgraph.g.vertex(u).typ().is_poly_pow2()
                {
                    self.grow(u);
                }
            }
            for u in self
                .subgraph
                .g
                .predecessors_of(v)
                .collect::<Vec<_>>()
                .into_iter()
            {
                // We are allowing NonFirst here, and if we later find some NonFirst vertex becomes first,
                // we will remove it from the subgraph before building the super vertex
                if !self.subgraph.contains(u)
                    && !self.subgraph.is_forward_reachable(u)
                    && matches!(
                        self.subgraph.g.vertex(u).node().sliceable_property(),
                        Always | NonFirst
                    )
                    && self.subgraph.g.vertex(u).typ().is_poly_pow2()
                {
                    self.grow(u);
                }
            }
            true
        } else {
            false
        }
    }

    fn as_large_as_possible(mut self) -> (DenseSet, super::CgSubgraph<'g, 's, Rt>, usize) {
        while self.try_grow() {}
        (self.subgraph.vertices, self.subgraph.g, self.subgraph.order)
    }
}

fn fuse_sliceable_subgraphs_from<'s, Rt: RuntimeType>(
    v: VertexId,
    cg: &mut super::Cg<'s, Rt>,
    fused_vids: &mut DenseSet,
    subgraph_minimum_order: usize,
) {
    let (mut subgraph, connected_cg, subgraph_order) =
        GrowingSubgraph::start_with(v, cg.connected_subgraph()).as_large_as_possible();

    // Remove those NonFirst vertices if they are first.
    // CAUTION that for now, only TupleGet's are NonFirst, and tuples can't be nested,
    // therefore we don't need to consider the case when removing a vertex causes other vertices
    // to become first.
    subgraph.ids().for_each(|v| {
        if connected_cg.vertex(v).node().sliceable_property() == SliceableProperty::NonFirst
            && !connected_cg.predecessors_of(v).all(|u| subgraph[u])
        {
            subgraph.remove(v)
        }
    });
    let subgraph = subgraph;

    if subgraph_order < subgraph_minimum_order {
        return;
    }

    let mut sub2big: Heap<sliceable_subgraph::VertexId, VertexId> =
        subgraph
            .iter_with_id()
            .fold(Heap::new(), |mut acc, (v, selected)| {
                if *selected {
                    acc.push(v);
                }
                acc
            });
    sub2big.iter().for_each(|v| fused_vids.add(*v));

    // Inputs are predecessors of the subgraph no in subgraph
    let subgraph_inputs_vids = sub2big
        .iter_with_id()
        .map(|(_sub_v, v)| connected_cg.predecessors_of(*v).filter(|u| !subgraph[*u]))
        .flatten()
        .collect::<HashSet<_>>();
    let subgraph_inputs_sub_vids = subgraph_inputs_vids
        .iter()
        .map(|v| sub2big.push(*v))
        .collect::<Vec<_>>();
    let sub2big = sub2big;

    let big2sub = sub2big
        .iter_with_id()
        .fold(HashMap::new(), |mut acc, (sub_v, v)| {
            acc.insert(*v, sub_v);
            acc
        });

    // Outputs are those in subgraph that have successors not in subgraph
    let subgraph_outputs = sub2big
        .iter_with_id()
        .filter_map(|(sub_v, v)| {
            if !connected_cg.successors_of(*v).all(|u| subgraph[u]) {
                Some((sub_v, *v))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // Relabel external vertices to subgraph vertices
    let mut sub_cg_vertices: Heap<sliceable_subgraph::VertexId, _> =
        sub2big.map_by_ref(&mut |_sub_v, v| {
            let outer_vertex = connected_cg.vertex(*v);
            let subgraph_node = sliceable_subgraph::VertexNode::relabeled_external_node(
                *v,
                outer_vertex.node(),
                |v| big2sub[&v],
            );
            sliceable_subgraph::Vertex::new(
                subgraph_node,
                outer_vertex.typ().clone(),
                outer_vertex.src().clone(),
            )
        });

    // Create output vertex in subgraph
    let subgraph_output_typ = Typ::Tuple(
        subgraph_outputs
            .iter()
            .map(|(_, v)| connected_cg.vertex(*v).typ().clone())
            .collect(),
    );
    let subgraph_output_src = subgraph_outputs
        .first()
        .map(|(_, v)| connected_cg.vertex(*v).src().clone())
        .unwrap();

    let subgraph_return_vid = sub_cg_vertices.push(sliceable_subgraph::Vertex::new(
        sliceable_subgraph::VertexNode::Return(
            subgraph_outputs.iter().map(|(sub_v, _)| *sub_v).collect(),
        ),
        subgraph_output_typ.clone(),
        subgraph_output_src.clone(),
    ));

    // Add subgraph to big graph
    let subgraph = sliceable_subgraph::Cg::new(
        subgraph_inputs_sub_vids.iter().cloned().collect(),
        Digraph::from_vertices(sub_cg_vertices),
        subgraph_return_vid,
    );
    let subgraph_vid = cg.g.add_vertex(super::Vertex::new(
        super::VertexNode::Subgraph(subgraph),
        subgraph_output_typ,
        subgraph_output_src,
    ));

    // Replace output vertices in original graph with TupleGet
    subgraph_outputs.iter().enumerate().for_each(|(i, (_, v))| {
        *cg.g.vertex_mut(*v).node_mut() = super::VertexNode::TupleGet(subgraph_vid, i);
    });
}

pub fn fuse<'s, Rt: RuntimeType>(
    cg: super::no_subgraph::Cg<'s, Rt>,
    subgraph_minimum_order: usize,
) -> super::Cg<'s, Rt> {
    let mut cg = cg.map_vertices(|v| v.map_node(|vn| vn.with_s()));

    let seq =
        cg.g.dfs_from(cg.output)
            .map(|(vid, _)| vid)
            .collect::<Vec<_>>();
    let mut fused_vids = DenseSet::empty(cg.g.order());

    for vid in seq {
        if fused_vids[vid] {
            continue;
        }

        fuse_sliceable_subgraphs_from(vid, &mut cg, &mut fused_vids, subgraph_minimum_order);
    }

    cg
}
