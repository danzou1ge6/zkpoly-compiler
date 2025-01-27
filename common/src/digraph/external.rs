use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::heap::{IdAllocator, UsizeId};

#[derive(Debug, Clone)]
pub struct Digraph<V, E, I> {
    /// vertex v |-> (successor u of v |-> edge uv)
    succ: BTreeMap<I, BTreeMap<I, E>>,
    /// vertex v |-> predecessors of v
    pred: BTreeMap<I, BTreeSet<I>>,
    vertices: BTreeMap<I, V>,
    vertex_allocator: IdAllocator<I>,
}

/// Accessing the graph
impl<V, E, I> Digraph<V, E, I>
where
    I: UsizeId,
{
    pub fn does_vertex_exist(&self, vertex_id: I) -> bool {
        self.succ.contains_key(&vertex_id)
    }
    pub fn has_edge(&self, i: I, j: I) -> bool {
        self.succ[&i].contains_key(&j)
    }
    /// Successors of a vertex
    pub fn successors<'a>(&'a self, vertex_id: I) -> impl Iterator<Item = I> + 'a {
        self.succ[&vertex_id].keys().copied()
    }
    /// Predecessors of a vertex
    pub fn predecessors<'a>(&'a self, vertex_id: I) -> impl Iterator<Item = I> + 'a {
        self.pred[&vertex_id].iter().copied()
    }
    /// Out-going degree of a vertex
    pub fn deg_out(&self, vertex_id: I) -> usize {
        self.succ[&vertex_id].len()
    }
    /// In-coming degree of a vertex
    pub fn deg_in(&self, vertex_id: I) -> usize {
        self.pred[&vertex_id].len()
    }
    /// All vertices' ID
    pub fn vertices<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.vertices.keys().copied()
    }
    /// Vertex by ID
    pub fn vertex<'a>(&'a self, vertex_id: I) -> &'a V {
        &self.vertices[&vertex_id]
    }
    /// Map vertices and edges by `fv` and `fe`, while maintaining graph structure
    pub fn map<E1, V1>(
        self,
        fv: &mut impl FnMut(I, V) -> V1,
        fe: &mut impl FnMut(I, I, E) -> E1,
    ) -> Digraph<V1, E1, I> {
        let succ1 = self
            .succ
            .into_iter()
            .map(|(i, successors)| {
                (
                    i,
                    successors
                        .into_iter()
                        .map(|(j, e)| (j, fe(i, j, e)))
                        .collect(),
                )
            })
            .collect();
        let vertices1 = self
            .vertices
            .into_iter()
            .map(|(i, v)| (i, fv(i, v)))
            .collect();

        Digraph {
            succ: succ1,
            pred: self.pred,
            vertices: vertices1,
            vertex_allocator: IdAllocator::new(),
        }
    }
    /// Visit each vertex in depth-first-search order.
    /// On each vertex is visited, the [`visit`] function is called with parameters
    /// (the visited vertex's ID,
    ///  the predecessor of the visited vertex and the in-coming edge payload (if any),
    ///  the vertex payload)
    /// and returns whether the node has been visited
    pub fn dfs(&self, begin: I, visit: &mut impl FnMut(I, Option<(I, &E)>, &V) -> bool) {
        if visit(begin, None, &self.vertices[&begin]) {
            return;
        }
        for (succ_id, _) in self.succ[&begin].iter() {
            self.dfs(*succ_id, visit);
        }
    }
    /// Returns whether this digraph contains cycles
    pub fn contains_cycles(&self) -> bool {
        let mut visited: BTreeMap<I, bool> = self
            .vertices
            .iter()
            .map(|(vertex_id, _)| (*vertex_id, false))
            .collect();
        let mut crychic = false;

        while let Some((not_visited_vertex_id, _)) =
            visited.iter().find(|(_, flag)| **flag == false)
        {
            self.dfs(*not_visited_vertex_id, &mut |vertex_id, from, vertex| {
                if visited[&vertex_id] {
                    crychic = true;
                    true
                } else {
                    *visited.get_mut(&vertex_id).unwrap() = true;
                    false
                }
            });

            if crychic {
                return true;
            }
        }

        false
    }
    /// Visit each vertex in forward topology order.
    /// On each vertex is visited, the [`visit`] function is called with parameters
    /// (vertex_id, vertex)
    pub fn forward_topology_traverse(&self, visit: &mut impl FnMut(I, &V)) {
        let mut degrees: BTreeMap<I, usize> = self
            .pred
            .iter()
            .map(|(vertex_id, predecessors)| (*vertex_id, predecessors.len()))
            .collect();

        let mut zero_deg_vertices: VecDeque<I> = degrees
            .iter()
            .filter_map(|(vertex_id, deg)| if *deg == 0 { Some(*vertex_id) } else { None })
            .collect();

        let mut cnt = 0;

        while let Some(v) = zero_deg_vertices.pop_front() {
            cnt += 1;
            visit(v, &self.vertices[&v]);
            for u in self.succ[&v].keys() {
                let deg_u = degrees.get_mut(u).unwrap();
                *deg_u -= 1;
                if *deg_u == 0 {
                    zero_deg_vertices.push_back(*u);
                }
            }
        }

        if cnt != degrees.len() {
            panic!(
                "Forward topology traversal did not visit all vertices. Make sure graph is acyclic"
            );
        }
    }
    /// Visit each vertex in backward topology order.
    /// On each vertex is visited, the [`visit`] function is called with parameters
    /// (vertex_id, vertex)
    pub fn backward_topology_traverse(&self, visit: &mut impl FnMut(I, &V)) {
        let mut degrees: BTreeMap<I, usize> = self
            .succ
            .iter()
            .map(|(vertex_id, predecessors)| (*vertex_id, predecessors.len()))
            .collect();

        let mut zero_deg_vertices: VecDeque<I> = degrees
            .iter()
            .filter_map(|(vertex_id, deg)| if *deg == 0 { Some(*vertex_id) } else { None })
            .collect();

        let mut cnt = 0;

        while let Some(v) = zero_deg_vertices.pop_front() {
            cnt += 1;
            visit(v, &self.vertices[&v]);
            for u in self.pred[&v].iter() {
                let deg_u = degrees.get_mut(u).unwrap();
                *deg_u -= 1;
                if *deg_u == 0 {
                    zero_deg_vertices.push_back(*u);
                }
            }
        }

        if cnt != degrees.len() {
            panic!("Backward topology traversal did not visit all vertices. Make sure graph is acyclic");
        }
    }
}

impl<V, E, I> Digraph<V, E, I>
where
    I: UsizeId + std::fmt::Debug,
{
    /// Connecting two existing vertices.
    /// Panics if the edeg already exists.
    pub fn add_edge(&mut self, i: I, j: I, e: E) {
        let successors = self.succ.get_mut(&i).unwrap();
        if successors.contains_key(&j) {
            panic!("edge {i:?}, {j:?} already exists");
        }
        successors.insert(i, e);
        self.pred.get_mut(&j).unwrap().insert(i);
    }
}

/// Editting the graph
impl<V, E, I> Digraph<V, E, I>
where
    I: UsizeId,
{
    pub fn new() -> Self {
        Self {
            succ: BTreeMap::new(),
            pred: BTreeMap::new(),
            vertices: BTreeMap::new(),
            vertex_allocator: IdAllocator::new(),
        }
    }
    /// Allocate a new vertex
    pub fn add_vertex(&mut self, v: V) -> I {
        let i = self.vertex_allocator.alloc();
        self.vertices.insert(i, v);
        self.succ.insert(i, BTreeMap::new());
        i
    }
    /// Removes an existing vertex
    pub fn remove_vertex(&mut self, i: I) -> V {
        let r = self.vertices.remove(&i).unwrap();
        self.succ.remove(&i).unwrap();
        self.pred.remove(&i).unwrap();
        self.succ.iter_mut().for_each(|(_, successors)| {
            successors.remove(&i);
        });
        self.pred.iter_mut().for_each(|(_, predecessors)| {
            predecessors.remove(&i);
        });
        r
    }
    /// Removes an existing edge
    pub fn remove_edge(&mut self, i: I, j: I) -> E {
        let r = self.succ.get_mut(&i).unwrap().remove(&j).unwrap();
        self.pred.get_mut(&j).unwrap().remove(&i);
        r
    }
    /// Remove predecessors of a vertex
    pub fn remove_predecessors(&mut self, i: I) {
        for pred in self.pred[&i].iter() {
            self.succ.get_mut(pred).unwrap().remove(&i).unwrap();
        }
        self.pred.get_mut(&i).unwrap().clear();
    }
    /// Remove successors of a vertex
    pub fn remove_successors(&mut self, i: I) {
        for succ in self.succ[&i].keys() {
            assert!(self.pred.get_mut(succ).unwrap().remove(&i) == true);
        }
        self.succ.get_mut(&i).unwrap().clear();
    }
    /// Vertex by ID, borrowed mutably
    pub fn vertex_mut(&mut self, i: I) -> &mut V {
        self.vertices.get_mut(&i).unwrap()
    }
}
