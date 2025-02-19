use std::collections::{BTreeSet, VecDeque};

use crate::heap::{Heap, UsizeId};

#[derive(Clone, Debug)]
pub struct Digraph<I, V>(pub Heap<I, V>);

impl<I, V> Digraph<I, V> {
    pub fn new() -> Self {
        Self(Heap::new())
    }
}

impl<I, V> Default for Digraph<I, V> {
    fn default() -> Self {
        Self::new()
    }
}

pub trait Predecessors<I> {
    fn predecessors(&self) -> impl Iterator<Item = I>;
}

pub struct DfsIterator<'g, I, V> {
    g: &'g Digraph<I, V>,
    visited: Heap<I, bool>,
    stack: Vec<I>,
}

impl<'g, I, V> Iterator for DfsIterator<'g, I, V>
where
    I: UsizeId,
    V: Predecessors<I>,
{
    type Item = (I, &'g V);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.stack.pop() {
            for succ in self.g.0[v].predecessors() {
                if !self.visited[succ] {
                    self.visited[succ] = true;
                    self.stack.push(succ);
                }
            }
            Some((v, &self.g.0[v]))
        } else {
            None
        }
    }
}

impl<'g, I, V> DfsIterator<'g, I, V> {
    pub fn add_begin(mut self, i: I) -> Self {
        self.stack.push(i);
        self
    }
}

pub struct TopologyIterator<'g, 's, I, V> {
    g: &'g Digraph<I, V>,
    deg: Heap<I, usize>,
    queue: VecDeque<I>,
    successors: &'s Heap<I, BTreeSet<I>>,
}

fn topology_sort_iterator_next<'g, I, V>(
    g: &'g Digraph<I, V>,
    deg: &mut Heap<I, usize>,
    queue: &mut VecDeque<I>,
    successors: &Heap<I, BTreeSet<I>>,
) -> Option<(I, &'g V)>
where
    I: UsizeId,
{
    if let Some(i) = queue.pop_front() {
        for succ in successors[i].iter().copied() {
            deg[succ] -= 1;
            if deg[succ] == 0 {
                queue.push_back(succ)
            }
        }
        Some((i, &g.0[i]))
    } else {
        None
    }
}

impl<'g, 's, I, V> Iterator for TopologyIterator<'g, 's, I, V>
where
    I: UsizeId,
    V: Predecessors<I>,
{
    type Item = (I, &'g V);
    fn next(&mut self) -> Option<Self::Item> {
        topology_sort_iterator_next(self.g, &mut self.deg, &mut self.queue, self.successors)
    }
}

pub struct OwnSuccessorTopologyIterator<'g, I, V> {
    g: &'g Digraph<I, V>,
    deg: Heap<I, usize>,
    queue: VecDeque<I>,
    successors: Heap<I, BTreeSet<I>>,
}

impl<'g, I, V> Iterator for OwnSuccessorTopologyIterator<'g, I, V>
where
    I: UsizeId,
    V: Predecessors<I>,
{
    type Item = (I, &'g V);
    fn next(&mut self) -> Option<Self::Item> {
        topology_sort_iterator_next(self.g, &mut self.deg, &mut self.queue, &self.successors)
    }
}

impl<I, V> Digraph<I, V>
where
    I: UsizeId,
    V: Predecessors<I>,
{
    pub fn dfs<'g>(&'g self) -> DfsIterator<'g, I, V> {
        DfsIterator {
            g: self,
            visited: Heap::repeat(false, self.order()),
            stack: Vec::new(),
        }
    }
    pub fn dfs_from<'g>(&'g self, i: I) -> DfsIterator<'g, I, V> {
        self.dfs().add_begin(i)
    }
    pub fn degrees_in(&self) -> Heap<I, usize> {
        let deg: Heap<I, usize> = self.0.map_by_ref(&mut |_, v| v.predecessors().count());
        deg
    }
    pub fn degrees_out(&self) -> Heap<I, usize> {
        let mut deg = Heap::repeat(0, self.order());
        for v in self.0.iter() {
            for succ in v.predecessors() {
                deg[succ] += 1;
            }
        }
        deg
    }

    pub fn successors(&self) -> Heap<I, BTreeSet<I>> {
        let mut succ = Heap::repeat(BTreeSet::new(), self.order());
        for (vid, v) in self.0.iter_with_id() {
            for succ_id in v.predecessors() {
                succ[succ_id].insert(vid);
            }
        }
        succ
    }
}

impl<I, V> Digraph<I, V>
where
    I: UsizeId,
    V: Predecessors<I>,
{
    pub fn topology_sort<'g>(&'g self) -> impl Iterator<Item = (I, &'g V)> + 'g {
        let successors = self.successors();
        let deg = self.degrees_in();
        let queue: VecDeque<I> = deg
            .iter_with_id()
            .filter_map(|(id, d)| if *d == 0 { Some(id) } else { None })
            .collect();
        OwnSuccessorTopologyIterator {
            g: self,
            deg,
            queue,
            successors,
        }
    }

    pub fn topology_sort_with_successors<'g, 's>(
        &'g self,
        successors: &'s Heap<I, BTreeSet<I>>,
    ) -> TopologyIterator<'g, 's, I, V> {
        let deg = self.degrees_in();
        let queue: VecDeque<I> = deg
            .iter_with_id()
            .filter_map(|(id, d)| if *d == 0 { Some(id) } else { None })
            .collect();
        TopologyIterator {
            g: self,
            deg,
            queue,
            successors,
        }
    }
}

impl<I, V> Digraph<I, V>
where
    I: UsizeId,
{
    pub fn vertex(&self, i: I) -> &V {
        &self.0[i]
    }
    pub fn vertex_mut(&mut self, i: I) -> &mut V {
        &mut self.0[i]
    }
    pub fn map<V1, I1>(self, f: &mut impl FnMut(I, V) -> V1) -> Digraph<I1, V1> {
        Digraph(self.0.map(f))
    }
    pub fn map_by_ref<V1, I1>(&self, f: &mut impl FnMut(I, &V) -> V1) -> Digraph<I1, V1> {
        Digraph(self.0.map_by_ref(f))
    }
    pub fn vertices(&self) -> impl Iterator<Item = I> {
        self.0.ids()
    }
    pub fn add_vertex(&mut self, v: V) -> I {
        self.0.push(v)
    }
}

impl<I, V> Digraph<I, V> {
    pub fn order(&self) -> usize {
        self.0.len()
    }
}
