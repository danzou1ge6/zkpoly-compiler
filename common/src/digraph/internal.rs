use std::collections::VecDeque;

use crate::heap::{Heap, UsizeId};

#[derive(Clone, Debug)]
pub struct Digraph<I, V>(Heap<I, V>);

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

pub struct TopologyIterator<'g, I, V> {
    g: &'g Digraph<I, V>,
    deg: Heap<I, usize>,
    queue: VecDeque<I>,
}

impl<'g, I, V> Iterator for TopologyIterator<'g, I, V>
where
    I: UsizeId,
    V: Predecessors<I>,
{
    type Item = (I, &'g V);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.queue.pop_front() {
            for succ in self.g.0[i].predecessors() {
                self.deg[succ] -= 1;
                if self.deg[succ] == 0 {
                    self.queue.push_back(succ)
                }
            }
            Some((i, &self.g.0[i]))
        } else {
            None
        }
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
    fn degrees(&self) -> Heap<I, usize> {
        let mut deg = Heap::repeat(0, self.order());
        for v in self.0.iter() {
            for succ in v.predecessors() {
                deg[succ] += 1;
            }
        }
        deg
    }
    pub fn topology_sort<'g>(&'g self) -> TopologyIterator<'g, I, V> {
        let deg = self.degrees();
        let queue: VecDeque<I> = deg
            .iter_with_id()
            .filter_map(|(id, d)| if *d == 0 { Some(id) } else { None })
            .collect();
        TopologyIterator {
            g: self,
            deg,
            queue,
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
