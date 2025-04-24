use std::collections::{BTreeMap, BTreeSet};

use super::{Cg, VertexId};
use zkpoly_common::{digraph::internal::Predecessors, heap::Heap, mm_heap::MmHeap};
use zkpoly_runtime::args::RuntimeType;

type ActiveRanking = (bool, i64);

struct Ctx {
    /// The estimated benifit of trying to kill an active vertex ASAP
    active_vertices: MmHeap<VertexId, ActiveRanking>,
    /// The estimated benefit of killing the vertex after it is activated
    unexecuted_score: BTreeMap<VertexId, i64>,
    ready_vertices: BTreeSet<VertexId>,
    n_unexecuted_successors: Heap<VertexId, usize>,
    n_unexecuted_predecessors: Heap<VertexId, usize>,
}

impl Ctx {
    fn choose_next_vertex(&self, successors: &Heap<VertexId, BTreeSet<VertexId>>) -> VertexId {
        if let Some(victim) = self
            .active_vertices
            .max()
            .map(|((x, _), vid)| if *x { Some(*vid) } else { None })
            .flatten()
        {
            // Arbitrarily choose a ready successor of the victim
            successors[victim]
                .iter()
                .filter(|&x| self.ready_vertices.contains(x))
                .next()
                .unwrap()
                .clone()
        } else {
            // If no active vertex has unexecuted ready successor, arbitrarily choose a ready vertex
            self.ready_vertices.first().unwrap().clone()
        }
    }

    fn active_ranking<'s, Rt: RuntimeType>(
        &self,
        cg: &Cg<'s, Rt>,
        successors: &Heap<VertexId, BTreeSet<VertexId>>,
        vid: VertexId,
    ) -> ActiveRanking {
        // Sum of the size of the unexecuted successors of a vertex
        let cost: u64 = successors[vid]
            .iter()
            .filter(|&x| self.unexecuted_score.contains_key(x))
            .filter(|&x| self.unexecuted_score[x] < self.active_vertices[&vid].1)
            .map(|x| cg.g.vertex(*x).typ().size().total())
            .sum();
        let gain = cg.g.vertex(vid).typ().size().total();
        let score = gain as i64 - cost as i64;

        let has_ready_successor = successors[vid]
            .iter()
            .any(|&x| self.ready_vertices.contains(&x));

        (has_ready_successor, score)
    }

    fn execute<'s, Rt: RuntimeType>(
        &mut self,
        cg: &Cg<'s, Rt>,
        successors: &Heap<VertexId, BTreeSet<VertexId>>,
        connected: &Heap<VertexId, bool>,
        vid: VertexId,
        mut die: impl FnMut(VertexId),
    ) {
        self.ready_vertices.remove(&vid);

        let has_ready_successor = successors[vid]
            .iter()
            .any(|&x| self.ready_vertices.contains(&x));

        let score = self.unexecuted_score.remove(&vid).unwrap();
        self.active_vertices
            .insert(vid, (has_ready_successor, score));

        for pred in cg.g.vertex(vid).predecessors() {
            // Update the active ranking of the predecessor
            self.active_vertices
                .insert(pred, self.active_ranking(cg, successors, pred));

            self.n_unexecuted_successors[pred] -= 1;
            if self.n_unexecuted_successors[pred] == 0 {
                self.active_vertices.remove(&pred);
                die(pred);
            }
        }

        for succ in successors[vid].iter().copied().filter(|x| connected[*x]) {
            self.n_unexecuted_predecessors[succ] -= 1;
            if self.n_unexecuted_predecessors[succ] == 0 {
                self.ready_vertices.insert(succ);
            }
        }
    }

    fn new<'s, Rt: RuntimeType>(
        cg: &Cg<'s, Rt>,
        successors: &Heap<VertexId, BTreeSet<VertexId>>,
        connected: &Heap<VertexId, bool>,
    ) -> Self {
        let deg_in = cg.g.degrees_in();

        let ready_vertices: BTreeSet<VertexId> = deg_in
            .iter_with_id()
            .filter_map(|(vid, deg)| {
                if *deg == 0 && connected[vid] {
                    Some(vid)
                } else {
                    None
                }
            })
            .collect();

        let n_unexecuted_successors = successors.map_by_ref(&mut |_, successors| {
            successors.iter().filter(|&&succ| connected[succ]).count()
        });

        let unexecuted_score =
            cg.g.topology_sort_inv()
                .fold(BTreeMap::new(), |mut acc, (vid, _)| {
                    let cost = successors[vid]
                        .iter()
                        .filter(|&&succ| connected[succ])
                        .map(|&succ| acc[&succ])
                        .sum::<i64>();
                    let gain = cg.g.vertex(vid).typ().size().total();
                    let score = gain as i64 - cost as i64;

                    acc.insert(vid, score);
                    acc
                });

        Ctx {
            active_vertices: MmHeap::new(),
            unexecuted_score,
            ready_vertices,
            n_unexecuted_successors,
            n_unexecuted_predecessors: deg_in,
        }
    }
}

/// Schedules a computation graph. It must be a DAG.
///
/// Returns the sequence of the vertices, and the sequence number of the vertex which a vertex dies after.
pub fn schedule<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
) -> (Vec<VertexId>, BTreeMap<VertexId, usize>) {
    let connected = cg.g.connected_component(cg.output);
    let total = connected.iter().filter(|&x| *x).count();

    let successors = cg.g.successors();
    let mut ctx = Ctx::new(cg, &successors, &connected);

    let mut counter: usize = 0;
    let mut seq = vec![];
    let mut die_at = BTreeMap::new();

    while ctx.ready_vertices.len() != 0 {
        let vid = ctx.choose_next_vertex(&successors);
        seq.push(vid);

        ctx.execute(cg, &successors, &connected, vid, |dead| {
            die_at.insert(dead, counter);
        });

        counter += 1;

        print!("[Scheduler] Scheduled {} / {}\r", counter, total);
    }
    println!();

    (seq, die_at)
}
