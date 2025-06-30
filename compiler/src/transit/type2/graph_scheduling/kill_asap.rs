use super::prelude::*;

type ActiveRanking = (bool, i64);

pub struct Scheduler<'a, 's, Rt: RuntimeType> {
    cg: CgSubgraph<'a, 's, Rt>,
    /// The estimated benifit of trying to kill an active vertex ASAP
    active_vertices: MmHeap<VertexId, ActiveRanking>,
    /// The estimated benefit of killing the vertex after it is activated
    unexecuted_score: BTreeMap<VertexId, i64>,
    ready_vertices: BTreeSet<VertexId>,
    n_unexecuted_successors: Heap<VertexId, usize>,
    n_unexecuted_predecessors: Heap<VertexId, usize>,
}

impl<'a, 's, Rt: RuntimeType> Scheduler<'a, 's, Rt> {
    pub fn new(cg: CgSubgraph<'a, 's, Rt>) -> Self {
        let ready_vertices: BTreeSet<VertexId> = cg
            .vertices()
            .filter_map(|vid| if cg.deg_in(vid) == 0 { Some(vid) } else { None })
            .collect();

        let unexecuted_score = cg
            .topology_sort_inv()
            .fold(BTreeMap::new(), |mut acc, (vid, _)| {
                let cost = cg
                    .successors_of(vid)
                    .map(|succ| cg.vertex(succ).typ().size().total() as i64)
                    .sum::<i64>();
                let gain = cg.vertex(vid).typ().size().total();
                let score = gain as i64 - cost as i64;

                acc.insert(vid, score);
                acc
            });

        Scheduler {
            active_vertices: MmHeap::new(),
            unexecuted_score,
            ready_vertices,
            n_unexecuted_successors: cg.degrees_out(),
            n_unexecuted_predecessors: cg.degrees_in().clone(),
            cg,
        }
    }

    fn active_ranking(
        &self,
        vid: VertexId,
    ) -> ActiveRanking {
        // Sum of the size of the unexecuted successors of a vertex
        let cost: u64 = self.cg.successors_of(vid)
            .filter(|x| self.unexecuted_score.contains_key(x))
            .filter(|x| self.unexecuted_score[x] < self.active_vertices[&vid].1)
            .map(|x| self.cg.vertex(x).typ().size().total())
            .sum();
        let gain = self.cg.vertex(vid).typ().size().total();
        let score = gain as i64 - cost as i64;

        let has_ready_successor = self.cg.successors_of(vid)
            .any(|x| self.ready_vertices.contains(&x));

        (has_ready_successor, score)
    }
}

impl<'a, 's, Rt: RuntimeType> super::Scheduler for Scheduler<'a, 's, Rt> {
    fn choose_next_vertex(&self) -> Option<VertexId> {
        if self.ready_vertices.is_empty() {
            return None;
        }

        let next = if let Some(victim) = self
            .active_vertices
            .max()
            .map(|((x, _), vid)| if *x { Some(*vid) } else { None })
            .flatten()
        {
            // Arbitrarily choose a ready successor of the victim
            self.cg
                .successors_of(victim)
                .filter(|x| self.ready_vertices.contains(x))
                .next()
                .unwrap()
                .clone()
        } else {
            // If no active vertex has unexecuted ready successor, arbitrarily choose a ready vertex
            self.ready_vertices.first().unwrap().clone()
        };

        Some(next)
    }

    fn execute(&mut self, vid: VertexId, mut die: impl FnMut(VertexId)) {
        self.ready_vertices.remove(&vid);

        let has_ready_successor = self.cg.successors_of(vid)
            .any(|x| self.ready_vertices.contains(&x));

        let score = self.unexecuted_score.remove(&vid).unwrap();
        self.active_vertices
            .insert(vid, (has_ready_successor, score));

        for pred in
            self.cg.vertex(vid)
                .predecessors()
                .collect::<BTreeSet<_>>()
                .into_iter()
        {
            // Update the active ranking of the predecessor
            self.active_vertices
                .insert(pred, self.active_ranking(pred));

            self.n_unexecuted_successors[pred] -= 1;
            if self.n_unexecuted_successors[pred] == 0 {
                self.active_vertices.remove(&pred);
                die(pred);
            }
        }

        for succ in self.cg.successors_of(vid) {
            self.n_unexecuted_predecessors[succ] -= 1;
            if self.n_unexecuted_predecessors[succ] == 0 {
                self.ready_vertices.insert(succ);
            }
        }
    }

    fn total(&self) -> usize {
        self.cg.order()
    }
}
