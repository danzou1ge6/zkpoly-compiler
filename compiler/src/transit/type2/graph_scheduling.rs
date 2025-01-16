use std::collections::{BTreeMap, BTreeSet};

use super::{Cg, ExprId};
use zkpoly_common::{digraph::internal::Predecessors, heap::Heap};
use zkpoly_runtime::args::RuntimeType;

fn choose_next_vertex<Rt: RuntimeType>(
    cg: &Cg<Rt>,
    active_vertices: &BTreeSet<ExprId>,
    ready_vertices: &BTreeSet<ExprId>,
    successors: &Heap<ExprId, BTreeSet<ExprId>>,
    deg_out: &Heap<ExprId, usize>,
) -> ExprId {
    if active_vertices.len() == 0 {
        return ready_vertices.iter().next().unwrap().clone();
    }
    // Find ready successors of an active vertex with least unexecuted successors

    // Number of unexecuted successors of an active vertex, if the vertex has any ready successor
    let num_unexecuted_successors: Vec<(ExprId, usize)> = active_vertices
        .iter()
        .map(|vid| (*vid, deg_out[*vid]))
        .filter(|(vid, _)| {
            successors[*vid]
                .iter()
                .any(|succ| ready_vertices.contains(succ))
        })
        .collect();
    assert!(num_unexecuted_successors.len() > 0);

    let min = num_unexecuted_successors
        .iter()
        .map(|(_, num)| num)
        .min()
        .unwrap()
        .clone();
    let candidate_vids: Vec<ExprId> = num_unexecuted_successors
        .into_iter()
        .filter_map(|(vid, num)| if num == min { Some(vid) } else { None })
        .filter_map(|vid| {
            if ready_vertices.contains(&vid) {
                Some(vid)
            } else {
                None
            }
        })
        .collect();

    if candidate_vids.len() == 1 {
        return candidate_vids[0];
    }

    // If there are many candidates, find the one whose active predecessors' space requirement is the largest
    let memory_needed = candidate_vids.into_iter().map(|vid| {
        let active_predecessors =
            cg.g.vertex(vid)
                .predecessors()
                .filter(|pred| active_vertices.contains(pred));
        let size: u64 = active_predecessors
            .map(|pred| cg.g.vertex(pred).typ().size())
            .sum();
        (vid, size)
    });
    memory_needed.max_by_key(|(_, size)| *size).unwrap().0
}

pub fn schedule<Rt: RuntimeType>(cg: &Cg<Rt>) -> BTreeMap<ExprId, usize> {
    let mut active_vertices: BTreeSet<ExprId> = BTreeSet::new();
    // Number of unexecuted predecessors
    let mut deg_in = cg.g.degrees_in();
    // Number of unexecuted successors
    let mut deg_out = cg.g.degrees_out();
    let mut ready_vertices: BTreeSet<ExprId> = deg_in
        .iter_with_id()
        .filter_map(|(vid, deg)| if *deg == 0 { Some(vid) } else { None })
        .collect();
    let successors = cg.g.successors();

    let mut counter: usize = 0;
    let mut seq = BTreeMap::new();

    while ready_vertices.len() != 0 {
        let vid = choose_next_vertex(cg, &active_vertices, &ready_vertices, &successors, &deg_out);

        seq.insert(vid, counter);
        counter += 1;

        ready_vertices.remove(&vid);
        active_vertices.insert(vid);
        let successors = successors[vid].clone();
        for succ in successors {
            deg_in[succ] -= 1;
            if deg_in[succ] == 0 {
                ready_vertices.insert(succ);
            }
        }
        for pred in cg.g.vertex(vid).predecessors() {
            deg_out[pred] -= 1;
            if deg_out[pred] == 0 {
                active_vertices.remove(&pred);
            }
        }
    }

    seq
}
