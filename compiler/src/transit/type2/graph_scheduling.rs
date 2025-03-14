use std::collections::{BTreeMap, BTreeSet};

use super::{Cg, VertexId};
use zkpoly_common::{digraph::internal::Predecessors, heap::Heap};
use zkpoly_runtime::args::RuntimeType;

fn choose_next_vertex<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
    active_vertices: &BTreeSet<VertexId>,
    ready_vertices: &BTreeSet<VertexId>,
    successors: &Heap<VertexId, BTreeSet<VertexId>>,
    deg_out: &Heap<VertexId, usize>,
) -> VertexId {
    // Number of unexecuted successors of an active vertex, if the vertex has any ready successor
    let num_unexecuted_successors: Vec<(VertexId, usize)> = active_vertices
        .iter()
        .map(|vid| (*vid, deg_out[*vid]))
        .filter(|(vid, _)| {
            successors[*vid]
                .iter()
                .any(|succ| ready_vertices.contains(succ))
        })
        .collect();

    // If no successor of an active vertex is ready, arbitarily choose one ready vertex
    if num_unexecuted_successors.len() == 0 {
        return ready_vertices.iter().next().unwrap().clone();
    }

    // Find ready successors of an active vertex with least unexecuted successors
    let min = num_unexecuted_successors
        .iter()
        .map(|(_, num)| num)
        .min()
        .unwrap()
        .clone();
    let candidate_vids: Vec<VertexId> = num_unexecuted_successors
        .into_iter()
        .filter_map(|(vid, num)| if num == min { Some(vid) } else { None })
        .flat_map(|vid| {
            successors[vid]
                .iter()
                .copied()
                .filter(|succ| ready_vertices.contains(succ))
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
            .map(|pred| cg.g.vertex(pred).typ().size().total())
            .sum();
        (vid, size)
    });
    memory_needed.max_by_key(|(_, size)| *size).unwrap().0
}

/// Schedules a computation graph. It must be a DAG.
///
/// Returns the sequence of the vertices, and the sequence number of the vertex which a vertex dies after.
pub fn schedule<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
) -> (Vec<VertexId>, BTreeMap<VertexId, usize>) {
    let connected = cg.g.connected_component(cg.output);

    let successors = cg.g.successors();
    let mut active_vertices: BTreeSet<VertexId> = BTreeSet::new();
    // Number of unexecuted predecessors
    let mut deg_in = cg.g.degrees_in();
    // Number of unexecuted successors
    let mut deg_out = successors.map_by_ref(&mut |_, successors| {
        successors.iter().filter(|&&succ| connected[succ]).count()
    });

    let mut ready_vertices: BTreeSet<VertexId> = deg_in
        .iter_with_id()
        .filter_map(|(vid, deg)| {
            if *deg == 0 && connected[vid] {
                Some(vid)
            } else {
                None
            }
        })
        .collect();

    let mut counter: usize = 0;
    let mut seq = vec![];
    let mut die_at = BTreeMap::new();

    while ready_vertices.len() != 0 {
        let vid = choose_next_vertex(cg, &active_vertices, &ready_vertices, &successors, &deg_out);

        seq.push(vid);

        ready_vertices.remove(&vid);
        active_vertices.insert(vid);
        for succ in successors[vid].iter().copied() {
            deg_in[succ] -= 1;
            if deg_in[succ] == 0 && connected[succ] {
                ready_vertices.insert(succ);
            }
        }
        for pred in cg.g.vertex(vid).predecessors() {
            deg_out[pred] -= 1;
            if deg_out[pred] == 0 {
                die_at.insert(pred, counter);
                active_vertices.remove(&pred);
            }
        }

        counter += 1;
    }

    (seq, die_at)
}
