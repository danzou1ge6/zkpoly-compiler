mod prelude {
    pub use std::collections::{BTreeMap, BTreeSet};

    pub use super::super::{Cg, CgSubgraph, VertexId};
    pub use zkpoly_common::{
        digraph::internal::{Predecessors, Successors},
        heap::Heap,
        mm_heap::MmHeap,
    };
    pub use zkpoly_runtime::args::RuntimeType;
}

use prelude::*;

trait Scheduler {
    fn total(&self) -> usize;
    fn choose_next_vertex(&self) -> Option<VertexId>;
    fn execute(&mut self, vid: VertexId, die: impl FnMut(VertexId));
}

mod kill_asap;

fn scheduler_with_scheduler<S: Scheduler>(
    scheduler: &mut S,
) -> (Vec<VertexId>, BTreeMap<VertexId, usize>) {
    let mut counter: usize = 0;
    let mut seq = vec![];
    let mut die_at = BTreeMap::new();

    while let Some(vid) = scheduler.choose_next_vertex() {
        seq.push(vid);

        scheduler.execute(vid, |dead| {
            die_at.insert(dead, counter);
        });

        counter += 1;
    }
    println!();

    (seq, die_at)
}

/// Schedules a computation graph. It must be a DAG.
///
/// Returns the sequence of the vertices, and the sequence number of the vertex which a vertex dies after.
pub fn schedule<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
) -> (Vec<VertexId>, BTreeMap<VertexId, usize>) {
    let connected = cg.g.connected_component(cg.output);
    let g = CgSubgraph::new(&cg.g, connected);

    let mut scheduler = kill_asap::Scheduler::new(g);

    scheduler_with_scheduler(&mut scheduler)
}
