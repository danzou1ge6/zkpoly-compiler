use std::collections::BTreeMap;

use zkpoly_runtime::args::RuntimeType;

use super::{Chunk, InstructionIndex, Track, TrackSpecific};

#[derive(Debug, Clone)]
pub struct TrackTasks {
    pub inst_track: BTreeMap<InstructionIndex, Track>,
    pub inst_depend: BTreeMap<InstructionIndex, Vec<InstructionIndex>>,
}

impl TrackTasks {
    pub fn new() -> Self {
        Self {
            inst_track: BTreeMap::new(),
            inst_depend: BTreeMap::new(),
        }
    }
}

pub fn split<'s, Rt: RuntimeType>(chunk: &Chunk<'s, Rt>) -> TrackTasks {
    let assigned_at = chunk.assigned_at();
    let malloc_at = chunk.malloc_at();
    let uses_at = chunk.use_not_deallocate_at();

    let mut track_tasks = TrackTasks::new();

    for (i, inst) in chunk.iter_instructions() {
        let track = inst.track(|reg_id| chunk.register_devices[&reg_id]);

        // An instruction depends not only on its uses, but also on memory allocations of its defs
        let depended_instructions = inst
            // - For each use
            .uses()
            .map(|reg_id| {
                // - The use's def
                assigned_at
                    .get(&reg_id)
                    .copied()
                    .into_iter()
                    // - The use's malloc
                    .chain(malloc_at.get(&reg_id).copied().into_iter())
                    // - The use's uses, if this instruction is a dealloc
                    .chain({
                        if inst.node.is_dealloc() {
                            Some(uses_at[&reg_id].iter().copied())
                        } else {
                            None
                        }
                        .into_iter()
                        .flatten()
                    })
            })
            .flatten()
            // - Depdends on malloc of defs
            .chain(
                inst.defs()
                    .filter_map(|reg_id| malloc_at.get(&reg_id).copied()),
            );

        // Collect last dependency of each track
        let mut last_of_each_track = TrackSpecific::<Option<InstructionIndex>>::default();

        for depended_inst in depended_instructions {
            let depended_track =
                chunk[depended_inst].track(|reg_id| chunk.register_devices[&reg_id]);
            let last = last_of_each_track
                .get_track(depended_track)
                .unwrap_or_else(|| depended_inst);
            last_of_each_track
                .get_track_mut(depended_track)
                .replace(std::cmp::max(last, depended_inst));
        }

        // Insert synchronization points
        track_tasks.inst_depend.insert(i, Vec::new());

        last_of_each_track
            .iter()
            .filter_map(|(track, last)| Some((track, last.as_ref()?)))
            .for_each(|(_, &last)| {
                if last != i {
                    track_tasks.inst_depend.get_mut(&i).unwrap().push(last);
                }
            });

        track_tasks.inst_track.insert(i, track);
    }

    track_tasks
}
