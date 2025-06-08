use std::collections::BTreeMap;

use zkpoly_runtime::args::RuntimeType;

use crate::{driver, transit::type2::memory_planning::MemoryBlock};

use super::{Chunk, InstructionIndex, RegisterId, Track, TrackSpecific};

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

#[derive(Default, Debug, Clone)]
struct MemoryBlockRwRelation {
    reads: Vec<InstructionIndex>,
    write: Option<InstructionIndex>,
}

#[derive(Debug, Clone)]
struct MemoryBlockRwWeb {
    blocks: BTreeMap<MemoryBlock, MemoryBlockRwRelation>,
}

impl MemoryBlockRwWeb {
    fn build<'s, Rt: RuntimeType>(chunk: &Chunk<'s, Rt>) -> Self {
        let mut r = Self {
            blocks: BTreeMap::new(),
        };

        for (i, inst) in chunk.iter_instructions() {
            for u in inst.mem_reads() {
                if let Some(mb) = chunk.reg_memory_blocks.get(&u).cloned() {
                    r.blocks.entry(mb).or_default().reads.push(i);
                }
            }

            for mu in inst.mem_writes() {
                if let Some(mb) = chunk.reg_memory_blocks.get(&mu).cloned() {
                    let w = &mut r.blocks.entry(mb).or_default().write;
                    if w.is_some() {
                        panic!("multiple writes to the same memory block not expected");
                    }
                    *w = Some(i);
                }
            }
        }

        r
    }
}

pub fn split<'s, Rt: RuntimeType>(
    chunk: &Chunk<'s, Rt>,
    hd_info: &driver::HardwareInfo,
) -> TrackTasks {
    let assigned_at = chunk.assigned_at();
    let malloc_at = chunk.malloc_at();
    let uses_at = chunk.use_not_deallocate_at();
    let rw_web = MemoryBlockRwWeb::build(chunk);

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
            )
            // - Write-after-Read dependencies
            .chain(
                inst.mem_writes()
                    .map(|reg_id| {
                        let mb = chunk.reg_memory_blocks[&reg_id].clone();
                        let relation = rw_web.blocks.get(&mb).unwrap();
                        if !relation.write.is_some_and(|w| w == i) {
                            panic!("expected mutable use recorded in rw-web");
                        }
                        relation.reads.iter().copied()
                    })
                    .flatten(),
            );

        // Collect last dependency of each track
        let mut depended_of_each_track =
            TrackSpecific::<Vec<InstructionIndex>>::default(hd_info.n_gpus());

        for depended_inst in depended_instructions {
            let depended_track =
                chunk[depended_inst].track(|reg_id| chunk.register_devices[&reg_id]);

            if depended_track.is_cpu() {
                let last = depended_of_each_track
                    .get_track(depended_track)
                    .get(0)
                    .cloned()
                    .unwrap_or_else(|| depended_inst);
                *depended_of_each_track.get_track_mut(depended_track) =
                    vec![std::cmp::max(last, depended_inst)];
            } else {
                depended_of_each_track
                    .get_track_mut(depended_track)
                    .push(depended_inst);
            }
        }

        // Insert synchronization points
        track_tasks.inst_depend.insert(i, Vec::new());

        depended_of_each_track
            .iter(hd_info.n_gpus())
            .for_each(|(_, depended)| {
                depended.iter().for_each(|&depended| {
                    if depended != i {
                        track_tasks.inst_depend.get_mut(&i).unwrap().push(depended);
                    }
                });
            });

        track_tasks.inst_track.insert(i, track);
    }

    track_tasks
}
