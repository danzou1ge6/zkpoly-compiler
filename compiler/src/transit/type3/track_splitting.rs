use zkpoly_runtime::args::RuntimeType;

use super::{Chunk, InstructionIndex, Track, TrackSpecific};

#[derive(Debug, Clone)]
pub enum Task {
    Sync(Track, InstructionIndex),
    Inst(InstructionIndex),
}

pub type TrackTasks = TrackSpecific<Vec<Task>>;

pub fn split<'s, Rt: RuntimeType>(chunk: &Chunk<'s, Rt>) -> TrackTasks {
    let assigned_at = chunk.assigned_at();

    let mut track_tasks = TrackTasks::default();

    for (i, inst) in chunk.iter_instructions() {
        let track = inst.track(|reg_id| chunk.register_devices[&reg_id]);

        let depended_instructions = inst.defs().map(|reg_id| assigned_at[&reg_id]);
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

        last_of_each_track.iter().for_each(|(track, last)| {
            if let Some(last) = *last {
                track_tasks
                    .get_track_mut(track)
                    .push(Task::Sync(track, last));
            }
        });

        track_tasks.get_track_mut(track).push(Task::Inst(i));
    }

    track_tasks
}
