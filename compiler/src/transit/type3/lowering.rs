use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::transit::type2;
use crate::transit::type2::object_analysis::size::LogBlockSizes;

use super::track_splitting::TrackTasks;
use super::{Track, VertexNode};
use kernel_gen::GeneratedFunctions;
use zkpoly_common::define_usize_id;
use zkpoly_common::devices::DeviceType;
use zkpoly_common::heap::{Heap, IdAllocator};
use zkpoly_common::load_dynamic::Libs;
use zkpoly_common::typ::Typ;
use zkpoly_runtime::args::{RuntimeType, VariableId};
use zkpoly_runtime::devices::{EventId, EventType, EventTypeTable, ThreadId};
use zkpoly_runtime::functions::FunctionTable;
use zkpoly_runtime::instructions::{AllocMethod, Instruction, InstructionNode};

mod emit_func;
mod kernel_gen;

#[derive(Debug, Clone)]
struct Cell {
    thread: ThreadId,
    inst: Instruction,
    tail: Vec<Instruction>,
}

impl Cell {
    pub fn new(inst: Instruction, tid: ThreadId) -> Self {
        Cell {
            inst,
            tail: vec![],
            thread: tid,
        }
    }
}

define_usize_id!(InstructionId);

#[derive(Debug, Clone)]
pub struct MultithreadChunk {
    instructions: Heap<InstructionId, Cell>,
    threads: Heap<ThreadId, Vec<InstructionId>>,
    primary_thread_id: ThreadSpecific<ThreadId>,
    t3idx2id: BTreeMap<super::InstructionIndex, InstructionId>,
    forks: BTreeMap<ThreadId, (ThreadId, InstructionId)>,
}

impl MultithreadChunk {
    pub fn new() -> Self {
        let mut threads = Heap::new();
        let primary_thread_id = ThreadSpecific::new(|| threads.push(vec![]));
        MultithreadChunk {
            instructions: Heap::new(),
            threads,
            primary_thread_id,
            t3idx2id: BTreeMap::new(),
            forks: BTreeMap::new(),
        }
    }

    pub fn emit(&mut self, thread_id: ThreadId, instruction: Instruction) -> InstructionId {
        let inst_id = self.instructions.push(Cell::new(instruction, thread_id));
        self.threads[thread_id].push(inst_id);
        inst_id
    }

    pub fn emit_primary(
        &mut self,
        thread: PrimaryThread,
        instruction: Instruction,
    ) -> InstructionId {
        self.emit(*self.primary_thread_id.get(thread), instruction)
    }

    pub fn emit_with_idx(
        &mut self,
        t3idx: super::InstructionIndex,
        thread_id: ThreadId,
        instruction: Instruction,
    ) {
        let inst_id: InstructionId = self.instructions.push(Cell::new(instruction, thread_id));
        self.t3idx2id.insert(t3idx, inst_id);
        self.threads[thread_id].push(inst_id);
    }

    pub fn emit_primary_with_idx(
        &mut self,
        t3idx: super::InstructionIndex,
        thread: PrimaryThread,
        instruction: Instruction,
    ) {
        self.emit_with_idx(t3idx, *self.primary_thread_id.get(thread), instruction);
    }

    pub fn append_at(&mut self, t3idx: super::InstructionIndex, tail: Instruction) {
        self.instructions[*self
            .t3idx2id
            .get(&t3idx)
            .unwrap_or_else(|| panic!("instruction for type3 {:?} not found", t3idx))]
        .tail
        .push(tail);
    }

    pub fn new_auxilary_thread(&mut self) -> ThreadId {
        self.threads.push(vec![])
    }

    pub fn emit_fork(&mut self, pthread: PrimaryThread, fork_to: ThreadId) {
        let fork_inst_id = self.emit_primary(
            pthread,
            Instruction::new(
                InstructionNode::Fork {
                    new_thread: fork_to,
                    instructions: Vec::new(),
                },
                Track::Cpu.into(),
                None,
            ),
        );
        self.forks.insert(
            fork_to,
            (*self.primary_thread_id.get(pthread), fork_inst_id),
        );
    }

    pub fn thread_instructions(&self, thread_id: ThreadId) -> impl Iterator<Item = &Instruction> {
        let inst_ids = &self.threads[thread_id];
        inst_ids.iter().flat_map(|&inst_id| {
            std::iter::once(&self.instructions[inst_id].inst)
                .chain(self.instructions[inst_id].tail.iter())
        })
    }

    pub fn primary_thread_instructions(
        &self,
        pthread: PrimaryThread,
    ) -> impl Iterator<Item = &Instruction> {
        self.thread_instructions(*self.primary_thread_id.get(pthread))
    }

    pub fn fillback_auxiliary_instructions(&mut self) {
        for (&fork_to, &(_, fork_inst_id)) in self.forks.iter() {
            let aux_inst = self.thread_instructions(fork_to).cloned().collect();

            let fork_inst = &mut self.instructions[fork_inst_id];
            if let InstructionNode::Fork { instructions, .. } = fork_inst.inst.node_mut() {
                *instructions = aux_inst;
            } else {
                panic!("expect to fillback to a fork instruction")
            }
        }
    }

    pub fn thread_of(&self, t3idx: super::InstructionIndex) -> ThreadId {
        let inst_id = self.t3idx2id[&t3idx];
        self.instructions[inst_id].thread
    }
}

impl From<super::Device> for DeviceType {
    fn from(value: super::Device) -> Self {
        match value {
            super::Device::Cpu => DeviceType::CPU,
            super::Device::Gpu(i) => DeviceType::GPU {
                device_id: i as i32,
            },
            super::Device::Stack => DeviceType::CPU,
            super::Device::Disk => DeviceType::Disk,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrimaryThread {
    MemoryManagement,
    Gpu,
    Cpu,
    ToDisk,
    FromDisk,
}

impl PrimaryThread {
    pub fn main() -> Self {
        PrimaryThread::MemoryManagement
    }
}

#[derive(Debug, Clone)]
pub struct ThreadSpecific<T> {
    memory_management: T,
    gpu: T,
    cpu: T,
    to_disk: T,
    from_disk: T,
}

impl<T> ThreadSpecific<T> {
    pub fn get(&self, thread: PrimaryThread) -> &T {
        match thread {
            PrimaryThread::MemoryManagement => &self.memory_management,
            PrimaryThread::Gpu => &self.gpu,
            PrimaryThread::Cpu => &self.cpu,
            PrimaryThread::FromDisk => &self.from_disk,
            PrimaryThread::ToDisk => &self.to_disk,
        }
    }

    pub fn get_mut(&mut self, thread: PrimaryThread) -> &mut T {
        match thread {
            PrimaryThread::MemoryManagement => &mut self.memory_management,
            PrimaryThread::Gpu => &mut self.gpu,
            PrimaryThread::Cpu => &mut self.cpu,
            PrimaryThread::FromDisk => &mut self.from_disk,
            PrimaryThread::ToDisk => &mut self.to_disk,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (PrimaryThread, &T)> {
        [
            (PrimaryThread::MemoryManagement, &self.memory_management),
            (PrimaryThread::Gpu, &self.gpu),
            (PrimaryThread::Cpu, &self.cpu),
            (PrimaryThread::ToDisk, &self.to_disk),
            (PrimaryThread::FromDisk, &self.from_disk),
        ]
        .into_iter()
    }

    pub fn new(mut f: impl FnMut() -> T) -> Self {
        ThreadSpecific {
            memory_management: f(),
            gpu: f(),
            cpu: f(),
            to_disk: f(),
            from_disk: f(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stream {
    ToGpu,
    FromGpu,
    Gpu,
    GpuMemory,
}

impl From<Stream> for zkpoly_runtime::instructions::Stream {
    fn from(value: Stream) -> Self {
        use Stream::*;
        match value {
            ToGpu => Self::ToGpu,
            FromGpu => Self::FromGpu,
            Gpu => Self::Gpu,
            GpuMemory => Self::GpuMemory,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamSpecific<T> {
    to_gpu: T,
    from_gpu: T,
    gpu: T,
    gpu_memory: T,
}

impl<T> StreamSpecific<T> {
    pub fn get(&self, stream: Stream) -> &T {
        match stream {
            Stream::ToGpu => &self.to_gpu,
            Stream::FromGpu => &self.from_gpu,
            Stream::Gpu => &self.gpu,
            Stream::GpuMemory => &self.gpu_memory,
        }
    }

    pub fn get_mut(&mut self, stream: Stream) -> &mut T {
        match stream {
            Stream::ToGpu => &mut self.to_gpu,
            Stream::FromGpu => &mut self.from_gpu,
            Stream::Gpu => &mut self.gpu,
            Stream::GpuMemory => &mut self.gpu_memory,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Stream, &T)> {
        [
            (Stream::ToGpu, &self.to_gpu),
            (Stream::FromGpu, &self.from_gpu),
            (Stream::Gpu, &self.gpu),
            (Stream::GpuMemory, &self.gpu_memory),
        ]
        .into_iter()
    }

    pub fn new(mut f: impl FnMut() -> T) -> Self
    where
        T: Clone,
    {
        StreamSpecific {
            to_gpu: f(),
            from_gpu: f(),
            gpu: f(),
            gpu_memory: f(),
        }
    }
}

impl PrimaryThread {
    pub fn for_track(track: Track) -> Self {
        match track {
            Track::CoProcess => PrimaryThread::Gpu,
            Track::Gpu(_) => PrimaryThread::Gpu,
            Track::MemoryManagement => PrimaryThread::MemoryManagement,
            Track::ToGpu => PrimaryThread::MemoryManagement,
            Track::FromGpu => PrimaryThread::MemoryManagement,
            Track::GpuMemory(_) => PrimaryThread::MemoryManagement,
            Track::Cpu => PrimaryThread::Cpu,
            Track::FromDisk => PrimaryThread::FromDisk,
            Track::ToDisk => PrimaryThread::ToDisk,
        }
    }
}

impl Stream {
    pub fn of_track(track: Track) -> Option<Self> {
        match track {
            Track::Gpu(_) => Some(Stream::Gpu),
            Track::ToGpu => Some(Stream::ToGpu),
            Track::FromGpu => Some(Stream::FromGpu),
            Track::GpuMemory(_) => Some(Stream::GpuMemory),
            _ => None,
        }
    }
}

fn lower_instruction<'s, Rt: RuntimeType>(
    t3idx: super::InstructionIndex,
    inst: &super::Instruction<'s>,
    _thread: ThreadId,
    track: Track,
    reg_id2var_id: &impl Fn(super::RegisterId) -> VariableId,
    stream2variable_id: &StreamSpecific<VariableId>,
    t3chunk: &super::Chunk<'s, Rt>,
    generated_functions: &GeneratedFunctions,
    emit_inst: &mut impl FnMut(Instruction),
) {
    let mut emit = |inode: InstructionNode| {
        let track = match &inode {
            InstructionNode::MoveRegister { .. } => Track::Cpu,
            InstructionNode::CopyRegister { .. } => Track::Cpu,
            _ => track,
        };

        emit_inst(Instruction::new(
            inode,
            track.into(),
            Stream::of_track(track).map(|x| x.into()),
        ))
    };

    match &inst.node {
        super::InstructionNode::Type2 {
            ids, temp, vertex, ..
        } => {
            ids.iter()
                .filter_map(|(id, inplace)| Some((*id, inplace.as_ref()?)))
                .for_each(|(id, &inplace)| {
                    emit(InstructionNode::MoveRegister {
                        src: reg_id2var_id(inplace),
                        dst: reg_id2var_id(id),
                    })
                });
            let mut vertex = vertex.clone();
            vertex.uses_mut().for_each(|u| {
                if let Some((id, _)) = ids
                    .iter()
                    .find(|(_, inplace)| inplace.is_some_and(|i| i == *u))
                {
                    *u = *id;
                }
            });
            let ids: Vec<_> = ids.iter().map(|x| x.0).collect();
            match &vertex {
                VertexNode::Constant(constant_id) => {
                    assert!(ids.len() == 1);
                    emit(InstructionNode::LoadConstant {
                        src: *constant_id,
                        dst: reg_id2var_id(ids[0]),
                    })
                }
                VertexNode::Blind(id, start_pos, end_pos) => {
                    assert!(ids.len() == 1);
                    assert!(ids[0] == *id);
                    let dst = reg_id2var_id(*id);
                    emit(InstructionNode::Blind {
                        dst,
                        start: *start_pos as usize,
                        end: *end_pos as usize,
                    });
                }
                VertexNode::Entry(idx) => {
                    assert!(ids.len() == 1);
                    emit(InstructionNode::LoadInput {
                        src: *idx,
                        dst: reg_id2var_id(ids[0]),
                    })
                }
                VertexNode::Return(id) => emit(InstructionNode::Return(reg_id2var_id(*id))),
                VertexNode::IndexPoly(operand, idx) => {
                    let operand = reg_id2var_id(*operand);
                    let target = reg_id2var_id(ids[0]);
                    let stream = Stream::of_track(track).map(|t| stream2variable_id.get(t).clone());
                    emit(InstructionNode::GetScalarFromArray {
                        src: operand,
                        dst: target,
                        idx: *idx as usize,
                        stream,
                    })
                }
                VertexNode::AssertEq(src, truth, msg) => {
                    assert!(ids.len() == 1);
                    let src = reg_id2var_id(*src);
                    let truth = reg_id2var_id(*truth);
                    emit(InstructionNode::AssertEq {
                        value: src,
                        expected: truth,
                        msg: msg.clone(),
                    });
                    emit(InstructionNode::CopyRegister {
                        src,
                        dst: reg_id2var_id(ids[0]),
                    })
                }
                VertexNode::Print(id, s) => {
                    assert!(ids.len() == 1);
                    emit(InstructionNode::Print(reg_id2var_id(*id), s.clone()));
                    emit(InstructionNode::CopyRegister {
                        src: reg_id2var_id(*id),
                        dst: reg_id2var_id(ids[0]),
                    })
                }
                _ => {
                    emit_func::emit_func(
                        t3idx,
                        &ids,
                        temp,
                        track,
                        &vertex,
                        reg_id2var_id,
                        stream2variable_id,
                        generated_functions.at(t3idx),
                        t3chunk,
                        &mut emit,
                    );
                }
            }
        }
        super::InstructionNode::Malloc { id, addr, device } => {
            let var_id = reg_id2var_id(*id);

            emit(InstructionNode::Allocate {
                device: DeviceType::from(device.clone()),
                typ: t3chunk.register_types[*id].erase_p(),
                id: var_id,
                alloc_method: addr.clone(),
            });
        }
        super::InstructionNode::Free { id, variant, .. } => emit(InstructionNode::Deallocate {
            id: reg_id2var_id(*id),
            alloc_method: *variant,
        }),
        super::InstructionNode::Transfer { id, from } => emit(InstructionNode::Transfer {
            src_device: DeviceType::from(t3chunk.register_devices[from]),
            dst_device: DeviceType::from(t3chunk.register_devices[id]),
            stream: Stream::of_track(track).map(|s| *stream2variable_id.get(s)),
            src_id: reg_id2var_id(*from),
            dst_id: reg_id2var_id(*id),
        }),
        super::InstructionNode::TransferToDefed { id, to, from } => {
            emit(InstructionNode::Transfer {
                src_device: DeviceType::from(t3chunk.register_devices[from]),
                dst_device: DeviceType::from(t3chunk.register_devices[to]),
                stream: Stream::of_track(track).map(|s| *stream2variable_id.get(s)),
                src_id: reg_id2var_id(*from),
                dst_id: reg_id2var_id(*to),
            });
            emit(InstructionNode::MoveRegister {
                src: reg_id2var_id(*to),
                dst: reg_id2var_id(*id),
            })
        }
        super::InstructionNode::StackFree { id } => emit(InstructionNode::RemoveRegister {
            id: reg_id2var_id(*id),
        }),
        super::InstructionNode::Tuple { id, oprands } => {
            let dst = reg_id2var_id(*id);
            let vars = oprands.iter().map(|&id| reg_id2var_id(id)).collect();
            emit(InstructionNode::AssembleTuple { vars, dst });
        }
        super::InstructionNode::SetPolyMeta {
            id,
            from,
            offset,
            len,
        } => emit(InstructionNode::SetSliceMeta {
            src: reg_id2var_id(*from),
            dst: reg_id2var_id(*id),
            offset: *offset as usize,
            len: *len as usize,
        }),
        super::InstructionNode::FillPoly { id, operand, .. } => {
            let (f_id, _) = generated_functions.at(t3idx);
            let device = t3chunk.register_devices[operand];
            let dst = reg_id2var_id(*operand);
            if device == super::Device::Cpu {
                emit(InstructionNode::FuncCall {
                    func_id: f_id,
                    arg_mut: vec![dst],
                    arg: vec![],
                });
            } else {
                let stream = Stream::of_track(track).map(|t| stream2variable_id.get(t).clone());
                emit(InstructionNode::FuncCall {
                    func_id: f_id,
                    arg_mut: vec![dst],
                    arg: vec![stream.unwrap()],
                });
            }

            emit(InstructionNode::MoveRegister {
                src: reg_id2var_id(*operand),
                dst: reg_id2var_id(*id),
            })
        }
        super::InstructionNode::SliceBuffer {
            id,
            operand,
            offset,
            len,
        } => emit(InstructionNode::SliceBuffer {
            src: reg_id2var_id(*operand),
            dst: reg_id2var_id(*id),
            offset: *offset,
            len: *len,
        }),
    };
}

/// A CPU thread `thread` waits for instruction `depended_t3idx` at `depended_track`
fn lower_cpu_waits_any(
    depended_t3idx: super::InstructionIndex,
    thread: ThreadId,
    depended_track: Track,
    stream2variable_id: &StreamSpecific<VariableId>,
    event_table: &mut EventTypeTable,
    chunk: &mut MultithreadChunk,
    instruct2cpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
    instruct2gpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
) {
    match Stream::of_track(depended_track) {
        Some(depended_stream) => {
            let event_id = instruct2gpu_event
                .get(&depended_t3idx)
                .copied()
                .unwrap_or_else(|| {
                    let event_id = event_table.push(EventType::new_gpu(0)); // one card for now
                    instruct2gpu_event.insert(depended_t3idx, event_id);

                    // Record/Wait are not considered to be on GPU
                    chunk.append_at(
                        depended_t3idx,
                        Instruction::new(
                            InstructionNode::Record {
                                stream: Some(*stream2variable_id.get(depended_stream)),
                                event: event_id,
                            },
                            Track::Cpu.into(),
                            None,
                        ),
                    );
                    event_id
                });

            chunk.emit(
                thread,
                Instruction::new(
                    InstructionNode::Wait {
                        slave: DeviceType::CPU,
                        stream: None,
                        event: event_id,
                    },
                    Track::Cpu.into(),
                    None,
                ),
            );
        }
        None => {
            let event_id = instruct2cpu_event
                .get(&depended_t3idx)
                .copied()
                .unwrap_or_else(|| {
                    let event_id = event_table.push(EventType::new_thread());
                    instruct2cpu_event.insert(depended_t3idx, event_id);

                    chunk.append_at(
                        depended_t3idx,
                        Instruction::new(
                            InstructionNode::Record {
                                stream: None,
                                event: event_id,
                            },
                            Track::Cpu.into(),
                            None,
                        ),
                    );
                    event_id
                });

            chunk.emit(
                thread,
                Instruction::new(
                    InstructionNode::Wait {
                        slave: DeviceType::CPU,
                        stream: None,
                        event: event_id,
                    },
                    Track::Cpu.into(),
                    None,
                ),
            );
        }
    }
}

/// A GPU stream `stream` waits for instruction `depended_t3idx` at `depended_track`, which must be a GPU stream.
/// The event wait function is issued at the thread `thread`.
fn lower_gpu_waits_gpu(
    depended_t3idx: super::InstructionIndex,
    launch_thread: ThreadId,
    stream: Stream,
    depended_track: Track,
    stream2variable_id: &StreamSpecific<VariableId>,
    event_table: &mut EventTypeTable,
    chunk: &mut MultithreadChunk,
    instruct2gpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
) {
    let event_id = instruct2gpu_event
        .get(&depended_t3idx)
        .copied()
        .unwrap_or_else(|| {
            let event_id = event_table.push(EventType::new_gpu(0));
            instruct2gpu_event.insert(depended_t3idx, event_id);

            chunk.append_at(
                depended_t3idx,
                Instruction::new(
                    InstructionNode::Record {
                        stream: Some(
                            *stream2variable_id.get(Stream::of_track(depended_track).unwrap()),
                        ),
                        event: event_id,
                    },
                    Track::Cpu.into(),
                    None,
                ),
            );
            event_id
        });

    chunk.emit(
        launch_thread,
        Instruction::new(
            InstructionNode::Wait {
                slave: DeviceType::GPU { device_id: 0 },
                stream: Some(*stream2variable_id.get(stream)),
                event: event_id,
            },
            Track::Cpu.into(),
            None,
        ),
    );
}

fn lower_dependency(
    depended_t3idx: super::InstructionIndex,
    thread: PrimaryThread,
    track: Track,
    depended_track: Track,
    stream2variable_id: &StreamSpecific<VariableId>,
    event_table: &mut EventTypeTable,
    chunk: &mut MultithreadChunk,
    instruct2cpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
    instruct2gpu_event: &mut BTreeMap<super::InstructionIndex, EventId>,
) {
    let thread = *chunk.primary_thread_id.get(thread);
    match (Stream::of_track(track), Stream::of_track(depended_track)) {
        (None, _) => lower_cpu_waits_any(
            depended_t3idx,
            thread,
            depended_track,
            stream2variable_id,
            event_table,
            chunk,
            instruct2cpu_event,
            instruct2gpu_event,
        ),
        (Some(stream), Some(..)) => lower_gpu_waits_gpu(
            depended_t3idx,
            thread,
            stream,
            depended_track,
            stream2variable_id,
            event_table,
            chunk,
            instruct2gpu_event,
        ),
        (a, b) => panic!(
            "cannot handle {:?}@{:?} waits {:?}@{:?} here",
            track, a, depended_track, b
        ),
    }
}

#[derive(Clone)]
pub struct Chunk<Rt: RuntimeType> {
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) f_table: FunctionTable<Rt>,
    pub(crate) event_table: EventTypeTable,
    pub(crate) n_variables: usize,
    pub(crate) n_threads: usize,
    pub(crate) lbss: LogBlockSizes,
}

impl<Rt: RuntimeType> Chunk<Rt> {
    // currently, we do this in runtime, so this function is commented out
    // /// Adjusts the GPU device IDs by a given offset.
    // /// Relative GPU IDs in instructions will be converted to absolute IDs.
    // pub fn adjust_gpu_device_ids(mut self, gpu_mapping: i32) -> Self {
    //     // // Adjust the GPU device IDs in the chunk's instructions
    //     // for instruction in self.instructions.iter_mut() {
    //     //     Self::_adjust_instruction_gpu_ids(instruction, gpu_offset);
    //     // }
    //     // Adjust the GPU device IDs in the event table
    //     for event in self.event_table.iter_mut() {
    //         if let EventType::GpuEvent(device_id) = event {
    //             *device_id += gpu_offset;
    //         }
    //     }
    //     self
    // }

    // /// Helper function to adjust GPU device IDs for a single instruction.
    // fn _adjust_instruction_gpu_ids(instruction: &mut Instruction, gpu_offset: i32) {
    //     match instruction {
    //         Instruction::Allocate { device, .. } => {
    //             if let DeviceType::GPU { device_id } = device {
    //                 *device_id += gpu_offset;
    //             }
    //         }
    //         Instruction::Transfer {
    //             src_device,
    //             dst_device,
    //             ..
    //         } => {
    //             if let DeviceType::GPU { device_id } = src_device {
    //                 *device_id += gpu_offset;
    //             }
    //             if let DeviceType::GPU { device_id } = dst_device {
    //                 *device_id += gpu_offset;
    //             }
    //         }
    //         Instruction::Wait { slave, .. } => {
    //             if let DeviceType::GPU { device_id } = slave {
    //                 *device_id += gpu_offset;
    //             }
    //         }
    //         Instruction::Fork {
    //             instructions: nested_instructions,
    //             ..
    //         } => {
    //             for nested_instr in nested_instructions.iter_mut() {
    //                 Self::_adjust_instruction_gpu_ids(nested_instr, gpu_offset);
    //             }
    //         }
    //         // Other instructions do not have top-level DeviceType fields that represent
    //         // assignable GPU resources in the same way, or their devices are implicitly handled.
    //         _ => {}
    //     }
    // }
}

pub fn emit_multithread_instructions<'s, Rt: RuntimeType>(
    track_tasks: &TrackTasks,
    mut t3chunk: super::Chunk<'s, Rt>,
    t2uf_table: type2::user_function::Table<Rt>,
    libs: &mut Libs,
    kernel_dir: PathBuf,
) -> (
    MultithreadChunk,
    FunctionTable<Rt>,
    EventTypeTable,
    StreamSpecific<VariableId>,
    IdAllocator<VariableId>,
    LogBlockSizes,
) {
    let mut event_table = EventTypeTable::new();
    let mut f_table = FunctionTable::<Rt>::new();
    let (mut variable_id_allcoator, reg_id2var_id) = t3chunk.take_reg_id_allocator().decompose();

    std::fs::create_dir_all(&kernel_dir).expect("failed to create directory for kernels");
    let generated_functions = kernel_gen::get_function_id(
        &mut f_table,
        &t3chunk,
        t2uf_table,
        &reg_id2var_id,
        libs,
        kernel_dir,
    );

    let stream2variable_id = StreamSpecific::new(|| variable_id_allcoator.alloc());

    let mut chunk = MultithreadChunk::new();

    let mut instruct2cpu_event = BTreeMap::new();
    let mut instruct2gpu_event = BTreeMap::new();

    for (t3idx, t3inst) in t3chunk.iter_instructions() {
        let track = track_tasks.inst_track[&t3idx];
        let pthread = PrimaryThread::for_track(track);

        if track.is_gpu()
            && track_tasks.inst_depend[&t3idx]
                .iter()
                .any(|&depended_t3idx| track_tasks.inst_track[&depended_t3idx].is_cpu())
        {
            // GPU waits for some CPU tasks
            let depended_threads: BTreeSet<_> = track_tasks.inst_depend[&t3idx]
                .iter()
                .copied()
                .filter(|&t3idx| track_tasks.inst_track[&t3idx].is_cpu())
                .map(|depended_t3idx| chunk.thread_of(depended_t3idx))
                .collect();

            if depended_threads.len() > 1 {
                // A GPU task waits for CPU tasks on multiple threads.
                // In this case, we need a new CPU thread that first waits for those CPU tasks,
                // then the CPU thread launches the GPU task.
                let aux_thread = chunk.new_auxilary_thread();

                for &depended_t3idx in track_tasks.inst_depend[&t3idx].iter() {
                    let depended_track = track_tasks.inst_track[&depended_t3idx];
                    lower_cpu_waits_any(
                        depended_t3idx,
                        aux_thread,
                        depended_track,
                        &stream2variable_id,
                        &mut event_table,
                        &mut chunk,
                        &mut instruct2cpu_event,
                        &mut instruct2gpu_event,
                    );
                }

                lower_instruction(
                    t3idx,
                    t3inst,
                    aux_thread,
                    track,
                    &reg_id2var_id,
                    &stream2variable_id,
                    &t3chunk,
                    &generated_functions,
                    &mut |inst| chunk.emit_with_idx(t3idx, aux_thread, inst),
                );

                chunk.emit_fork(pthread, aux_thread);
            } else {
                // A GPU task waits for CPU tasks on a single thread.
                // In this case, we can just launch the GPU task on the same thread.
                let thread = depended_threads.first().copied().unwrap();
                let stream = Stream::of_track(track).unwrap();

                // First waits for GPU tasks
                for &depended_t3idx in track_tasks.inst_depend[&t3idx].iter() {
                    let depended_track = track_tasks.inst_track[&depended_t3idx];
                    if Stream::of_track(depended_track).is_some() {
                        lower_gpu_waits_gpu(
                            depended_t3idx,
                            thread,
                            stream,
                            depended_track,
                            &stream2variable_id,
                            &mut event_table,
                            &mut chunk,
                            &mut instruct2gpu_event,
                        );
                    }
                }

                lower_instruction(
                    t3idx,
                    t3inst,
                    thread,
                    track,
                    &reg_id2var_id,
                    &stream2variable_id,
                    &t3chunk,
                    &generated_functions,
                    &mut |inst| chunk.emit_with_idx(t3idx, thread, inst),
                );
            }
        } else {
            // GPU waits for only GPU tasks, or CPU waits.
            for &depended_t3idx in track_tasks.inst_depend[&t3idx].iter() {
                let depended_track = track_tasks.inst_track[&depended_t3idx];
                lower_dependency(
                    depended_t3idx,
                    pthread,
                    track,
                    depended_track,
                    &stream2variable_id,
                    &mut event_table,
                    &mut chunk,
                    &mut instruct2cpu_event,
                    &mut instruct2gpu_event,
                );
            }

            lower_instruction(
                t3idx,
                t3inst,
                *chunk.primary_thread_id.get(pthread),
                track,
                &reg_id2var_id,
                &stream2variable_id,
                &t3chunk,
                &generated_functions,
                &mut |inst| chunk.emit_primary_with_idx(t3idx, pthread, inst),
            );
        }
    }

    (
        chunk,
        f_table,
        event_table,
        stream2variable_id,
        variable_id_allcoator,
        t3chunk.lbss,
    )
}

pub fn lower<'s, Rt: RuntimeType>(
    mut mt_chunk: MultithreadChunk,
    f_table: FunctionTable<Rt>,
    event_table: EventTypeTable,
    stream2variable_id: StreamSpecific<VariableId>,
    variable_id_allocator: IdAllocator<VariableId>,
    lbss: LogBlockSizes,
) -> Chunk<Rt> {
    let mut instructions = Vec::new();

    // Create streams
    stream2variable_id.iter().for_each(|(stream, &var_id)| {
        instructions.push(Instruction::new(
            InstructionNode::Allocate {
                device: DeviceType::GPU { device_id: 0 },
                typ: Typ::Stream,
                id: var_id,
                alloc_method: AllocMethod::default(),
            },
            Track::Cpu.into(),
            Some(stream.into()),
        ))
    });

    // Fillback auxiliary threads
    mt_chunk.fillback_auxiliary_instructions();

    // Fork primary threads
    mt_chunk
        .primary_thread_id
        .iter()
        .filter(|(pthread, _)| *pthread != PrimaryThread::main())
        .for_each(|(_pthread, &thread)| {
            instructions.push(Instruction::new(
                InstructionNode::Fork {
                    new_thread: thread,
                    instructions: mt_chunk.thread_instructions(thread).cloned().collect(),
                },
                Track::Cpu.into(),
                None,
            ))
        });

    // Emit main thread instructions
    instructions.extend(
        mt_chunk
            .primary_thread_instructions(PrimaryThread::main())
            .cloned(),
    );

    Chunk {
        instructions,
        f_table,
        event_table,
        n_threads: mt_chunk.threads.len(),
        n_variables: variable_id_allocator.n_allocated(),
        lbss,
    }
}

pub mod pretty_print;
pub mod serialization;
