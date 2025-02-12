use std::collections::BTreeMap;

use super::track_splitting::{split, TrackTasks};
use super::{kernel_generation, Track};
use zkpoly_common::define_usize_id;
use zkpoly_common::heap::{Heap, IdAllocator};
use zkpoly_common::typ::Typ;
use zkpoly_runtime::args::{RuntimeType, VariableId};
use zkpoly_runtime::devices::{DeviceType, Event, EventId, EventTable, ThreadId};
use zkpoly_runtime::functions::{FunctionId, FunctionTable};
use zkpoly_runtime::instructions::Instruction;

#[derive(Debug, Clone)]
struct Cell {
    inst: Instruction,
    tail: Vec<Instruction>,
}

impl Cell {
    pub fn new(inst: Instruction) -> Self {
        Cell { inst, tail: vec![] }
    }
}

define_usize_id!(InstructionId);

#[derive(Debug, Clone)]
struct MultithreadChunk {
    instructions: Heap<InstructionId, Cell>,
    threads: Heap<ThreadId, Vec<InstructionId>>,
    primary_thread_id: ThreadSpecific<ThreadId>,
    t3idx2id: BTreeMap<super::InstructionIndex, InstructionId>,
    forks: BTreeMap<ThreadId, InstructionId>,
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
        let inst_id = self.instructions.push(Cell::new(instruction));
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
        let inst_id: InstructionId = self.instructions.push(Cell::new(instruction));
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
        self.instructions[self.t3idx2id[&t3idx]].tail.push(tail);
    }

    pub fn new_auxilary_thread(&mut self) -> ThreadId {
        self.threads.push(vec![])
    }

    pub fn emit_fork(&mut self, pthread: PrimaryThread, fork_to: ThreadId) {
        let fork_inst_id = self.emit_primary(
            pthread,
            Instruction::Fork {
                new_thread: fork_to,
                instructions: Vec::new(),
            },
        );
        self.forks.insert(fork_to, fork_inst_id);
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
        for (&fork_to, &fork_inst_id) in self.forks.iter() {
            let aux_inst = self.thread_instructions(fork_to).cloned().collect();

            let fork_inst = &mut self.instructions[fork_inst_id];
            if let Instruction::Fork { instructions, .. } = &mut fork_inst.inst {
                *instructions = aux_inst;
            } else {
                panic!("expect to fillback to a fork instruction")
            }
        }
    }
}

impl From<super::Device> for DeviceType {
    fn from(value: super::Device) -> Self {
        match value {
            super::Device::Cpu => DeviceType::CPU,
            super::Device::Gpu => DeviceType::GPU { device_id: 0 },
            super::Device::Stack => DeviceType::CPU,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PrimaryThread {
    MemoryManagement,
    Gpu,
    Cpu,
}

impl PrimaryThread {
    pub fn main() -> Self {
        PrimaryThread::MemoryManagement
    }
}

#[derive(Debug, Clone)]
struct ThreadSpecific<T> {
    memory_management: T,
    gpu: T,
    cpu: T,
}

impl<T> ThreadSpecific<T> {
    pub fn get(&self, thread: PrimaryThread) -> &T {
        match thread {
            PrimaryThread::MemoryManagement => &self.memory_management,
            PrimaryThread::Gpu => &self.gpu,
            PrimaryThread::Cpu => &self.cpu,
        }
    }

    pub fn get_mut(&mut self, thread: PrimaryThread) -> &mut T {
        match thread {
            PrimaryThread::MemoryManagement => &mut self.memory_management,
            PrimaryThread::Gpu => &mut self.gpu,
            PrimaryThread::Cpu => &mut self.cpu,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (PrimaryThread, &T)> {
        [
            (PrimaryThread::MemoryManagement, &self.memory_management),
            (PrimaryThread::Gpu, &self.gpu),
            (PrimaryThread::Cpu, &self.cpu),
        ]
        .into_iter()
    }

    pub fn new(mut f: impl FnMut() -> T) -> Self {
        ThreadSpecific {
            memory_management: f(),
            gpu: f(),
            cpu: f(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stream {
    ToGpu,
    FromGpu,
    Gpu,
    GpuMemory,
}

#[derive(Debug, Clone)]
struct StreamSpecific<T> {
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
            Track::Gpu => PrimaryThread::Gpu,
            Track::MemoryManagement => PrimaryThread::MemoryManagement,
            Track::ToGpu => PrimaryThread::MemoryManagement,
            Track::FromGpu => PrimaryThread::MemoryManagement,
            Track::GpuMemory => PrimaryThread::MemoryManagement,
            Track::Cpu => PrimaryThread::Cpu,
        }
    }
}

impl Stream {
    pub fn of_track(track: Track) -> Option<Self> {
        match track {
            Track::Gpu => Some(Stream::Gpu),
            Track::ToGpu => Some(Stream::ToGpu),
            Track::FromGpu => Some(Stream::FromGpu),
            Track::GpuMemory => Some(Stream::GpuMemory),
            _ => None,
        }
    }
}

fn allocate_stream(variable_id_allocator: &mut IdAllocator<VariableId>) -> Instruction {
    Instruction::Allocate {
        device: DeviceType::CPU,
        typ: Typ::Stream,
        id: variable_id_allocator.alloc(),
        offset: None,
    }
}

fn lower_typ<Rt: RuntimeType>(t3typ: &super::type2::Typ<Rt>) -> Typ {
    use super::type2::Typ::*;
    match t3typ {
        Poly { deg, .. } => Typ::ScalarArray { len: *deg as usize },
        PointBase { log_n } => Typ::PointBase {
            len: 2usize.pow(*log_n),
        },
        Scalar => Typ::Scalar,
        Transcript => Typ::Transcript,
        Point => Typ::Point,
        Rng => Typ::Rng,
        Tuple(children) => Typ::Tuple(children.iter().map(|t| lower_typ(t)).collect()),
        Array(elem, len) => Typ::Array(Box::new(lower_typ(elem)), *len as usize),
        Any(type_id, size) => Typ::Any(*type_id, *size),
        _Phantom(..) => unreachable!(),
    }
}

fn lower_instruction<'s, Rt: RuntimeType>(
    t3idx: super::InstructionIndex,
    inst: &super::Instruction<'s>,
    thread: ThreadId,
    track: Track,
    reg_id2var_id: &impl Fn(super::RegisterId) -> VariableId,
    stream2variable_id: &StreamSpecific<VariableId>,
    t3chunk: &super::Chunk<'s, Rt>,
    f_table: &mut FunctionTable<Rt>,
    emit: &mut impl FnMut(Instruction),
) {
    match &inst.node {
        super::InstructionNode::Type2 { ids, temp, vertex } => {
            kernel_generation::generate(
                ids,
                *temp,
                vertex,
                t3chunk,
                thread,
                reg_id2var_id,
                f_table,
                emit,
            );
        }
        super::InstructionNode::GpuMalloc { id, addr } => {
            let var_id = reg_id2var_id(*id);

            let physical_addr = t3chunk.gpu_addr_mapping[*addr].0 .0;

            emit(Instruction::Allocate {
                device: DeviceType::from(t3chunk.register_devices[id]),
                typ: lower_typ(&t3chunk.register_types[id]),
                id: var_id,
                offset: Some(physical_addr as usize),
            });
        }
        super::InstructionNode::GpuFree { id } => {}
        super::InstructionNode::CpuMalloc { id, size } => {
            let var_id = reg_id2var_id(*id);

            emit(Instruction::Allocate {
                device: DeviceType::CPU,
                typ: lower_typ(&t3chunk.register_types[id]),
                id: var_id,
                offset: None,
            });
        }
        super::InstructionNode::CpuFree { id } => emit(Instruction::Deallocate {
            id: reg_id2var_id(*id),
        }),
        super::InstructionNode::Transfer { id, from, rot } => emit(Instruction::Transfer {
            src_device: DeviceType::from(t3chunk.register_devices[from]),
            dst_device: DeviceType::from(t3chunk.register_devices[id]),
            stream: Stream::of_track(track).map(|s| *stream2variable_id.get(s)),
            src_id: reg_id2var_id(*from),
            dst_id: reg_id2var_id(*id),
        }),
        super::InstructionNode::StackFree { id } => emit(Instruction::Deallocate {
            id: reg_id2var_id(*id),
        }),
        super::InstructionNode::Tuple { id, oprands } => todo!("Tuple Generation"),
        super::InstructionNode::Move { id, from } => unimplemented!(),
        super::InstructionNode::RotateAndSlice {
            id,
            operand,
            rot,
            slice,
        } => unimplemented!(),
    };
}

/// A CPU thread `thread` waits for instruction `depended_t3idx` at `depended_track`
fn lower_cpu_waits_any(
    depended_t3idx: super::InstructionIndex,
    thread: ThreadId,
    depended_track: Track,
    stream2variable_id: &StreamSpecific<VariableId>,
    event_table: &mut EventTable,
    chunk: &mut MultithreadChunk,
) {
    match Stream::of_track(depended_track) {
        Some(depended_stream) => {
            let event_id = event_table.push(Event::new_gpu());

            chunk.append_at(
                depended_t3idx,
                Instruction::Record {
                    stream: Some(*stream2variable_id.get(depended_stream)),
                    event: event_id,
                },
            );

            chunk.emit(
                thread,
                Instruction::Wait {
                    slave: DeviceType::CPU,
                    stream: None,
                    event: event_id,
                },
            );
        }
        None => {
            let event_id = event_table.push(Event::new_thread());

            chunk.append_at(
                depended_t3idx,
                Instruction::Record {
                    stream: None,
                    event: event_id,
                },
            );

            chunk.emit(
                thread,
                Instruction::Wait {
                    slave: DeviceType::CPU,
                    stream: None,
                    event: event_id,
                },
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
    event_table: &mut EventTable,
    chunk: &mut MultithreadChunk,
) {
    let event_id = event_table.push(Event::new_gpu());

    chunk.append_at(
        depended_t3idx,
        Instruction::Record {
            stream: Some(*stream2variable_id.get(Stream::of_track(depended_track).unwrap())),
            event: event_id,
        },
    );

    chunk.emit(
        launch_thread,
        Instruction::Wait {
            slave: DeviceType::GPU { device_id: 0 },
            stream: Some(*stream2variable_id.get(stream)),
            event: event_id,
        },
    );
}

fn lower_dependency(
    depended_t3idx: super::InstructionIndex,
    thread: PrimaryThread,
    track: Track,
    depended_track: Track,
    stream2variable_id: &StreamSpecific<VariableId>,
    event_table: &mut EventTable,
    chunk: &mut MultithreadChunk,
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
        ),
        (Some(stream), Some(..)) => lower_gpu_waits_gpu(
            depended_t3idx,
            thread,
            stream,
            depended_track,
            stream2variable_id,
            event_table,
            chunk,
        ),
        _ => unreachable!(),
    }
}

pub struct Chunk<Rt: RuntimeType> {
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) f_table: FunctionTable<Rt>,
    pub(crate) event_table: EventTable,
}

fn emit_multithread_instructions<'s, Rt: RuntimeType>(
    mut t3chunk: super::Chunk<'s, Rt>,
) -> (
    MultithreadChunk,
    FunctionTable<Rt>,
    EventTable,
    StreamSpecific<VariableId>,
) {
    let track_tasks = split(&t3chunk);
    let mut event_table = EventTable::new();
    let mut f_table = FunctionTable::<Rt>::new();
    let (mut variable_id_allcoator, reg_id2var_id) = t3chunk.take_reg_id_allocator().decompose();

    let stream2variable_id = StreamSpecific::new(|| variable_id_allcoator.alloc());

    let mut chunk = MultithreadChunk::new();

    for (t3idx, t3inst) in t3chunk.iter_instructions() {
        let track = track_tasks.inst_track[&t3idx];
        let pthread = PrimaryThread::for_track(track);

        if Stream::of_track(track).is_none()
            && track_tasks.inst_depend[&t3idx]
                .iter()
                .any(|&depended_t3idx| {
                    Stream::of_track(track_tasks.inst_track[&depended_t3idx]).is_none()
                })
        {
            // A CPU track must wait for another CPU track
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
                &mut f_table,
                &mut |inst| chunk.emit_with_idx(t3idx, aux_thread, inst),
            );

            chunk.emit_fork(pthread, aux_thread);
        } else {
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
                &mut f_table,
                &mut |inst| chunk.emit_primary_with_idx(t3idx, pthread, inst),
            );
        }
    }

    (chunk, f_table, event_table, stream2variable_id)
}

pub fn lower<'s, Rt: RuntimeType>(mut t3chunk: super::Chunk<'s, Rt>) -> Chunk<Rt> {
    let (mut mt_chunk, f_table, event_table, stream2variable_id) =
        emit_multithread_instructions(t3chunk);

    let mut instructions = Vec::new();

    // Create streams
    stream2variable_id.iter().for_each(|(stream, &var_id)| {
        instructions.push(Instruction::Allocate {
            device: DeviceType::CPU,
            typ: Typ::Stream,
            id: var_id,
            offset: None,
        })
    });

    // Fork primary threads
    mt_chunk
        .primary_thread_id
        .iter()
        .filter(|(pthread, _)| *pthread != PrimaryThread::main())
        .for_each(|(pthread, &thread)| {
            instructions.push(Instruction::Fork {
                new_thread: thread,
                instructions: mt_chunk.thread_instructions(thread).cloned().collect(),
            })
        });

    // Fillback auxiliary threads
    mt_chunk.fillback_auxiliary_instructions();

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
    }
}
