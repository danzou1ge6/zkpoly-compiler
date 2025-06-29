use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};
use zkpoly_common::bijection::Bijection;

use zkpoly_cuda_api::stream::{CudaEventRaw, CudaStream};

use crate::{
    devices::ThreadId,
    functions::FuncMeta,
    instructions::{self, Instruction},
};

type StreamTable = BTreeMap<instructions::Stream, (CudaStream, CudaEventRaw, Instant)>;

#[derive(Clone)]
pub struct Writer {
    sender: crossbeam_channel::Sender<Message>,
    thread: ThreadId,
    streams: Arc<Mutex<StreamTable>>,
    run_begin_time: Instant,
}

enum RuntimeUptime {
    Stream(CudaEventRaw),
    Sync(Duration),
}

pub struct InstructionBeginGuard<'a> {
    inst: Instruction,
    /// Only some when the instruction is a kernel launch instruction
    launch_guard: Option<MutexGuard<'a, StreamTable>>,
    begin_uptime: RuntimeUptime,
}

impl Writer {
    pub fn with_thread_id(self, thread: ThreadId) -> Self {
        Self { thread, ..self }
    }

    fn stream_uptime(
        &self,
        guard: &MutexGuard<StreamTable>,
        stream_number: instructions::Stream,
    ) -> CudaEventRaw {
        let (stream, _stream_begin, _) = guard.get(&stream_number).unwrap();
        let event = CudaEventRaw::new();
        event.record(stream);
        event
    }

    fn sync_uptime(&self) -> Duration {
        let dur = Instant::now().duration_since(self.run_begin_time);
        dur
    }

    pub fn begin_instruction<'a>(&'a self, inst: Instruction) -> InstructionBeginGuard<'a> {
        if let Some(stream_number) = inst.need_timing_stream() {
            let guard = self.streams.lock().unwrap();
            let begin_uptime = self.stream_uptime(&guard, stream_number);

            InstructionBeginGuard {
                inst,
                launch_guard: Some(guard),
                begin_uptime: RuntimeUptime::Stream(begin_uptime),
            }
        } else {
            let begin_uptime = self.sync_uptime();

            InstructionBeginGuard {
                inst,
                launch_guard: None,
                begin_uptime: RuntimeUptime::Sync(begin_uptime),
            }
        }
    }

    pub fn end_instruction(&self, ibg: InstructionBeginGuard<'_>, function_meta: Option<FuncMeta>) {
        if let Some(guard) = ibg.launch_guard {
            let end_uptime = self.stream_uptime(&guard, ibg.inst.unwrap_stream());

            self.sender
                .send(Message {
                    thread: self.thread,
                    node: MessageNode::InstructionExecuted(InstructionExecution {
                        instruction: ibg.inst,
                        start: ibg.begin_uptime,
                        end: RuntimeUptime::Stream(end_uptime),
                        function_meta,
                        thread: self.thread,
                    }),
                })
                .expect("channel closed");

            drop(guard);
        } else {
            let end_uptime = self.sync_uptime();

            self.sender
                .send(Message {
                    thread: self.thread,
                    node: MessageNode::InstructionExecuted(InstructionExecution {
                        instruction: ibg.inst,
                        start: ibg.begin_uptime,
                        end: RuntimeUptime::Sync(end_uptime),
                        function_meta,
                        thread: self.thread,
                    }),
                })
                .expect("channel closed");
        }
    }

    pub fn new_stream(
        &self,
        stream_number: instructions::Stream,
        stream: CudaStream,
        begin_event: CudaEventRaw,
    ) {
        self.streams
            .lock()
            .unwrap()
            .insert(stream_number, (stream, begin_event, Instant::now()));
    }
}

pub struct Logger {
    executed_instructions: Vec<InstructionExecution<RuntimeUptime>>,
}

impl Logger {
    pub fn new() -> Self {
        Self {
            executed_instructions: Vec::new(),
        }
    }

    pub fn spawn(self) -> LoggerHandle {
        let (sender, receiver) = crossbeam_channel::unbounded::<Message>();
        let handle = std::thread::spawn(move || {
            let mut logger = self;
            loop {
                use MessageNode::*;
                match receiver.recv().expect("channel closed").node {
                    Terminate => break,
                    InstructionExecuted(ie) => logger.executed_instructions.push(ie),
                }
            }
            logger
        });
        LoggerHandle {
            writer: Writer {
                sender: sender,
                thread: ThreadId::from(0),
                streams: Arc::new(Mutex::new(BTreeMap::new())),
                run_begin_time: Instant::now(),
            },
            join_handle: handle,
        }
    }
}

pub struct LoggerHandle {
    join_handle: std::thread::JoinHandle<Logger>,
    writer: Writer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Log {
    executed_instructions: Vec<InstructionExecution<Uptime>>,
}

impl LoggerHandle {
    pub fn writer(&self) -> Writer {
        self.writer.clone()
    }

    pub fn join(self) -> Log {
        self.writer
            .sender
            .send(Message {
                thread: ThreadId::from(0),
                node: MessageNode::Terminate,
            })
            .expect("channel closed");

        let logger = self.join_handle.join().expect("thread panicked");
        Log {
            executed_instructions: logger
                .executed_instructions
                .into_iter()
                .map(|ie| {
                    let stream_number = ie.instruction.stream();
                    InstructionExecution {
                        instruction: ie.instruction,
                        start: ie.start.realize(&self.writer, stream_number),
                        end: ie.end.realize(&self.writer, stream_number),
                        function_meta: ie.function_meta,
                        thread: ie.thread,
                    }
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uptime {
    nanos: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionExecution<U> {
    instruction: Instruction,
    start: U,
    end: U,
    function_meta: Option<FuncMeta>,
    thread: ThreadId,
}

impl RuntimeUptime {
    pub fn realize(self, writer: &Writer, stream_number: Option<instructions::Stream>) -> Uptime {
        use RuntimeUptime::*;
        match self {
            Stream(event) => {
                let guard = writer.streams.lock().unwrap();
                let (_, stream_begin, stream_begin_instant) =
                    &guard.get(&stream_number.unwrap()).unwrap();
                let t = stream_begin.elapsed(&event);
                Uptime {
                    nanos: (t * 1e6) as u128
                        + stream_begin_instant
                            .duration_since(writer.run_begin_time)
                            .as_nanos(),
                }
            }
            Sync(dur) => Uptime {
                nanos: dur.as_nanos(),
            },
        }
    }
}

enum MessageNode {
    InstructionExecuted(InstructionExecution<RuntimeUptime>),
    Terminate,
}

struct Message {
    thread: ThreadId,
    node: MessageNode,
}

impl Log {
    pub fn waterfall(&self) -> waterfall::Builder {
        let mut builder = waterfall::Builder::new("Runtime Statistics".to_string());

        let mut track_id_acc: usize = 0;
        let track_ids = self
            .executed_instructions
            .iter()
            .fold(Bijection::new(), |mut acc, ie| {
                let track = ie.instruction.track();
                if acc.get_forward(&track).is_none() {
                    acc.insert(track, track_id_acc);
                    track_id_acc += 1;
                }
                acc
            });

        let mut track_ids_list = track_ids.iter().collect::<Vec<_>>();
        track_ids_list.sort_by_key(|(_, x)| **x);

        track_ids_list
            .into_iter()
            .zip(waterfall::color_loop())
            .for_each(|((track, _), color)| {
                builder.add_category(waterfall::Category::new(
                    format!("{:?}", track),
                    color.to_string(),
                ));
            });

        self.executed_instructions
            .iter()
            .enumerate()
            .for_each(|(idx, ie)| {
                let tid = track_ids.get_forward(&ie.instruction.track()).unwrap();

                use instructions::InstructionNode::*;
                match ie.instruction.node() {
                    Wait { .. } | Record { .. } | Fork { .. } => {}
                    _ => builder.add_entry(
                        *tid,
                        waterfall::Entry {
                            id: format!("{}", idx),
                            label: instructions::instruction_label_by_meta(
                                ie.instruction.node(),
                                ie.function_meta.as_ref(),
                            ),
                            start: ie.start.nanos,
                            end: ie.end.nanos,
                            success: true,
                            worker: format!("Thread {}", usize::from(ie.thread)),
                            content: format!("{:?}", &ie.instruction),
                        },
                    ),
                };
            });

        builder
    }
}
