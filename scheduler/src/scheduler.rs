mod prelude {
    pub use std::collections::{HashMap, HashSet};
    pub use std::sync::{mpsc, Arc, Mutex};
    pub use std::thread::{self};
    pub use std::time::{Duration, Instant};
    pub use zkpoly_common::define_usize_id;
    pub use zkpoly_common::heap::{Heap, IdAllocator, UsizeId};
    pub use zkpoly_compiler::driver::artifect::{GpuMapping, Pools};
    pub use zkpoly_compiler::driver::{Artifect, HardwareInfo, MemoryInfo};
    pub use zkpoly_memory_pool::buddy_disk_pool::DiskMemoryPool;
    pub use zkpoly_memory_pool::static_allocator::CpuStaticAllocator;
    pub use zkpoly_runtime::args::{EntryTable, RuntimeType, Variable};
    pub use zkpoly_runtime::async_rng::AsyncRng;
    pub use zkpoly_runtime::debug::Log;
    pub use zkpoly_runtime::runtime::{Runtime, RuntimeDebug};
}

use prelude::*;

define_usize_id!(TaskId);

mod erased;
mod submitter;

pub use submitter::exposed::*;
use submitter::ProgramId;
pub use submitter::Submitter;

struct WorkerFinishRespond {
    r: submitter::RunReturn,
    id: TaskId,
    program: ProgramId,
    version: MemoryInfo,
    cards: Vec<i32>,
}

struct LaunchedTask {
    /// The `version`, `cards` and `memory_offset` fields in `accepted`
    /// are expected to be set here.
    accepted: AcceptedTask,
    runtime: erased::BoxedRuntime,
    runtime_debug: RuntimeDebug,
}

struct AcceptedTask {
    id: TaskId,
    submitted: submitter::SubmittedTask,
    version: Option<MemoryInfo>,
    cards: Vec<i32>,
    memory_offset: Option<usize>,
}

impl std::fmt::Debug for AcceptedTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcceptedTask")
            .field("id", &self.id)
            .field("version", &self.version)
            .field("cards", &self.cards)
            .field("program", &self.submitted.program)
            .field("memory_offset", &self.memory_offset)
            .finish()
    }
}

struct ExecutorHandle {
    thread: thread::JoinHandle<()>,
}

/// Handle to the scheduler thread.
#[derive(Debug)]
pub struct SchedulerHandle {
    shutdown_sender: mpsc::Sender<()>,
    thread: thread::JoinHandle<()>,
}

impl SchedulerHandle {
    /// Shutdown the scheduler thread and executor threads.
    pub fn shutdown(self) {
        self.shutdown_sender
            .send(())
            .expect("send shutdown command failed");
        self.thread.join().expect("join scheduler failed");
    }
}

/// Configuration for the scheduler.
pub struct SchedulerConfig {
    num_executors: usize,
    memory_check: bool,
    runtime_debug: RuntimeDebug,
    schedule_window_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            num_executors: 4,
            memory_check: true,
            runtime_debug: RuntimeDebug::none(),
            schedule_window_size: 32,
        }
    }
}

impl SchedulerConfig {
    /// Controls number of executors launched by the scheduler.
    pub fn with_num_executors(self, x: usize) -> Self {
        Self {
            num_executors: x,
            ..self
        }
    }

    /// Controls whether memory sanity checks are enabled, for debugging.
    pub fn with_memory_check(self, x: bool) -> Self {
        Self {
            memory_check: x,
            ..self
        }
    }

    /// Controls runtime debug options.
    pub fn with_runtime_debug(self, x: RuntimeDebug) -> Self {
        Self {
            runtime_debug: x,
            ..self
        }
    }

    /// Configures the schedule window size.
    pub fn with_schedule_window_size(self, x: usize) -> Self {
        Self {
            schedule_window_size: x,
            ..self
        }
    }
}

struct Knowlede {
    /// Duration for all versions are expected to be found here.
    /// There fore, they must be initialized to some estimations before hand.
    time_experience: Heap<ProgramId, Vec<(MemoryInfo, Duration)>>,
}

impl Knowlede {
    fn estimate_time(&self, program: ProgramId, version: &MemoryInfo) -> Duration {
        let assoc = &self.time_experience[program];
        assoc
            .iter()
            .find(|(v1, _)| v1 == version)
            .map(|(_, t)| *t)
            .unwrap()
    }

    fn update(&mut self, program: ProgramId, version: &MemoryInfo, duration: Duration) {
        const SMOOTH_FACTOR: f32 = 0.4;

        let assoc = &mut self.time_experience[program];
        let (_, old) = assoc.iter_mut().find(|(v1, _)| v1 == version).unwrap();

        *old = old.mul_f32(SMOOTH_FACTOR) + duration.mul_f32(1.0 - SMOOTH_FACTOR);
    }
}

struct Control {
    reception: mpsc::Receiver<submitter::Message>,
    executors: Vec<ExecutorHandle>,
    shutdown_receiver: mpsc::Receiver<()>,
    task_launcher: crossbeam_channel::Sender<LaunchedTask>,
    result_receiver: crossbeam_channel::Receiver<WorkerFinishRespond>,
    result_relayer: HashMap<TaskId, crossbeam_channel::Sender<submitter::RunReturn>>,
}

mod core;

use core::Core;
use std::marker::PhantomData;

/// The scheduler.
/// It runs in a dedicated thread, and communicates with executors and users with channels.
pub struct Scheduler {
    hd_info: HardwareInfo,
    control: Control,
    core: Core,
    cpu_memory: CpuStaticAllocator,
    disk_memory: Arc<Mutex<DiskMemoryPool>>,
    rng: AsyncRng,
    shutdown_sender: mpsc::Sender<()>,
}

impl Scheduler {
    fn add_task(&mut self, submit: submitter::Submit) -> TaskId {
        let id = self.core.add_task(submit.task);
        self.control.result_relayer.insert(id, submit.result_sender);
        id
    }

    fn finish_task(&mut self, resp: WorkerFinishRespond) {
        self.core.finish(
            resp.id,
            &resp.cards,
            resp.program,
            &resp.version,
            resp.r.time,
        );

        let _ = self
            .control
            .result_relayer
            .remove(&resp.id)
            .unwrap()
            .try_send(resp.r);
    }

    fn schedule_task(&mut self) {
        while let Some(accepted) = self.core.schedule() {
            let pools = Pools {
                cpu: self.cpu_memory.slice(
                    accepted.memory_offset.unwrap(),
                    accepted.version.as_ref().unwrap().memory_limit() as usize,
                ),
                gpu: self.hd_info.gpu_allocators_for(
                    self.core.config.memory_check,
                    accepted.cards.iter().copied(),
                ),
                disk: self.disk_memory.clone(),
            };

            let runtime = self.core.programs[accepted.submitted.program].prepare_dispatcher(
                accepted.version.as_ref().unwrap(),
                pools,
                self.rng.clone(),
                Arc::new({
                    let mapping = accepted.cards.clone();
                    move |i| mapping[i as usize]
                }),
            );

            let launched = LaunchedTask {
                accepted,
                runtime,
                runtime_debug: self.core.config.runtime_debug.clone(),
            };
            self.control
                .task_launcher
                .send(launched)
                .expect("all task spawn closed");
        }
    }

    /// Launch the scheduler thread.
    pub fn launch(mut self) -> SchedulerHandle {
        let shutdown_sender = self.shutdown_sender.clone();

        let jh = thread::spawn(move || {
            println!(
                "调度器线程启动，管理 {} 个执行器",
                self.core.config.num_executors
            );

            loop {
                // 检查是否有新的用户任务
                while let Ok(msg) = self.control.reception.try_recv() {
                    match msg {
                        submitter::Message::Submit(submitted) => {
                            let id = self.add_task(submitted);
                            self.schedule_task();
                            println!("调度器接收到新任务 {:?}", id);
                        }
                        submitter::Message::Add(ar, sender) => {
                            self.core.knowledge.time_experience.push(
                                ar.versions()
                                    .map(|ver| (ver.clone(), Duration::from_secs(1)))
                                    .collect(),
                            );
                            let id = self.core.programs.push(ar);
                            let _ = sender.send(id);
                        }
                    }
                }

                // 执行器完成
                while let Ok(resp) = self.control.result_receiver.try_recv() {
                    println!("任务 {:?} 完成", resp.id);
                    self.finish_task(resp);
                    self.schedule_task();
                }

                if let Ok(..) = self.control.shutdown_receiver.try_recv() {
                    break;
                }

                // 短暂休眠避免忙等待
                thread::sleep(Duration::from_millis(10));
            }

            drop(self.control.task_launcher);

            println!("准备调度器退出，等待工作线程结束");
            for exe in self.control.executors {
                exe.thread.join().unwrap();
            }

            println!("调度器退出");
        });

        SchedulerHandle {
            shutdown_sender,
            thread: jh,
        }
    }
}

/// Make a new scheduler and launch executor threads.
pub fn make_scheduler<Rt: RuntimeType>(
    hd_info: HardwareInfo,
    config: SchedulerConfig,
    rng: AsyncRng,
    disk_pool: DiskMemoryPool,
    programs: Programs<Rt>,
) -> (Scheduler, submitter::Submitter<Rt>) {
    let cards_per_request = 1; // For now this is a constant.

    assert!(
        hd_info.gpus().count() % cards_per_request == 0,
        "Total cards must be a multiple of cards per request"
    );

    // 创建任务接收通道（用户 -> 调度器）
    let (task_submitter, task_reception) = mpsc::channel::<submitter::Message>();

    // 创建执行任务通道（调度器 -> 执行器）
    let (task_launcher, task_spawn) = crossbeam_channel::unbounded::<LaunchedTask>();

    // 创建执行器状态通道（执行器 -> 调度器）
    let (result_sender, result_receiver) = crossbeam_channel::unbounded::<WorkerFinishRespond>();

    let (shutdown_sender, shutdown_receiver) = mpsc::channel::<()>();

    let executors = (0..config.num_executors)
        .map(|i| {
            let task_spawn = task_spawn.clone();
            let result_sender = result_sender.clone();

            let executor_thread = thread::spawn(move || {
                println!("执行器线程 {} 启动", i);

                while let Ok(task) = task_spawn.recv() {
                    println!("执行器 {} 开始执行任务 {:?}", i, task.accepted.id,);

                    // 执行任务
                    let start = std::time::Instant::now();

                    let mut runtime = task.runtime;
                    let (r, log) =
                        runtime.run(task.accepted.submitted.inputs.as_ref(), task.runtime_debug);
                    let elapsed = start.elapsed();

                    drop(runtime);

                    println!(
                        "执行器 {} 完成任务 {:?} 在 {:?}",
                        i, task.accepted.id, elapsed,
                    );

                    // 发送结果
                    if let Err(e) = result_sender.send(WorkerFinishRespond {
                        r: submitter::RunReturn {
                            ret_value: r,
                            log,
                            time: elapsed,
                        },
                        id: task.accepted.id,
                        program: task.accepted.submitted.program,
                        version: task.accepted.version.unwrap(),
                        cards: task.accepted.cards,
                    }) {
                        eprintln!(
                            "执行器 {} 发送任务 {:?} 结果失败: {:?}",
                            i, task.accepted.id, e
                        );
                    }
                }
            });

            ExecutorHandle {
                thread: executor_thread,
            }
        })
        .collect::<Vec<_>>();

    let memory_check = config.memory_check;
    let core = Core::new(
        config,
        core::GlobalResources::new(
            (0..hd_info.gpus().count()).map(|x| x as i32).collect(),
            hd_info.cpu().memory_limit() as usize,
        ),
        programs.erase(),
    );

    let control = Control {
        reception: task_reception,
        executors,
        shutdown_receiver,
        task_launcher,
        result_receiver,
        result_relayer: HashMap::new(),
    };

    let sch = Scheduler {
        cpu_memory: hd_info.cpu_allocator(memory_check),
        hd_info,
        control,
        core,
        disk_memory: Arc::new(Mutex::new(disk_pool)),
        rng,
        shutdown_sender,
    };

    let smt = submitter::Submitter {
        sender: task_submitter,
        _phantom: PhantomData,
    };

    (sch, smt)
}
