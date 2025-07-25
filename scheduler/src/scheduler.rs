use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self};
use std::time::{Duration, Instant};
use zkpoly_common::define_usize_id;
use zkpoly_common::heap::{Heap, IdAllocator};
use zkpoly_compiler::driver::artifect::Pools;
use zkpoly_compiler::driver::{Artifect, HardwareInfo, MemoryInfo};
use zkpoly_memory_pool::buddy_disk_pool::DiskMemoryPool;
use zkpoly_memory_pool::static_allocator::CpuStaticAllocator;
use zkpoly_runtime::args::{EntryTable, RuntimeType, Variable};
use zkpoly_runtime::async_rng::AsyncRng;
use zkpoly_runtime::debug::Log;
use zkpoly_runtime::runtime::{Runtime, RuntimeDebug};

define_usize_id!(ProgramId);
define_usize_id!(TaskId);

#[derive(Debug)]
struct HalfMemory {
    assigned_tasks: HashSet<TaskId>,
    used: usize,
    offset: usize,
    total: usize,
    estimated_free_at: Instant,
}

impl HalfMemory {
    fn empty(offset: usize, size: usize) -> Self {
        Self {
            assigned_tasks: HashSet::new(),
            used: 0,
            offset,
            total: size,
            estimated_free_at: Instant::now(),
        }
    }

    fn free(&self) -> bool {
        self.assigned_tasks.is_empty()
    }

    fn add_task<Rt: RuntimeType>(
        &mut self,
        task: &AcceptedTask<Rt>,
        estimated_free_at: Instant,
    ) -> usize {
        let offset = self.offset + self.used;

        let task_size = task.version.as_ref().unwrap().memory_limit() as usize;
        self.assigned_tasks.insert(task.id);
        self.used += task_size;
        self.estimated_free_at = self.estimated_free_at.max(estimated_free_at);

        offset
    }

    fn space_left(&self) -> usize {
        self.total - self.used
    }
}

#[derive(Debug)]
enum AvailableMemory {
    Intact(Option<TaskId>),
    Halved([HalfMemory; 2]),
}

impl AvailableMemory {
    fn try_switch_to_intact(&mut self) {
        match &self {
            AvailableMemory::Halved([h1, h2]) if h1.free() && h2.free() => {
                *self = AvailableMemory::Intact(None)
            }
            _ => {}
        }
    }

    fn try_switch_to_halved(&mut self, total_memory: usize) {
        if let AvailableMemory::Intact(None) = &self {
            let half_size = total_memory / 2;
            *self = AvailableMemory::Halved([
                HalfMemory::empty(0, half_size),
                HalfMemory::empty(half_size, total_memory - half_size),
            ]);
        }
    }
}

#[derive(Debug)]
pub struct GlobalResources {
    available_cards: Vec<i32>,
    available_memory: AvailableMemory,
    /// Total bytes of CPU memory
    total_memory: usize,
}

impl GlobalResources {
    pub fn new(cards: Vec<i32>, available_memory: usize) -> Self {
        Self {
            available_memory: AvailableMemory::Intact(None),
            total_memory: available_memory,
            available_cards: cards,
        }
    }
}

fn try_take_cards(cards: &mut Vec<i32>, n: usize) -> Option<Vec<i32>> {
    if cards.len() >= n {
        Some((0..n).map(|_| cards.pop().unwrap()).collect())
    } else {
        None
    }
}

impl GlobalResources {
    fn try_take_cards(&mut self, n: usize) -> Option<Vec<i32>> {
        try_take_cards(&mut self.available_cards, n)
    }
}

impl GlobalResources {}

#[derive(Debug)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
}

pub struct RunReturn<Rt: RuntimeType> {
    pub ret_value: Option<Variable<Rt>>,
    pub log: Log,
    pub time: Duration,
}

pub struct WorkerFinishRespond<Rt: RuntimeType> {
    r: RunReturn<Rt>,
    id: TaskId,
    program: ProgramId,
    version: MemoryInfo,
    cards: Vec<i32>,
}

struct LaunchedTask<Rt: RuntimeType> {
    /// The `version`, `cards` and `memory_offset` fields in `accepted`
    /// are expected to be set here.
    accepted: AcceptedTask<Rt>,
    runtime: Runtime<Rt>,
}

struct AcceptedTask<Rt: RuntimeType> {
    id: TaskId,
    submitted: SubmittedTask<Rt>,
    version: Option<MemoryInfo>,
    cards: Vec<i32>,
    memory_offset: Option<usize>,
}

impl<Rt: RuntimeType> std::fmt::Debug for AcceptedTask<Rt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcceptedTask")
            .field("id", &self.id)
            .field("version", &self.version)
            .field("cards", &self.cards)
            .field("memory_offset", &self.memory_offset)
            .finish()
    }
}

pub struct SubmittedTask<Rt: RuntimeType> {
    program: ProgramId,
    inputs: EntryTable<Rt>,
    debug_opt: RuntimeDebug,
}

impl<Rt: RuntimeType> SubmittedTask<Rt> {
    pub fn new(program: ProgramId, inputs: EntryTable<Rt>) -> Self {
        Self {
            program,
            inputs,
            debug_opt: RuntimeDebug::none(),
        }
    }

    pub fn with_debug_opt(self, x: RuntimeDebug) -> Self {
        Self {
            debug_opt: x,
            ..self
        }
    }
}

pub struct Submit<Rt: RuntimeType> {
    task: SubmittedTask<Rt>,
    result_sender: crossbeam_channel::Sender<RunReturn<Rt>>,
}

struct ExecutorHandle {
    thread: thread::JoinHandle<()>,
}

#[derive(Debug)]
pub struct SchedulerHandle {
    shutdown_sender: mpsc::Sender<()>,
    thread: thread::JoinHandle<()>,
}

impl SchedulerHandle {
    pub fn shutdown(self) {
        self.shutdown_sender
            .send(())
            .expect("send shutdown command failed");
        self.thread.join().expect("join scheduler failed");
    }
}

pub struct SchedulerConfig {
    cards_per_request: usize,
    num_executors: usize,
    memory_check: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            cards_per_request: 1,
            num_executors: 4,
            memory_check: true,
        }
    }
}

impl SchedulerConfig {
    pub fn with_cards_per_request(self, x: usize) -> Self {
        Self {
            cards_per_request: x,
            ..self
        }
    }

    pub fn with_num_executors(self, x: usize) -> Self {
        Self {
            num_executors: x,
            ..self
        }
    }

    pub fn with_memory_check(self, x: bool) -> Self {
        Self {
            memory_check: x,
            ..self
        }
    }
}

pub enum SubmitterMessage<Rt: RuntimeType> {
    Submit(Submit<Rt>),
    Add(Artifect<Rt>, mpsc::Sender<ProgramId>),
}

#[derive(Debug, Clone)]
pub struct Submitter<Rt: RuntimeType> {
    sender: mpsc::Sender<SubmitterMessage<Rt>>,
}

pub type SubmitResult<T, Rt> = Result<T, mpsc::SendError<SubmitterMessage<Rt>>>;

impl<Rt: RuntimeType> Submitter<Rt> {
    pub fn submit(
        &self,
        task: SubmittedTask<Rt>,
    ) -> SubmitResult<crossbeam_channel::Receiver<RunReturn<Rt>>, Rt> {
        let (sender, receiver) = crossbeam_channel::unbounded::<RunReturn<Rt>>();

        self.sender.send(SubmitterMessage::Submit(Submit {
            task,
            result_sender: sender,
        }))?;

        Ok(receiver)
    }

    pub fn add_artifect(&self, artifect: Artifect<Rt>) -> SubmitResult<ProgramId, Rt> {
        let (sender, receiver) = mpsc::channel::<ProgramId>();

        self.sender.send(SubmitterMessage::Add(artifect, sender))?;

        Ok(receiver.recv().unwrap())
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

struct Control<Rt: RuntimeType> {
    reception: mpsc::Receiver<SubmitterMessage<Rt>>,
    executors: Vec<ExecutorHandle>,
    shutdown_receiver: mpsc::Receiver<()>,
    task_launcher: crossbeam_channel::Sender<LaunchedTask<Rt>>,
    result_receiver: crossbeam_channel::Receiver<WorkerFinishRespond<Rt>>,
    result_relayer: HashMap<TaskId, crossbeam_channel::Sender<RunReturn<Rt>>>,
}

mod core {

    use super::*;

    pub struct Core<Rt: RuntimeType> {
        pub config: SchedulerConfig,
        /// Newly submitted tasks go here
        pending_tasks: Vec<AcceptedTask<Rt>>,
        /// We schedule tasks window by window. by algorithms described in
        ///   Approximate Algorithms for Scheduling Parallelizable Tasks
        /// Takss from `pending_tasks` are dumped to this queue periodically
        scheduling_window: Vec<AcceptedTask<Rt>>,
        scheduling_window_sorted: bool,
        scheduling_window_allotted: bool,
        resources: GlobalResources,
        pub programs: Heap<ProgramId, Artifect<Rt>>,
        task_id_allocator: IdAllocator<TaskId>,
        pub knowledge: Knowlede,
    }

    impl<Rt: RuntimeType> Core<Rt> {
        pub fn new(
            config: SchedulerConfig,
            resources: GlobalResources,
            programs: Heap<ProgramId, Artifect<Rt>>,
        ) -> Self {
            Self {
                config: config,
                pending_tasks: Vec::new(),
                scheduling_window: Vec::new(),
                scheduling_window_sorted: false,
                scheduling_window_allotted: false,
                resources: resources,
                knowledge: Knowlede {
                    time_experience: programs.map_by_ref(&mut |_, artifect| {
                        artifect
                            .versions()
                            .map(|ver| (ver.clone(), Duration::from_secs(1)))
                            .collect()
                    }),
                },
                programs: programs,
                task_id_allocator: IdAllocator::new(),
            }
        }
        pub fn add_task(&mut self, submit: SubmittedTask<Rt>) -> TaskId {
            let id = self.task_id_allocator.alloc();
            let accepted = AcceptedTask {
                id,
                submitted: submit,
                cards: Vec::new(),
                version: None,
                memory_offset: None,
            };
            self.pending_tasks.push(accepted);
            id
        }

        fn dump_tasks_to_scheduling_window(&mut self) {
            self.scheduling_window
                .extend(std::mem::take(&mut self.pending_tasks).into_iter());
            self.scheduling_window_sorted = false;
            self.scheduling_window_allotted = false;
        }

        /// Determine CPU memory assigned to each task in the scheduling window.
        /// Window must be non-empty>
        fn allot_memory(&mut self) {
            // Initially set
            //   beta_j^1 = arg min_i t_j (i) dot i
            // where beta_j^k is the CPU memory allocated to task j at iteration k,
            // and t_j (i) is estimated execution of task j with i bytes assigned
            self.scheduling_window.iter_mut().for_each(|task| {
                let version = self.programs[task.submitted.program]
                    .versions()
                    .min_by_key(|m| {
                        self.knowledge
                            .estimate_time(task.submitted.program, *m)
                            .mul_f64(m.memory_limit_gigabytes())
                    })
                    .expect("artifect does not provide any version");
                task.version = Some(version.clone());
            });

            loop {
                // In each iteration k, we find task j_0 such that
                //   j0 = arg max t_j (beta_j^(k - 1))
                // and allot more memory to it
                //   beta_j0^k = arg min_(i > beta_j0^(k - 1)) t_j (i) dot i
                // while maintaining
                //   beta_j^k = beta_j^(k - 1)
                // for all j != j0
                let bottoleneck = self
                    .scheduling_window
                    .iter_mut()
                    .max_by_key(|task| {
                        self.knowledge
                            .estimate_time(task.submitted.program, task.version.as_ref().unwrap())
                    })
                    .expect("empty schedule window");
                let version = self.programs[bottoleneck.submitted.program]
                    .versions()
                    .filter(|m| {
                        m.memory_limit() > bottoleneck.version.as_ref().unwrap().memory_limit()
                    })
                    .min_by_key(|m| {
                        self.knowledge
                            .estimate_time(bottoleneck.submitted.program, *m)
                            .mul_f64(m.memory_limit_gigabytes())
                    });
                if let Some(version) = version {
                    bottoleneck.version = Some(version.clone());
                } else {
                    self.scheduling_window_allotted = true;
                    return;
                }
            }
        }

        /// Choose from schedule window a task to run.
        /// The returned [`AcceptedTask`] guarantees that the `version`, `cards` and `memory_offset` fields are set.
        pub fn schedule(&mut self) -> Option<AcceptedTask<Rt>> {
            if self.scheduling_window.is_empty() {
                self.dump_tasks_to_scheduling_window();
            }

            if self.scheduling_window.is_empty() {
                return None;
            }

            println!(
                "调度队列为 {:?}",
                self.scheduling_window
                    .iter()
                    .map(|t| t.id)
                    .collect::<Vec<_>>()
            );

            if !self.scheduling_window_allotted {
                self.allot_memory();
            }

            // If there are tasks consuming more than half of total, try schedule them first
            if let Some(j) = self.scheduling_window.iter().position(|task| {
                task.version.as_ref().unwrap().memory_limit() * 2
                    > self.resources.total_memory as u64
            }) {
                self.resources.available_memory.try_switch_to_intact();

                if let Some(cards) = self.resources.try_take_cards(self.config.cards_per_request) {
                    if let AvailableMemory::Intact(None) = self.resources.available_memory {
                        let mut task = self.scheduling_window.remove(j);
                        task.cards = cards;
                        task.memory_offset = Some(0);
                        self.resources.available_memory = AvailableMemory::Intact(Some(task.id));

                        println!("调度任务 {:?} ，使用整块内存", &task);

                        return Some(task);
                    }
                }
                return None;
            }

            self.resources
                .available_memory
                .try_switch_to_halved(self.resources.total_memory);

            // Otherwise, sort the schedule window by execution time,
            // then try schedule first one to the half that is estimated to become free earlier
            if !self.scheduling_window_sorted {
                self.scheduling_window.sort_by(|a, b| {
                    self.knowledge
                        .estimate_time(a.submitted.program, b.version.as_ref().unwrap())
                        .cmp(
                            &self
                                .knowledge
                                .estimate_time(a.submitted.program, b.version.as_ref().unwrap()),
                        )
                        .reverse()
                });
                self.scheduling_window_sorted = true;
            }

            if let Some(task) = self.scheduling_window.last() {
                if let AvailableMemory::Halved(halves) = &mut self.resources.available_memory {
                    halves.sort_by_key(|halve| halve.estimated_free_at);

                    for half in halves {
                        if let Some(cards) = try_take_cards(
                            &mut self.resources.available_cards,
                            self.config.cards_per_request,
                        ) {
                            if half.space_left()
                                >= task.version.as_ref().unwrap().memory_limit() as usize
                            {
                                let mut task = self.scheduling_window.pop().unwrap();
                                let estimated_free_at = Instant::now()
                                    + self.knowledge.estimate_time(
                                        task.submitted.program,
                                        task.version.as_ref().unwrap(),
                                    );
                                task.cards = cards;
                                task.memory_offset = Some(half.add_task(&task, estimated_free_at));

                                println!("调度任务 {:?} ，使用半块内存", &task);

                                return Some(task);
                            }
                        }
                    }
                }
                return None;
            }

            None
        }

        pub fn finish(
            &mut self,
            id: TaskId,
            cards: &[i32],
            program: ProgramId,
            version: &MemoryInfo,
            duration: Duration,
        ) {
            self.knowledge.update(program, version, duration);

            match &mut self.resources.available_memory {
                AvailableMemory::Intact(t) => {
                    if t.is_some_and(|x| x == id) {
                        self.resources.available_memory = AvailableMemory::Intact(None)
                    } else {
                        panic!(
                            "finishing a task {:?} but availabel memory is Intact({:?})",
                            id, t
                        )
                    }
                }
                AvailableMemory::Halved([h1, h2]) => {
                    if !h1.assigned_tasks.remove(&id) && !h2.assigned_tasks.remove(&id) {
                        panic!(
                            "finishing a task {:?} but no memory is assigned to it in neither half",
                            id
                        );
                    }
                }
            }

            self.resources.available_cards.extend_from_slice(cards);
        }
    }
}

use core::Core;

pub struct Scheduler<Rt: RuntimeType> {
    hd_info: HardwareInfo,
    control: Control<Rt>,
    core: Core<Rt>,
    cpu_memory: CpuStaticAllocator,
    disk_memory: Arc<Mutex<DiskMemoryPool>>,
    rng: AsyncRng,
    shutdown_sender: mpsc::Sender<()>,
}

impl<Rt: RuntimeType> Scheduler<Rt> {
    fn add_task(&mut self, submit: Submit<Rt>) -> TaskId {
        let id = self.core.add_task(submit.task);
        self.control.result_relayer.insert(id, submit.result_sender);
        id
    }

    fn finish_task(&mut self, resp: WorkerFinishRespond<Rt>) {
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

            let launched = LaunchedTask { accepted, runtime };
            self.control
                .task_launcher
                .send(launched)
                .expect("all task spawn closed");
        }
    }

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
                        SubmitterMessage::Submit(submitted) => {
                            let id = self.add_task(submitted);
                            self.schedule_task();
                            println!("调度器接收到新任务 {:?}", id);
                        }
                        SubmitterMessage::Add(ar, sender) => {
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

pub fn make_scheduler<Rt: RuntimeType>(
    hd_info: HardwareInfo,
    config: SchedulerConfig,
    rng: AsyncRng,
    disk_pool: DiskMemoryPool,
    programs: Heap<ProgramId, Artifect<Rt>>,
) -> (Scheduler<Rt>, Submitter<Rt>) {
    assert!(
        hd_info.gpus().count() % config.cards_per_request == 0,
        "Total cards must be a multiple of cards per request"
    );

    // 创建任务接收通道（用户 -> 调度器）
    let (task_submitter, task_reception) = mpsc::channel::<SubmitterMessage<Rt>>();

    // 创建执行任务通道（调度器 -> 执行器）
    let (task_launcher, task_spawn) = crossbeam_channel::unbounded::<LaunchedTask<Rt>>();

    // 创建执行器状态通道（执行器 -> 调度器）
    let (result_sender, result_receiver) =
        crossbeam_channel::unbounded::<WorkerFinishRespond<Rt>>();

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
                    let ((r, log, _), _) = runtime.run(
                        &task.accepted.submitted.inputs,
                        task.accepted.submitted.debug_opt,
                    );
                    let elapsed = start.elapsed();

                    println!(
                        "执行器 {} 完成任务 {:?} 在 {:?}",
                        i, task.accepted.id, elapsed,
                    );

                    // 发送结果
                    if let Err(e) = result_sender.send(WorkerFinishRespond {
                        r: RunReturn {
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
        GlobalResources::new(
            (0..hd_info.gpus().count()).map(|x| x as i32).collect(),
            hd_info.cpu().memory_limit() as usize,
        ),
        programs,
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

    let smt = Submitter {
        sender: task_submitter,
    };

    (sch, smt)
}
