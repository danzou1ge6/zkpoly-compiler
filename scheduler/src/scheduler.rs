use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use zkpoly_compiler::driver::artifect::Pools;
use zkpoly_compiler::driver::{Artifect, HardwareInfo};
use zkpoly_memory_pool::static_allocator::CpuStaticAllocator;
use zkpoly_runtime::args::{EntryTable, RuntimeType, Variable};
use zkpoly_runtime::async_rng::AsyncRng;
use zkpoly_runtime::runtime::{RuntimeDebug, RuntimeInfo};

// 任务状态
#[derive(Debug)]
pub enum TaskStatus {
    Pending,
    Running { assigned_cards: Vec<usize> }, // 任务正在运行，包含已分配的卡片
    Completed,
}

// 任务结构
struct Task<Rt: RuntimeType> {
    artifect: Artifect<Rt>,
    status: Arc<Mutex<TaskStatus>>,
    result_sender: mpsc::Sender<(Option<Variable<Rt>>, RuntimeInfo<Rt>)>, // 用于发送任务结果
    hardware_info: HardwareInfo,
    rng: AsyncRng,
    inputs: EntryTable<Rt>,
    debug_opt: RuntimeDebug,
}

pub struct Scheduler<Rt: RuntimeType> {
    pub cards_per_request: usize,
    pub total_cards: usize,
    sender: crossbeam_channel::Sender<Task<Rt>>,
    _scheduler_threads: Vec<thread::JoinHandle<()>>,
}

impl<Rt: RuntimeType> Scheduler<Rt> {
    pub fn new(cards_per_request: usize, total_cards: usize) -> Self {
        assert!(
            total_cards % cards_per_request == 0,
            "Total cards must be a multiple of cards per request"
        );

        let (sender, receiver) = crossbeam_channel::unbounded();
        let mut scheduler_threads = Vec::new();

        for i in (0..total_cards).step_by(cards_per_request) {
            let receiver: crossbeam_channel::Receiver<Task<Rt>> = receiver.clone();
            let cur_cards = (i..(i + cards_per_request))
                .into_iter()
                .collect::<Vec<usize>>();
            scheduler_threads.push(thread::spawn(move || {
                let cur_cards_clone = cur_cards.clone();
                let gpu_mapping = Arc::new(move |x: i32| cur_cards_clone[x as usize] as i32);
                while let Ok(task) = receiver.recv() {
                    let mut status = task.status.lock().unwrap();
                    assert!(
                        matches!(*status, TaskStatus::Pending),
                        "Task must be pending before running"
                    );
                    *status = TaskStatus::Running {
                        assigned_cards: cur_cards.clone(),
                    }; // 更新任务状态为运行中，并记录已分配的卡片
                    drop(status); // 释放锁

                    let result: (Option<Variable<Rt>>, RuntimeInfo<Rt>);
                    let pools = task.artifect.create_pools(&task.hardware_info, true);
                    let mut runtime = task.artifect.prepare_dispatcher(
                        pools,
                       task.rng,
                        gpu_mapping.clone(),
                    );

                    let start = std::time::Instant::now();
                    (result, _) = runtime.run(&task.inputs, task.debug_opt);
                    let elapsed = start.elapsed();

                    // 更新任务状态为已完成
                    let mut status = task.status.lock().unwrap();
                    *status = TaskStatus::Completed;
                    drop(status); // 释放锁
                    println!(
                        "Scheduler thread {} completed task with GPUs {:?} in {:?}",
                        i / cards_per_request,
                        cur_cards,
                        elapsed
                    );

                    task.result_sender
                        .send(result)
                        .expect("Failed to send task result");
                }
            }))
        }

        Self {
            cards_per_request,
            total_cards,
            sender,
            _scheduler_threads: scheduler_threads,
        }
    }

    pub fn add_request(
        &self,
        request: Artifect<Rt>,
        hd_info: HardwareInfo,
        rng: AsyncRng,
        inputs: EntryTable<Rt>,
        debug_opt: RuntimeDebug,
    ) -> (
        Arc<Mutex<TaskStatus>>,
        mpsc::Receiver<(Option<Variable<Rt>>, RuntimeInfo<Rt>)>,
    ) {
        let (result_sender, result_receiver) = mpsc::channel();
        let status = Arc::new(Mutex::new(TaskStatus::Pending));

        let task = Task {
            artifect: request,
            status: Arc::clone(&status),
            result_sender,
            hardware_info: hd_info,
            rng,
            inputs,
            debug_opt,
        };

        self.sender
            .send(task)
            .expect("Failed to send task to scheduler");

        (status, result_receiver)
    }
}
