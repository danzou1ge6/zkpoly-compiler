use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use zkpoly_compiler::driver::{Artifect, HardwareInfo};
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
    scheduler_threads: Vec<thread::JoinHandle<()>>,
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

                    let gpu_allocator = task
                        .hardware_info
                        .gpus()
                        .enumerate()
                        .map(|(id, gpu)| {
                            zkpoly_cuda_api::mem::CudaAllocator::new(
                                cur_cards[id] as i32,
                                gpu.memory_limit() as usize,
                                true,
                            )
                        })
                        .collect::<Vec<_>>();

                    let result = task
                        .artifect
                        .prepare_dispatcher(gpu_allocator, task.rng, cur_cards[0] as i32) // currently, cur_cards are continuous
                        .run(&task.inputs, task.debug_opt);

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
            scheduler_threads,
        }
    }

    pub fn add_request(
        &mut self,
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
