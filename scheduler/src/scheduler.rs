use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;
use zkpoly_compiler::driver::{Artifect, HardwareInfo};
use zkpoly_runtime::args::{EntryTable, RuntimeType, Variable};
use zkpoly_runtime::async_rng::AsyncRng;
use zkpoly_runtime::debug::Log;
use zkpoly_runtime::runtime::{RuntimeDebug, RuntimeInfo};

// 资源需求结构体
#[derive(Debug, Clone)]
pub struct ResourceRequirement {
    pub cpu_memory_mb: u64, // CPU内存需求 (MB)
    pub disk_space_mb: u64, // 磁盘空间需求 (MB)
}

// 全局资源管理器
#[derive(Debug)]
pub struct GlobalResourceManager {
    available_cpu_memory_mb: Arc<Mutex<u64>>,
    available_disk_space_mb: Arc<Mutex<u64>>,
    total_cpu_memory_mb: u64,
    total_disk_space_mb: u64,
}

impl GlobalResourceManager {
    pub fn new(total_cpu_memory_mb: u64, total_disk_space_mb: u64) -> Self {
        Self {
            available_cpu_memory_mb: Arc::new(Mutex::new(total_cpu_memory_mb)),
            available_disk_space_mb: Arc::new(Mutex::new(total_disk_space_mb)),
            total_cpu_memory_mb,
            total_disk_space_mb,
        }
    }

    // 尝试分配资源，如果资源不足返回false
    pub fn try_allocate(&self, requirement: &ResourceRequirement) -> bool {
        let mut cpu_memory = self.available_cpu_memory_mb.lock().unwrap();
        let mut disk_space = self.available_disk_space_mb.lock().unwrap();

        if *cpu_memory >= requirement.cpu_memory_mb && *disk_space >= requirement.disk_space_mb {
            *cpu_memory -= requirement.cpu_memory_mb;
            *disk_space -= requirement.disk_space_mb;
            true
        } else {
            false
        }
    }

    // 释放资源
    pub fn release(&self, requirement: &ResourceRequirement) {
        let mut cpu_memory = self.available_cpu_memory_mb.lock().unwrap();
        let mut disk_space = self.available_disk_space_mb.lock().unwrap();

        *cpu_memory += requirement.cpu_memory_mb;
        *disk_space += requirement.disk_space_mb;

        // 确保不超过总量
        if *cpu_memory > self.total_cpu_memory_mb {
            *cpu_memory = self.total_cpu_memory_mb;
        }
        if *disk_space > self.total_disk_space_mb {
            *disk_space = self.total_disk_space_mb;
        }
    }

    // 获取当前可用资源信息
    pub fn get_available_resources(&self) -> (u64, u64) {
        let cpu_memory = *self.available_cpu_memory_mb.lock().unwrap();
        let disk_space = *self.available_disk_space_mb.lock().unwrap();
        (cpu_memory, disk_space)
    }

    // 获取资源使用率
    pub fn get_utilization(&self) -> (f64, f64) {
        let (available_cpu, available_disk) = self.get_available_resources();
        let cpu_utilization = 1.0 - (available_cpu as f64 / self.total_cpu_memory_mb as f64);
        let disk_utilization = 1.0 - (available_disk as f64 / self.total_disk_space_mb as f64);
        (cpu_utilization, disk_utilization)
    }
}

// 任务状态
#[derive(Debug)]
pub enum TaskStatus {
    Pending,
    Running { assigned_cards: Vec<usize> }, // 任务正在运行，包含已分配的卡片
    Completed,
}

// 执行任务结构，包含分配的卡片信息
struct ExecuteTask<Rt: RuntimeType> {
    task_id: u64,
    artifect: Artifect<Rt>,
    hardware_info: HardwareInfo,
    rng: AsyncRng,
    inputs: EntryTable<Rt>,
    debug_opt: RuntimeDebug,
    resource_requirement: ResourceRequirement,
    assigned_cards: Vec<usize>,
    status: Arc<Mutex<TaskStatus>>,
    result_sender: mpsc::Sender<(
        Option<Variable<Rt>>,
        Option<DebugInfoCollector>,
        RuntimeInfo<Rt>,
    )>,
}

// 用户提交的任务结构
struct UserTask<Rt: RuntimeType> {
    task_id: u64,
    artifect: Artifect<Rt>,
    status: Arc<Mutex<TaskStatus>>,
    result_sender: mpsc::Sender<(
        Option<Variable<Rt>>,
        Option<DebugInfoCollector>,
        RuntimeInfo<Rt>,
    )>,
    hardware_info: HardwareInfo,
    rng: AsyncRng,
    inputs: EntryTable<Rt>,
    debug_opt: RuntimeDebug,
    resource_requirement: ResourceRequirement,
}

// 执行器状态
#[derive(Debug, Clone)]
struct ExecutorState {
    id: usize,
    assigned_cards: Vec<usize>,
    is_busy: bool,
}

pub struct Scheduler<Rt: RuntimeType> {
    pub cards_per_request: usize,
    pub total_cards: usize,
    task_sender: mpsc::Sender<UserTask<Rt>>,
    _scheduler_thread: thread::JoinHandle<()>,
    _executor_threads: Vec<thread::JoinHandle<()>>,
    resource_manager: Arc<GlobalResourceManager>,
    next_task_id: Arc<Mutex<u64>>,
}

impl<Rt: RuntimeType> Scheduler<Rt> {
    pub fn new(
        cards_per_request: usize,
        total_cards: usize,
        total_cpu_memory_mb: u64,
        total_disk_space_mb: u64,
    ) -> Self {
        assert!(
            total_cards % cards_per_request == 0,
            "Total cards must be a multiple of cards per request"
        );

        let resource_manager = Arc::new(GlobalResourceManager::new(
            total_cpu_memory_mb,
            total_disk_space_mb,
        ));

        let next_task_id = Arc::new(Mutex::new(0));

        // 创建任务接收通道（用户 -> 调度器）
        let (task_sender, task_receiver) = mpsc::channel::<UserTask<Rt>>();

        // 创建执行任务通道（调度器 -> 执行器）
        let (exec_task_sender, exec_task_receiver) =
            crossbeam_channel::unbounded::<ExecuteTask<Rt>>();

        // 创建执行器状态通道（执行器 -> 调度器）
        let (executor_status_sender, executor_status_receiver) =
            crossbeam_channel::unbounded::<(usize, bool)>();

        // 计算执行器数量（默认4个）
        let num_executors = 4;

        // 初始化执行器状态
        let mut executors = Vec::new();
        for i in 0..num_executors {
            let start_card = (i * total_cards) / num_executors;
            let end_card = ((i + 1) * total_cards) / num_executors;
            let assigned_cards: Vec<usize> = (start_card..end_card).collect();

            executors.push(ExecutorState {
                id: i,
                assigned_cards,
                is_busy: false,
            });
        }

        // 启动调度器线程
        let resource_manager_clone = Arc::clone(&resource_manager);
        let executor_status_receiver_clone = executor_status_receiver.clone();
        let scheduler_thread = thread::spawn(move || {
            let mut executors = executors;
            let mut pending_tasks = Vec::<UserTask<Rt>>::new();

            println!("调度器线程启动，管理 {} 个执行器", num_executors);

            loop {
                // 检查是否有新的用户任务
                while let Ok(user_task) = task_receiver.try_recv() {
                    println!("调度器接收到新任务 ID: {}", user_task.task_id);
                    pending_tasks.push(user_task);
                }

                // 检查执行器状态更新
                while let Ok((executor_id, is_busy)) = executor_status_receiver_clone.try_recv() {
                    if let Some(executor) = executors.iter_mut().find(|e| e.id == executor_id) {
                        executor.is_busy = is_busy;
                        if !is_busy {
                            println!(
                                "执行器 {} 已空闲，卡片: {:?}",
                                executor_id, executor.assigned_cards
                            );
                        }
                    }
                }

                // 尝试调度待处理的任务
                let mut tasks_to_remove = Vec::new();
                for (i, user_task) in pending_tasks.iter().enumerate() {
                    // 寻找空闲的执行器
                    if let Some(executor) = executors.iter().find(|e| !e.is_busy) {
                        // 检查资源是否可用
                        if resource_manager_clone.try_allocate(&user_task.resource_requirement) {
                            // 更新任务状态
                            {
                                let mut status = user_task.status.lock().unwrap();
                                *status = TaskStatus::Running {
                                    assigned_cards: executor.assigned_cards.clone(),
                                };
                            }

                            // 创建执行任务
                            let execute_task = ExecuteTask {
                                task_id: user_task.task_id,
                                artifect: user_task.artifect.clone(),
                                hardware_info: user_task.hardware_info.clone(),
                                rng: user_task.rng.clone(),
                                inputs: user_task.inputs.clone(),
                                debug_opt: user_task.debug_opt,
                                resource_requirement: user_task.resource_requirement.clone(),
                                assigned_cards: executor.assigned_cards.clone(),
                                status: Arc::clone(&user_task.status),
                                result_sender: user_task.result_sender.clone(),
                            };

                            let executor_id = executor.id;

                            // 发送任务给执行器
                            if exec_task_sender.send(execute_task).is_ok() {
                                println!(
                                    "调度器分发任务 {} 给执行器 {}，卡片: {:?}，资源需求: {}MB CPU, {}MB 磁盘",
                                    user_task.task_id,
                                    executor_id,
                                    executor.assigned_cards,
                                    user_task.resource_requirement.cpu_memory_mb,
                                    user_task.resource_requirement.disk_space_mb
                                );

                                // 标记执行器为忙碌
                                if let Some(exec) =
                                    executors.iter_mut().find(|e| e.id == executor_id)
                                {
                                    exec.is_busy = true;
                                }

                                tasks_to_remove.push(i);
                                break;
                            } else {
                                // 如果发送失败，释放资源
                                resource_manager_clone.release(&user_task.resource_requirement);
                            }
                        }
                    }
                }

                // 移除已调度的任务
                for &i in tasks_to_remove.iter().rev() {
                    pending_tasks.remove(i);
                }

                // 短暂休眠避免忙等待
                thread::sleep(Duration::from_millis(10));
            }
        });

        // 启动执行器线程
        let mut executor_threads = Vec::new();
        for i in 0..num_executors {
            let exec_task_receiver_clone = exec_task_receiver.clone();
            let executor_status_sender_clone = executor_status_sender.clone();
            let resource_manager_clone = Arc::clone(&resource_manager);

            let executor_thread = thread::spawn(move || {
                println!("执行器线程 {} 启动", i);

                while let Ok(execute_task) = exec_task_receiver_clone.recv() {
                    // 通知调度器此执行器正在忙碌
                    let _ = executor_status_sender_clone.send((i, true));

                    println!(
                        "执行器 {} 开始执行任务 {}，使用卡片: {:?}",
                        i, execute_task.task_id, execute_task.assigned_cards
                    );

                    // 创建GPU映射函数
                    let cards = execute_task.assigned_cards.clone();
                    let gpu_mapping =
                        Arc::new(move |x: i32| cards[x as usize % cards.len()] as i32);

                    // 执行任务
                    let start = std::time::Instant::now();
                    let pools = execute_task
                        .artifect
                        .create_pools(&execute_task.hardware_info, true);
                    let mut runtime = execute_task.artifect.prepare_dispatcher(
                        pools,
                        execute_task.rng,
                        gpu_mapping,
                    );

                    let (result, _) = runtime.run(&execute_task.inputs, execute_task.debug_opt);
                    let elapsed = start.elapsed();

                    // 释放资源
                    resource_manager_clone.release(&execute_task.resource_requirement);

                    // 更新任务状态为已完成
                    {
                        let mut status = execute_task.status.lock().unwrap();
                        *status = TaskStatus::Completed;
                    }

                    let (cpu_util, disk_util) = resource_manager_clone.get_utilization();
                    println!(
                        "执行器 {} 完成任务 {} 在 {:?}。资源利用率: CPU {:.1}%, 磁盘 {:.1}%",
                        i,
                        execute_task.task_id,
                        elapsed,
                        cpu_util * 100.0,
                        disk_util * 100.0
                    );

                    // 发送结果
                    if let Err(_) = execute_task.result_sender.send(result) {
                        eprintln!("执行器 {} 发送任务 {} 结果失败", i, execute_task.task_id);
                    }

                    // 通知调度器此执行器已空闲
                    let _ = executor_status_sender_clone.send((i, false));
                }
            });

            executor_threads.push(executor_thread);
        }

        Self {
            cards_per_request,
            total_cards,
            task_sender,
            _scheduler_thread: scheduler_thread,
            _executor_threads: executor_threads,
            resource_manager,
            next_task_id,
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
        mpsc::Receiver<(Option<Variable<Rt>>, Log, RuntimeInfo<Rt>)>,
    ) {
        let task_id = {
            let mut id = self.next_task_id.lock().unwrap();
            *id += 1;
            *id
        };

        let (result_sender, result_receiver) = mpsc::channel();
        let status = Arc::new(Mutex::new(TaskStatus::Pending));

        let user_task = UserTask {
            task_id,
            artifect: request,
            status: Arc::clone(&status),
            result_sender,
            hardware_info: hd_info,
            rng,
            inputs,
            debug_opt,
            resource_requirement,
        };

        self.task_sender
            .send(user_task)
            .expect("调度器已关闭，无法发送任务");

        println!("用户提交任务 ID: {}", task_id);
        (status, result_receiver)
    }

    // 获取当前资源状态
    pub fn get_resource_status(&self) -> (u64, u64, f64, f64) {
        let (available_cpu, available_disk) = self.resource_manager.get_available_resources();
        let (cpu_util, disk_util) = self.resource_manager.get_utilization();
        (available_cpu, available_disk, cpu_util, disk_util)
    }

    // 预估任务是否能够立即执行（不保证一定能执行，因为资源状态可能变化）
    pub fn can_execute_immediately(&self, requirement: &ResourceRequirement) -> bool {
        let (available_cpu, available_disk) = self.resource_manager.get_available_resources();
        available_cpu >= requirement.cpu_memory_mb && available_disk >= requirement.disk_space_mb
    }
}
