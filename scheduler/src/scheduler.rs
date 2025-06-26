use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;
use zkpoly_compiler::driver::artifect::Pools;
use zkpoly_compiler::driver::{Artifect, HardwareInfo};
use zkpoly_memory_pool::static_allocator::CpuStaticAllocator;
use zkpoly_runtime::args::{EntryTable, RuntimeType, Variable};
use zkpoly_runtime::async_rng::AsyncRng;
use zkpoly_runtime::debug::Log;
use zkpoly_runtime::runtime::{RuntimeDebug, RuntimeInfo};

// 资源需求结构体
#[derive(Debug, Clone)]
pub struct ResourceRequirement {
    pub cpu_memory_mb: u64,    // CPU内存需求 (MB)
    pub disk_space_mb: u64,    // 磁盘空间需求 (MB)
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

// 任务结构
struct Task<Rt: RuntimeType> {
    artifect: Artifect<Rt>,
    status: Arc<Mutex<TaskStatus>>,
    result_sender: mpsc::Sender<(Option<Variable<Rt>>, Log, RuntimeInfo<Rt>)>, // 用于发送任务结果
    hardware_info: HardwareInfo,
    rng: AsyncRng,
    inputs: EntryTable<Rt>,
    debug_opt: RuntimeDebug,
    resource_requirement: ResourceRequirement, // 任务的资源需求
}

pub struct Scheduler<Rt: RuntimeType> {
    pub cards_per_request: usize,
    pub total_cards: usize,
    sender: crossbeam_channel::Sender<Task<Rt>>,
    _scheduler_threads: Vec<thread::JoinHandle<()>>,
    resource_manager: Arc<GlobalResourceManager>,
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

        let (sender, receiver) = crossbeam_channel::unbounded();
        let mut scheduler_threads = Vec::new();

        for i in (0..total_cards).step_by(cards_per_request) {
            let receiver: crossbeam_channel::Receiver<Task<Rt>> = receiver.clone();
            let resource_manager_clone = Arc::clone(&resource_manager);
            let cur_cards = (i..(i + cards_per_request))
                .into_iter()
                .collect::<Vec<usize>>();
            scheduler_threads.push(thread::spawn(move || {
                let cur_cards_clone = cur_cards.clone();
                let gpu_mapping = Arc::new(move |x: i32| cur_cards_clone[x as usize] as i32);
                while let Ok(task) = receiver.recv() {
                    // 等待资源可用
                    while !resource_manager_clone.try_allocate(&task.resource_requirement) {
                        println!(
                            "Scheduler thread {}: Waiting for resources - need {}MB CPU memory, {}MB disk space",
                            i / cards_per_request,
                            task.resource_requirement.cpu_memory_mb,
                            task.resource_requirement.disk_space_mb
                        );
                        let (available_cpu, available_disk) = resource_manager_clone.get_available_resources();
                        println!(
                            "Available resources: {}MB CPU memory, {}MB disk space",
                            available_cpu, available_disk
                        );
                        thread::sleep(Duration::from_millis(100)); // 等待100ms后重试
                    }

                    let mut status = task.status.lock().unwrap();
                    assert!(
                        matches!(*status, TaskStatus::Pending),
                        "Task must be pending before running"
                    );
                    *status = TaskStatus::Running {
                        assigned_cards: cur_cards.clone(),
                    }; // 更新任务状态为运行中，并记录已分配的卡片
                    drop(status); // 释放锁

                    println!(
                        "Scheduler thread {}: Starting task with GPUs {:?}, allocated {}MB CPU memory, {}MB disk space",
                        i / cards_per_request,
                        cur_cards,
                        task.resource_requirement.cpu_memory_mb,
                        task.resource_requirement.disk_space_mb
                    );

                    let result: (Option<Variable<Rt>>, Log, RuntimeInfo<Rt>);
                    let pools = task.artifect.create_pools(&task.hardware_info, true);
                    let mut runtime = task.artifect.prepare_dispatcher(
                        pools,
                        task.rng,
                        gpu_mapping.clone(),
                    );

                    let start = std::time::Instant::now();
                    (result, _) = runtime.run(&task.inputs, task.debug_opt);
                    let elapsed = start.elapsed();

                    // 释放资源
                    resource_manager_clone.release(&task.resource_requirement);

                    // 更新任务状态为已完成
                    let mut status = task.status.lock().unwrap();
                    *status = TaskStatus::Completed;
                    drop(status); // 释放锁
                    
                    let (cpu_util, disk_util) = resource_manager_clone.get_utilization();
                    println!(
                        "Scheduler thread {} completed task with GPUs {:?} in {:?}. Resource utilization: CPU {:.1}%, Disk {:.1}%",
                        i / cards_per_request,
                        cur_cards,
                        elapsed,
                        cpu_util * 100.0,
                        disk_util * 100.0
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
            resource_manager,
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
        let (result_sender, result_receiver) = mpsc::channel();
        let status = Arc::new(Mutex::new(TaskStatus::Pending));

        let apply_utilization_ratio = |need: u64| -> u64 {
            ((need as f64) * 1.2) as u64
        };

        let resource_requirement = ResourceRequirement {
            cpu_memory_mb: request.memory_statistics().cpu_peak_usage.div_ceil(2u64.pow(20)),
            disk_space_mb: apply_utilization_ratio(request.memory_statistics().disk_peak_usage).div_ceil(2u64.pow(20)),
        };

        let task = Task {
            artifect: request,
            status: Arc::clone(&status),
            result_sender,
            hardware_info: hd_info,
            rng,
            inputs,
            debug_opt,
            resource_requirement,
        };

        self.sender
            .send(task)
            .expect("Failed to send task to scheduler");

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
