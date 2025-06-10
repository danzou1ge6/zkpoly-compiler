use std::sync::{Arc, Mutex, Condvar, mpsc};
use std::collections::VecDeque;
use std::thread;
use crate::runtime::RuntimeDebug;
use crate::{args::{RuntimeType, Variable}, runtime::{Runtime, RuntimeInfo}};

// 任务状态
#[derive(Debug)]
enum TaskStatus {
    Pending,
    Running,
    Completed,
}

// 任务结构
struct Task<Rt: RuntimeType> {
    runtime: Runtime<Rt>,
    status: TaskStatus,
    assigned_cards: Vec<usize>,
    result_sender: mpsc::Sender<Option<Variable<Rt>>>, // 用于发送任务结果
}

pub struct Scheduler<Rt: RuntimeType> {
    pub cards_per_request: usize,
    pub total_cards: usize,
    tasks: Arc<Mutex<VecDeque<Task<Rt>>>>,
    available_cards: Arc<Mutex<Vec<usize>>>,
    condvar: Arc<Condvar>,
    scheduler_thread: Option<thread::JoinHandle<()>>,
}

impl<Rt: RuntimeType> Scheduler<Rt> {
    pub fn new(cards_per_request: usize, total_cards: usize) -> Self {
        let mut available_cards = Vec::with_capacity(total_cards);
        for i in 0..total_cards {
            available_cards.push(i);
        }

        let tasks = Arc::new(Mutex::new(VecDeque::<Task<Rt>>::new()));
        let available_cards = Arc::new(Mutex::new(available_cards));
        let condvar = Arc::new(Condvar::new());

        let tasks_clone = Arc::clone(&tasks);
        let available_cards_clone = Arc::clone(&available_cards);
        let condvar_clone = Arc::clone(&condvar);
        let cards_per_request = cards_per_request;

        // 创建调度器线程
        let scheduler_thread = thread::spawn(move || {
            loop {
                let mut tasks = tasks_clone.lock().unwrap();
                
                // 等待任务队列非空
                while tasks.is_empty() {
                    tasks = condvar_clone.wait(tasks).unwrap();
                }

                // 检查队首任务
                if let Some(task) = tasks.front_mut() {
                    let mut available = available_cards_clone.lock().unwrap();
                    
                    // 如果有足够的卡可用
                    if available.len() >= cards_per_request {
                        // 分配GPU卡
                        let assigned: Vec<usize> = available.drain(0..cards_per_request).collect();
                        task.assigned_cards = assigned.clone();
                        task.status = TaskStatus::Running;

                        // 取出任务准备执行
                        let task = tasks.pop_front().unwrap();
                        drop(tasks); // 释放锁让其他任务可以提交

                        // 创建debug配置
                        let debug_opt = RuntimeDebug::None;

                        // 调整GPU设备ID并执行任务
                        let mut runtime = task.runtime;
                        runtime.adjust_gpu_device_ids(assigned[0] as i32);
                        
                        // 执行Runtime获取结果
                        let mut input_table = EntryTable::new(); // 根据实际需求初始化
                        let (result, _) = runtime.run(&mut input_table, debug_opt);
                        
                        // 发送结果
                        task.result_sender.send(result).unwrap();

                        // 归还GPU卡
                        let mut available = available_cards_clone.lock().unwrap();
                        available.extend(assigned);
                        drop(available);
                        
                        // 通知有新的可用资源
                        condvar_clone.notify_one();
                    }
                }
            }
        });

        Self {
            cards_per_request,
            total_cards,
            tasks,
            available_cards,
            condvar,
            scheduler_thread: Some(scheduler_thread),
        }
    }

    pub fn add_request(&mut self, request: Runtime<Rt>) -> mpsc::Receiver<Option<Variable<Rt>>> {
        // 创建channel用于接收任务结果
        let (sender, receiver) = mpsc::channel();
        
        let task = Task {
            runtime: request,
            status: TaskStatus::Pending,
            assigned_cards: Vec::new(),
            result_sender: sender,
        };

        let mut tasks = self.tasks.lock().unwrap();
        tasks.push_back(task);
        
        // 通知调度器线程有新任务
        self.condvar.notify_one();

        receiver
    }

    pub fn return_cards(&mut self, card_ids: Vec<usize>) {
        let mut available = self.available_cards.lock().unwrap();
        available.extend(card_ids);
        
        // 通知调度器线程有新的可用资源
        self.condvar.notify_one();
    }
}

impl<Rt: RuntimeType> Drop for Scheduler<Rt> {
    fn drop(&mut self) {
        // 清理资源
        if let Some(handle) = self.scheduler_thread.take() {
            // TODO: 实现优雅退出
            // 比如设置一个退出标志，等待线程完成当前任务
        }
    }
}