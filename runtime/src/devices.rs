use std::sync::{mpsc::Receiver, Condvar, Mutex};
use zkpoly_common::heap;
use zkpoly_cuda_api::stream::CudaEvent;

zkpoly_common::define_usize_id!(EventId);
zkpoly_common::define_usize_id!(ThreadId);

pub type EventTable = heap::Heap<EventId, Event>;
pub type ThreadTable = heap::Heap<ThreadId, Mutex<Option<Receiver<i32>>>>;

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    CPU,
    GPU { device_id: i32 },
    Disk,
}

impl DeviceType {
    pub fn unwrap_gpu(&self) -> i32 {
        match self {
            DeviceType::GPU { device_id } => *device_id,
            _ => panic!("unwrap_gpu: not a GPU device"),
        }
    }

    pub fn is_gpu(&self) -> bool {
        match self {
            DeviceType::GPU { .. } => true,
            _ => false,
        }
    }

    pub fn is_cpu(&self) -> bool {
        match self {
            DeviceType::CPU => true,
            _ => false,
        }
    }

    pub fn is_disk(&self) -> bool {
        match self {
            DeviceType::Disk => true,
            _ => false,
        }
    }
}

pub enum EventType {
    GpuEvent,
    ThreadEvent,
}

pub enum Event {
    GpuEvent(CudaEvent),
    ThreadEvent { cond: Condvar, lock: Mutex<bool> },
}

impl Event {
    pub fn new_gpu() -> Self {
        Self::GpuEvent(CudaEvent::new())
    }

    pub fn new_thread() -> Self {
        Self::ThreadEvent {
            cond: Condvar::new(),
            lock: Mutex::new(false),
        }
    }
}

#[test]
fn test_threadpool() {
    use std::sync::mpsc::channel;
    use threadpool::ThreadPool;

    let n_workers = 4;
    let n_jobs = 8;
    let pool = ThreadPool::new(n_workers);

    let (tx, rx) = channel();
    for _ in 0..n_jobs {
        let tx = tx.clone();
        pool.execute(move || {
            println!("Hello, world!");
            tx.send(1)
                .expect("channel will be there waiting for the pool");
        });
    }

    assert_eq!(rx.iter().take(n_jobs).fold(0, |a, b| a + b), 8);
}
