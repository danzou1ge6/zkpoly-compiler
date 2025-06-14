
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::sync::{mpsc::Receiver, Arc};
use zkpoly_common::{cpu_event::CpuEvent, heap};
use zkpoly_cuda_api::stream::CudaEvent;

zkpoly_common::define_usize_id!(EventId);
zkpoly_common::define_usize_id!(ThreadId);

pub type EventTypeTable = heap::Heap<EventId, EventType>;
pub type EventTable = heap::Heap<EventId, Event>;
pub type ThreadTable = heap::Heap<ThreadId, Option<Receiver<i32>>>;

pub fn new_thread_table(len: usize) -> ThreadTable {
    heap::Heap::repeat_with(|| (None), len)
}

#[derive(Clone, Serialize, Deserialize)]
pub enum EventType {
    GpuEvent(i32),
    ThreadEvent,
}

impl EventType {
    pub fn new_gpu(device_id: i32) -> Self {
        Self::GpuEvent(device_id)
    }

    pub fn new_thread() -> Self {
        Self::ThreadEvent
    }
}

pub fn instantizate_event_table(
    ett: EventTypeTable,
    gpu_mapping: Arc<dyn Fn(i32) -> i32>,
) -> EventTable {
    ett.map(&mut |_, mut typ| {
        if let EventType::GpuEvent(device_id) = &mut typ {
            *device_id = gpu_mapping(*device_id)
        }
        Event::new_from_typ(typ)
    })
}

pub enum Event {
    GpuEvent(CudaEvent),
    ThreadEvent(CpuEvent),
}

impl Event {
    pub fn typ(&self) -> EventType {
        match self {
            Event::GpuEvent(event) => EventType::GpuEvent(event.get_device()),
            Event::ThreadEvent(_) => EventType::ThreadEvent,
        }
    }

    pub fn new_from_typ(typ: EventType) -> Self {
        match typ {
            EventType::GpuEvent(device_id) => Self::GpuEvent(CudaEvent::new(device_id)),
            EventType::ThreadEvent => Self::ThreadEvent(CpuEvent::new()),
        }
    }
}

impl Serialize for Event {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.typ().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Event {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let typ = EventType::deserialize(deserializer)?;
        Ok(Event::new_from_typ(typ))
    }
}

#[derive(Clone, Debug)]
struct SafePtr<T> {
    ptr: *const T,
}

unsafe impl<T> Send for SafePtr<T> {}
unsafe impl<T> Sync for SafePtr<T> {}

#[derive(Clone, Debug)]
struct SafePtrMut<T> {
    ptr: *mut T,
}

unsafe impl<T> Send for SafePtrMut<T> {}
unsafe impl<T> Sync for SafePtrMut<T> {}

pub fn write_to_disk(ptr: *const u8, disk_pos: &Vec<(i32, usize)>, size: usize) {
    let safe_ptr = SafePtr { ptr };
    let part_size = size / disk_pos.len();
    disk_pos
        .par_iter()
        .enumerate()
        .for_each(|(disk_id, (fd, offset))| {
            let safe_ptr_clone = safe_ptr.clone();
            let ptr = safe_ptr_clone.ptr;
            let offset = offset.clone();
            let fd = fd.clone();
            let cpu_offset = part_size * disk_id;

            unsafe {
                let res = libc::pwrite(fd, ptr.add(cpu_offset).cast(), part_size, offset as i64);
                if res < part_size as isize {
                    panic!(
                        "Failed to write {} bytes to disk {} at offset {}: {}",
                        part_size, disk_id, offset, res
                    );
                }
            }
        });
}

pub fn read_from_disk(ptr: *mut u8, disk_pos: &Vec<(i32, usize)>, size: usize) {
    let safe_ptr = SafePtrMut { ptr };
    let part_size = size / disk_pos.len();

    disk_pos
        .par_iter()
        .enumerate()
        .for_each(|(disk_id, (fd, offset))| {
            let safe_ptr_clone = safe_ptr.clone();
            let ptr = safe_ptr_clone.ptr;
            let (fd, offset) = (fd.clone(), offset.clone());
            let cpu_offset = part_size * disk_id;

            unsafe {
                let res = libc::pread(fd, ptr.add(cpu_offset).cast(), part_size, offset as i64);
                if res < part_size as isize {
                    panic!(
                        "Failed to read {} bytes from disk {} at offset {}: {}",
                        part_size, disk_id, offset, res
                    );
                }
            }
        });
}
