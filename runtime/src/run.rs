use std::sync::{mpsc::Sender, Arc};

use group::ff::Field;
use threadpool::ThreadPool;

use crate::{
    alloc::{allocate, deallocate},
    args::{RuntimeType, Variable, VariableTable},
    devices::{DeviceType, Event, EventTable, StreamTable, ThreadTable},
    instructions::Instruction,
    poly::Polynomial,
    transport::Transport,
    typ::Typ,
};

use zkpoly_cuda_api::mem::CudaAllocator;

use zkpoly_memory_pool::PinnedMemoryPool;

#[macro_export]
macro_rules! only_cpu {
    ($device:expr) => {
        match $device {
            DeviceType::CPU => {}
            _ => panic!("only cpu device can do this"),
        }
    };
}

pub struct RuntimeInfo<T: RuntimeType> {
    pub variable: Arc<VariableTable<T>>,
    pub pool: Arc<ThreadPool>,
    pub streams: Arc<StreamTable>,
    pub events: Arc<EventTable>,
    pub threads: Arc<ThreadTable>,
    pub main_thread: bool,
}

pub fn run<T: RuntimeType>(
    instructions: Vec<Instruction>,
    info: RuntimeInfo<T>,
    mem_allocator: Option<PinnedMemoryPool>,
    gpu_allocator: Option<Vec<CudaAllocator>>,
    epilogue: Option<Sender<i32>>,
) {
    for instruction in instructions {
        match instruction {
            Instruction::Allocate {
                device,
                typ,
                id,
                offset,
            } => {
                // only main thread can allocate memory
                assert!(info.main_thread);
                allocate::<T>(
                    &info,
                    device,
                    typ,
                    id,
                    offset,
                    &mem_allocator,
                    &gpu_allocator,
                );
            }
            Instruction::Deallocate { id } => {
                // only main thread can deallocate memory
                assert!(info.main_thread);
                deallocate::<T>(&info, id, &mem_allocator);
            }
            Instruction::Transfer {
                src_device,
                dst_device,
                stream,
                src_id,
                dst_id,
            } => {
                let src_guard = info.variable[src_id].read().unwrap();
                let mut dst_guard = info.variable[dst_id].write().unwrap();
                let src = src_guard.as_ref().unwrap();
                let dst = dst_guard.as_mut().unwrap();
                match src_device {
                    DeviceType::CPU => match dst_device {
                        DeviceType::CPU => match src {
                            Variable::Poly(src) => {
                                if let Variable::Poly(dst) = dst {
                                    src.cpu2cpu(dst);
                                } else {
                                    panic!("cannot transfer to non-poly variable");
                                }
                            }
                            _ => todo!(),
                        },
                        DeviceType::GPU { .. } => match src {
                            Variable::Poly(src) => {
                                if let Variable::Poly(dst) = dst {
                                    src.cpu2gpu(dst, &info.streams[stream.unwrap()]);
                                } else {
                                    panic!("cannot transfer to non-poly variable");
                                }
                            }
                            _ => todo!(),
                        },
                        DeviceType::Disk => todo!(),
                    },
                    DeviceType::GPU { .. } => match dst_device {
                        DeviceType::CPU => match src {
                            Variable::Poly(src) => {
                                if let Variable::Poly(dst) = dst {
                                    src.gpu2cpu(dst, &info.streams[stream.unwrap()]);
                                } else {
                                    panic!("cannot transfer to non-poly variable");
                                }
                            }
                            _ => todo!(),
                        },
                        DeviceType::GPU { .. } => match src {
                            Variable::Poly(src) => {
                                if let Variable::Poly(dst) = dst {
                                    src.gpu2gpu(dst, &info.streams[stream.unwrap()]);
                                } else {
                                    panic!("cannot transfer to non-poly variable");
                                }
                            }
                            _ => todo!(),
                        },
                        DeviceType::Disk => todo!(),
                    },
                    DeviceType::Disk => todo!(),
                }
            }
            Instruction::FuncCall {
                device,
                stream,
                func_id,
                arg_ids,
            } => {
                // let func =
            }
            Instruction::Wait {
                slave,
                stream,
                event,
            } => {
                let ref event = info.events[event];
                match event {
                    Event::GpuEvent(cuda_event) => match slave {
                        DeviceType::CPU => {
                            cuda_event.sync();
                        }
                        DeviceType::GPU { .. } => {
                            let stream = &info.streams[stream.unwrap()];
                            stream.wait(cuda_event);
                        }
                        DeviceType::Disk => unreachable!(),
                    },
                    Event::ThreadEvent { cond, lock } => {
                        let mut started = lock.lock().unwrap();
                        while !*started {
                            started = cond.wait(started).unwrap();
                        }
                    }
                }
            }
            Instruction::Record { stream, event } => {
                let ref event = info.events[event];
                match event {
                    Event::GpuEvent(cuda_event) => {
                        let stream = &info.streams[stream.unwrap()];
                        stream.record(cuda_event);
                    }
                    Event::ThreadEvent { cond, lock } => {
                        let mut started = lock.lock().unwrap();
                        *started = true;
                        cond.notify_all();
                    }
                }
            }
            Instruction::Fork {
                new_thread,
                instructions,
            } => {
                let (tx, rx) = std::sync::mpsc::channel();
                info.threads[new_thread].lock().unwrap().replace(rx);
                let info_clone = RuntimeInfo {
                    variable: info.variable.clone(),
                    pool: info.pool.clone(),
                    streams: info.streams.clone(),
                    events: info.events.clone(),
                    threads: info.threads.clone(),
                    main_thread: false,
                };
                info.pool.execute(move || {
                    run::<T>(instructions, info_clone, None, None, Some(tx));
                });
            }
            Instruction::Join { thread } => {
                let rx = info.threads[thread].lock().unwrap().take().unwrap();
                rx.recv().unwrap();
            }
        }
    }
    if !info.main_thread {
        epilogue
            .unwrap()
            .send(1)
            .expect("channel will be there waiting for the pool");
    }
}
