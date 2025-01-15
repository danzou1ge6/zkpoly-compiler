use std::sync::{mpsc::Sender, Arc};

use threadpool::ThreadPool;

use crate::{
    alloc::{allocate, deallocate},
    args::{RuntimeType, Variable, VariableTable},
    devices::{DeviceType, Event, EventTable, ThreadTable},
    functions::{
        FunctionTable,
        FunctionValue::{Fn, FnMut, FnOnce},
    },
    instructions::Instruction,
    transport::Transport,
};

use zkpoly_cuda_api::mem::CudaAllocator;

use zkpoly_memory_pool::PinnedMemoryPool;

pub struct RuntimeInfo<T: RuntimeType> {
    pub variable: Arc<VariableTable<T>>,
    pub pool: Arc<ThreadPool>,
    pub funcs: Arc<FunctionTable<T>>,
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
                let mut guard = info.variable[id].write().unwrap();
                assert!(guard.is_none());
                *guard = Some(allocate::<T>(
                    &info,
                    device,
                    typ,
                    offset,
                    &mem_allocator,
                    &gpu_allocator,
                ));
            }
            Instruction::Deallocate { id } => {
                // only main thread can deallocate memory
                assert!(info.main_thread);
                let mut guard = info.variable[id].write().unwrap();
                if let Some(var) = guard.as_mut() {
                    deallocate::<T>(&info, var, &mem_allocator);
                    *guard = None;
                }
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
                let src: &Variable<T> = src_guard.as_ref().unwrap();
                let dst: &mut Variable<T> = dst_guard.as_mut().unwrap();
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
                                    let stream_guard =
                                        info.variable[stream.unwrap()].read().unwrap();
                                    src.cpu2gpu(
                                        dst,
                                        stream_guard.as_ref().unwrap().unwrap_stream(),
                                    );
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
                                    let stream_guard =
                                        info.variable[stream.unwrap()].read().unwrap();
                                    src.gpu2cpu(
                                        dst,
                                        stream_guard.as_ref().unwrap().unwrap_stream(),
                                    );
                                } else {
                                    panic!("cannot transfer to non-poly variable");
                                }
                            }
                            _ => todo!(),
                        },
                        DeviceType::GPU { .. } => match src {
                            Variable::Poly(src) => {
                                if let Variable::Poly(dst) = dst {
                                    let stream_guard =
                                        info.variable[stream.unwrap()].read().unwrap();
                                    src.gpu2gpu(
                                        dst,
                                        stream_guard.as_ref().unwrap().unwrap_stream(),
                                    );
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
                func_id,
                arg_mut,
                arg,
            } => {
                let mut arg_mut_guards: Vec<_> = arg_mut
                    .iter()
                    .map(|id| info.variable[*id].write().unwrap())
                    .collect();
                let arg_guards: Vec<_> = arg
                    .iter()
                    .map(|id| info.variable[*id].read().unwrap())
                    .collect();

                let args_mut: Vec<_> = arg_mut_guards
                    .iter_mut()
                    .map(|guard| guard.as_mut().unwrap())
                    .collect();
                let args: Vec<_> = arg_guards
                    .iter()
                    .map(|guard| guard.as_ref().unwrap())
                    .collect();

                let ref target = info.funcs[func_id].f;
                match target {
                    FnOnce(mutex) => {
                        let mut f_guard = mutex.lock().unwrap();
                        let f = f_guard.take().unwrap();
                        f(args_mut, args).unwrap();
                    }
                    FnMut(mutex) => {
                        let mut f_guard = mutex.lock().unwrap();
                        f_guard(args_mut, args).unwrap();
                    }
                    Fn(f) => {
                        f(args_mut, args).unwrap();
                    }
                }
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
                            let stream_guard = info.variable[stream.unwrap()].read().unwrap();
                            stream_guard
                                .as_ref()
                                .unwrap()
                                .unwrap_stream()
                                .wait(cuda_event);
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
                        let stream_guard = info.variable[stream.unwrap()].read().unwrap();
                        stream_guard
                            .as_ref()
                            .unwrap()
                            .unwrap_stream()
                            .record(cuda_event);
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
                    funcs: info.funcs.clone(),
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
