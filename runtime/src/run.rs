use std::sync::{mpsc::Sender, Arc, Condvar};

use group::ff::Field;
use threadpool::ThreadPool;

use crate::{
    args::{Variable, VariableTable},
    devices::{DeviceType, Event, EventTable, StreamTable, ThreadTable},
    instructions::Instruction,
    poly::Polynomial,
    transport::Transport,
    typ::Typ,
};

use zkpoly_cuda_api::mem::{cudaError_cudaErrorECCUncorrectable, CudaAllocator};

use zkpoly_memory_pool::PinnedMemoryPool;

macro_rules! only_cpu {
    ($device:expr) => {
        match $device {
            DeviceType::CPU => {}
            _ => panic!("only cpu device can do this"),
        }
    };
}

pub fn run<F: Field>(
    instructions: Vec<Instruction>,
    variable: Arc<VariableTable>,
    pool: Arc<ThreadPool>,
    streams: Arc<StreamTable>,
    events: Arc<EventTable>,
    threads: Arc<ThreadTable>,
    sender: Option<Sender<i32>>,
    mem_allocator: Option<PinnedMemoryPool>,
    gpu_allocator: Option<Vec<CudaAllocator>>,
    main_thread: bool,
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
                assert!(main_thread);
                let mut target = variable[id].write().unwrap();
                match typ {
                    Typ::Poly {
                        typ: ref poly_typ,
                        log_n,
                    } => {
                        let poly = match device {
                            DeviceType::CPU => Polynomial::<F>::new(
                                poly_typ.clone(),
                                log_n,
                                mem_allocator.as_ref().unwrap().allocate(log_n.clone()),
                                device.clone(),
                            ),
                            DeviceType::GPU { device_id } => Polynomial::<F>::new(
                                poly_typ.clone(),
                                log_n,
                                gpu_allocator.as_ref().unwrap()[device_id as usize]
                                    .allocate(offset.unwrap()),
                                device.clone(),
                            ),
                            DeviceType::Disk => todo!(),
                        };
                        *target = Some(Variable::new(typ, device, Box::new(poly)));
                    }
                    Typ::Scalar => {
                        only_cpu!(device.clone());
                        *target = Some(Variable::new(typ, device, Box::new(F::ZERO)));
                    }
                    Typ::Transcript => todo!(),
                    Typ::Point => todo!(),
                    Typ::Tuple(vec) => todo!(),
                    Typ::Array(typ, _) => todo!(),
                    Typ::Any(type_id, _) => todo!(),
                };
            }
            Instruction::Deallocate { id } => {
                // only main thread can deallocate memory
                assert!(main_thread);
                let mut target = variable[id].write().unwrap();
                if let Some(var) = target.as_ref() {
                    match var.device {
                        DeviceType::CPU => match var.typ {
                            Typ::Poly { .. } => {
                                let poly =
                                    var.value.as_ref().downcast_ref::<Polynomial<F>>().unwrap();
                                mem_allocator.as_ref().unwrap().deallocate(poly.values);
                                *target = None;
                            }
                            _ => {
                                *target = None;
                            }
                        },
                        DeviceType::GPU { .. } => {
                            *target = None;
                        }
                        DeviceType::Disk => todo!(),
                    }
                }
            }
            Instruction::Transfer {
                src_device,
                dst_device,
                stream,
                src_id,
                dst_id,
            } => {
                let src_guard = variable[src_id].read().unwrap();
                let mut dst_guard = variable[dst_id].write().unwrap();
                let src = src_guard.as_ref().unwrap();
                let dst = dst_guard.as_mut().unwrap();
                match src_device {
                    DeviceType::CPU => match dst_device {
                        DeviceType::CPU => match &src.typ {
                            Typ::Poly { .. } => {
                                src.value
                                    .as_ref()
                                    .downcast_ref::<Polynomial<F>>()
                                    .unwrap()
                                    .cpu2cpu(dst.value.as_mut().downcast_mut().unwrap());
                            }
                            _ => todo!(),
                        },
                        DeviceType::GPU { .. } => match &src.typ {
                            Typ::Poly { .. } => {
                                src.value
                                    .as_ref()
                                    .downcast_ref::<Polynomial<F>>()
                                    .unwrap()
                                    .cpu2gpu(
                                        dst.value.as_mut().downcast_mut().unwrap(),
                                        &streams[stream.unwrap()],
                                    );
                            }
                            _ => todo!(),
                        },
                        DeviceType::Disk => todo!(),
                    },
                    DeviceType::GPU { .. } => match dst_device {
                        DeviceType::CPU => match &src.typ {
                            Typ::Poly { .. } => {
                                src.value
                                    .as_ref()
                                    .downcast_ref::<Polynomial<F>>()
                                    .unwrap()
                                    .gpu2cpu(
                                        dst.value.as_mut().downcast_mut().unwrap(),
                                        &streams[stream.unwrap()],
                                    );
                            }
                            _ => todo!(),
                        },
                        DeviceType::GPU { .. } => match &src.typ {
                            Typ::Poly { .. } => {
                                src.value
                                    .as_ref()
                                    .downcast_ref::<Polynomial<F>>()
                                    .unwrap()
                                    .gpu2gpu(
                                        dst.value.as_mut().downcast_mut().unwrap(),
                                        &streams[stream.unwrap()],
                                    );
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
                todo!()
            }
            Instruction::Wait {
                slave,
                stream,
                event,
            } => {
                let ref event = events[event];
                match event {
                    Event::GpuEvent(cuda_event) => match slave {
                        DeviceType::CPU => {
                            cuda_event.sync();
                        }
                        DeviceType::GPU { .. } => {
                            let stream = &streams[stream.unwrap()];
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
                let ref event = events[event];
                match event {
                    Event::GpuEvent(cuda_event) => {
                        let stream = &streams[stream.unwrap()];
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
                threads[new_thread].lock().unwrap().replace(rx);
                let (variable_clone, pool_clone, stream_clone, events_clone, threads_clone) = (
                    variable.clone(),
                    pool.clone(),
                    streams.clone(),
                    events.clone(),
                    threads.clone(),
                );
                pool.execute(move || {
                    run::<F>(
                        instructions,
                        variable_clone,
                        pool_clone,
                        stream_clone,
                        events_clone,
                        threads_clone,
                        Some(tx),
                        None,
                        None,
                        false,
                    );
                });
            }
            Instruction::Join { thread } => {
                let rx = threads[thread].lock().unwrap().take().unwrap();
                rx.recv().unwrap();
            }
        }
    }
    if !main_thread {
        sender
            .unwrap()
            .send(1)
            .expect("channel will be there waiting for the pool");
    }
}
