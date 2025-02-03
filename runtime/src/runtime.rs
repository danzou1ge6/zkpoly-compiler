use std::sync::{mpsc::Sender, Arc};

use threadpool::ThreadPool;

use zkpoly_common::load_dynamic::Libs;

use crate::{
    args::{RuntimeType, Variable, VariableTable},
    devices::{DeviceType, Event, EventTable, ThreadTable},
    functions::{
        FunctionTable,
        FunctionValue::{Fn, FnMut, FnOnce},
    },
    instructions::Instruction,
};

use zkpoly_cuda_api::mem::CudaAllocator;

use zkpoly_memory_pool::PinnedMemoryPool;

pub mod alloc;
pub mod transfer;

pub struct Runtime<T: RuntimeType> {
    instructions: Vec<Instruction>,
    variable: VariableTable<T>,
    pool: ThreadPool,
    funcs: FunctionTable<T>,
    events: EventTable,
    threads: ThreadTable,
    mem_allocator: PinnedMemoryPool,
    gpu_allocator: Vec<CudaAllocator>,
    _libs: Libs,
}

impl<T: RuntimeType> Runtime<T> {
    pub fn new(
        instructions: Vec<Instruction>,
        variable: VariableTable<T>,
        pool: ThreadPool,
        funcs: FunctionTable<T>,
        events: EventTable,
        threads: ThreadTable,
        mem_allocator: PinnedMemoryPool,
        gpu_allocator: Vec<CudaAllocator>,
        libs: Libs,
    ) -> Self {
        Self {
            instructions,
            variable,
            pool,
            funcs,
            events,
            threads,
            mem_allocator,
            gpu_allocator,
            _libs: libs,
        }
    }
    pub fn run(self) -> RuntimeInfo<T> {
        let info = RuntimeInfo {
            variable: Arc::new(self.variable),
            pool: Arc::new(self.pool),
            funcs: Arc::new(self.funcs),
            events: Arc::new(self.events),
            threads: Arc::new(self.threads),
            main_thread: true,
        };
        info.run(
            self.instructions,
            Some(self.mem_allocator),
            Some(self.gpu_allocator),
            None,
        );
        info
    }
}

#[derive(Clone)]
pub struct RuntimeInfo<T: RuntimeType> {
    pub variable: Arc<VariableTable<T>>,
    pub pool: Arc<ThreadPool>,
    pub funcs: Arc<FunctionTable<T>>,
    pub events: Arc<EventTable>,
    pub threads: Arc<ThreadTable>,
    pub main_thread: bool,
}

impl<T: RuntimeType> RuntimeInfo<T> {
    pub fn run(
        &self,
        instructions: Vec<Instruction>,
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
                    assert!(self.main_thread);
                    let mut guard = self.variable[id].write().unwrap();
                    assert!(guard.is_none());
                    *guard =
                        Some(self.allocate(device, typ, offset, &mem_allocator, &gpu_allocator));
                }
                Instruction::Deallocate { id } => {
                    // only main thread can deallocate memory
                    assert!(self.main_thread);
                    let mut guard = self.variable[id].write().unwrap();
                    if let Some(var) = guard.as_mut() {
                        self.deallocate(var, &mem_allocator);
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
                    let src_guard = self.variable[src_id].read().unwrap();
                    let mut dst_guard = self.variable[dst_id].write().unwrap();
                    let src: &Variable<T> = src_guard.as_ref().unwrap();
                    let dst: &mut Variable<T> = dst_guard.as_mut().unwrap();
                    self.transfer(src, dst, src_device, dst_device, stream);
                }
                Instruction::FuncCall {
                    func_id,
                    arg_mut,
                    arg,
                } => {
                    let mut arg_mut_guards: Vec<_> = arg_mut
                        .iter()
                        .map(|id| self.variable[*id].write().unwrap())
                        .collect();
                    let arg_guards: Vec<_> = arg
                        .iter()
                        .map(|id| self.variable[*id].read().unwrap())
                        .collect();

                    let args_mut: Vec<_> = arg_mut_guards
                        .iter_mut()
                        .map(|guard| guard.as_mut().unwrap())
                        .collect();
                    let args: Vec<_> = arg_guards
                        .iter()
                        .map(|guard| guard.as_ref().unwrap())
                        .collect();

                    let ref target = self.funcs[func_id].f;
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
                    let ref event = self.events[event];
                    match event {
                        Event::GpuEvent(cuda_event) => match slave {
                            DeviceType::CPU => {
                                cuda_event.sync();
                            }
                            DeviceType::GPU { .. } => {
                                let stream_guard = self.variable[stream.unwrap()].read().unwrap();
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
                    let ref event = self.events[event];
                    match event {
                        Event::GpuEvent(cuda_event) => {
                            let stream_guard = self.variable[stream.unwrap()].read().unwrap();
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
                    self.threads[new_thread].lock().unwrap().replace(rx);
                    let mut sub_info = self.clone();
                    sub_info.main_thread = false;
                    self.pool.execute(move || {
                        sub_info.run(instructions, None, None, Some(tx));
                    });
                }
                Instruction::Join { thread } => {
                    let rx = self.threads[thread].lock().unwrap().take().unwrap();
                    rx.recv().unwrap();
                }
                Instruction::Rotation { id, shift } => {
                    let mut guard = self.variable[id].write().unwrap();
                    let poly = guard.as_mut().unwrap().unwrap_scalar_array_mut();
                    poly.rotate(shift);
                }
            }
        }
        if !self.main_thread {
            epilogue
                .unwrap()
                .send(1)
                .expect("channel will be there waiting for the pool");
        }
    }
}
