use std::{
    sync::{mpsc::Sender, Arc},
    thread,
    time::Instant,
};

pub use threadpool::ThreadPool;

use zkpoly_common::load_dynamic::Libs;

use crate::{
    args::{new_variable_table, ConstantTable, EntryTable, RuntimeType, Variable, VariableTable},
    async_rng::AsyncRng,
    devices::{new_thread_table, DeviceType, Event, EventTable, ThreadTable},
    functions::{
        FunctionTable,
        FunctionValue::{Fn, FnMut, FnOnce},
    },
    instructions::Instruction,
};

use zkpoly_cuda_api::{
    bindings::{cudaDeviceSynchronize, cudaGetErrorString, cudaError_cudaSuccess},
    cuda_check,
    mem::CudaAllocator,
};

use zkpoly_memory_pool::PinnedMemoryPool;

pub mod alloc;
pub mod assert_eq;
pub mod transfer;

pub struct Runtime<T: RuntimeType> {
    instructions: Vec<Instruction>,
    variable: VariableTable<T>,
    constant: ConstantTable<T>,
    inputs: EntryTable<T>,
    pool: ThreadPool,
    funcs: FunctionTable<T>,
    events: EventTable,
    threads: ThreadTable,
    mem_allocator: PinnedMemoryPool,
    gpu_allocator: Vec<CudaAllocator>,
    rng: AsyncRng,
    _libs: Libs,
}

impl<T: RuntimeType> Runtime<T> {
    pub fn with_variables(mut self, variable: VariableTable<T>) -> Self {
        self.variable = variable;
        self
    }
    pub fn new(
        instructions: Vec<Instruction>,
        n_variables: usize,
        constant: ConstantTable<T>,
        inputs: EntryTable<T>,
        funcs: FunctionTable<T>,
        pool: ThreadPool,
        events: EventTable,
        n_threads: usize,
        mem_allocator: PinnedMemoryPool,
        gpu_allocator: Vec<CudaAllocator>,
        rng: AsyncRng,
        libs: Libs,
    ) -> Self {
        Self {
            instructions,
            variable: new_variable_table(n_variables),
            constant,
            inputs,
            pool,
            funcs,
            events,
            threads: new_thread_table(n_threads),
            mem_allocator,
            gpu_allocator,
            rng,
            _libs: libs,
        }
    }
    pub fn run(self) -> (Option<Variable<T>>, RuntimeInfo<T>) {
        let info = RuntimeInfo {
            variable: Arc::new(self.variable),
            constant: Arc::new(self.constant),
            inputs: Arc::new(self.inputs),
            pool: Arc::new(self.pool),
            funcs: Arc::new(self.funcs),
            events: Arc::new(self.events),
            threads: Arc::new(self.threads),
            rng: self.rng,
            main_thread: true,
            bench_start: Some(Instant::now()),
        };
        self.mem_allocator.preallocate(30);
        let r = info.run(
            self.instructions,
            Some(self.mem_allocator),
            Some(self.gpu_allocator),
            None,
            0,
            Arc::new(std::sync::Mutex::new(())),
        );
        (r, info)
    }
}

#[derive(Clone)]
pub struct RuntimeInfo<T: RuntimeType> {
    pub variable: Arc<VariableTable<T>>,
    pub constant: Arc<ConstantTable<T>>,
    pub inputs: Arc<EntryTable<T>>,
    pub pool: Arc<ThreadPool>,
    pub funcs: Arc<FunctionTable<T>>,
    pub events: Arc<EventTable>,
    pub threads: Arc<ThreadTable>,
    pub rng: AsyncRng,
    pub main_thread: bool,
    pub bench_start: Option<Instant>,
}

impl<T: RuntimeType> RuntimeInfo<T> {
    pub fn run(
        &self,
        instructions: Vec<Instruction>,
        mem_allocator: Option<PinnedMemoryPool>,
        gpu_allocator: Option<Vec<CudaAllocator>>,
        epilogue: Option<Sender<i32>>,
        _thread_id: usize,
        global_mutex: Arc<std::sync::Mutex<()>>,
    ) -> Option<Variable<T>> {
        for (i, instruction) in instructions.into_iter().enumerate() {
            // if thread_id == 3 {
            //     println!("variable13: {:?}", self.variable[13.into()].read().unwrap());
            // }
            let _guard: Option<std::sync::MutexGuard<'_, ()>> = if self.bench_start.is_some() {
                Some(global_mutex.lock().unwrap())
            } else {
                None
            };
            // let _guard = global_mutex.lock().unwrap();
            // println!("instruction: {:?}, thread_id {:?}", instruction, _thread_id);
            let (start_time, instruct_copy) = if self.bench_start.is_some() {
                (Some(Instant::now()), Some(instruction.clone()))
            } else {
                (None, None)
            };
            let mut function_name = None;
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
                        self.deallocate(var, id, &mem_allocator);
                        *guard = None;
                    } else {
                        panic!("deallocate a non-existing variable");
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
                    // dbg!("FuncCall", func_id, arg_mut.clone(), arg.clone());
                    let arg_holder = arg
                        .iter()
                        .map(|id| self.variable[*id].read().unwrap().clone())
                        .collect::<Vec<_>>();
                    let mut arg_mut_holder = arg_mut
                        .iter()
                        .map(|id| self.variable[*id].write().unwrap().clone())
                        .collect::<Vec<_>>();

                    let args_mut: Vec<_> = arg_mut_holder
                        .iter_mut()
                        .map(|guard| guard.as_mut().unwrap())
                        .collect();
                    let args: Vec<_> = arg_holder
                        .iter()
                        .map(|guard| guard.as_ref().unwrap())
                        .collect();

                    function_name = Some(self.funcs[func_id].meta.name.clone());
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
                    event: event_id,
                } => {
                    // println!("waiting for event{:?}", event_id);
                    if _guard.is_some() {
                        drop(_guard);
                    }
                    let ref event = self.events[event_id];
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
                        Event::ThreadEvent(cpu_event) => {
                            cpu_event.wait();
                        }
                    }
                    // println!("event{:?} done", event_id);
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
                        Event::ThreadEvent(cpu_event) => {
                            cpu_event.notify();
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
                    let global_mutex = global_mutex.clone();
                    thread::spawn(move || {
                        sub_info.run(
                            instructions,
                            None,
                            None,
                            Some(tx),
                            new_thread.into(),
                            global_mutex,
                        );
                    });
                }
                Instruction::Join { thread } => {
                    let rx = self.threads[thread].lock().unwrap().take().unwrap();
                    rx.recv().unwrap();
                }
                Instruction::Rotation { src, dst, shift } => {
                    let guard = self.variable[src].read().unwrap();
                    let mut poly = guard.as_ref().unwrap().unwrap_scalar_array().clone();
                    drop(guard);
                    poly.rotate(shift);
                    let mut guard = self.variable[dst].write().unwrap();
                    assert!(guard.is_none());
                    *guard = Some(Variable::ScalarArray(poly));
                }
                Instruction::Slice {
                    src,
                    dst,
                    start,
                    end,
                } => {
                    let src_guard = self.variable[src].read().unwrap();
                    let slice = src_guard
                        .as_ref()
                        .unwrap()
                        .unwrap_scalar_array()
                        .slice(start, end);
                    drop(src_guard);
                    let mut dst_guard = self.variable[dst].write().unwrap();
                    assert!(dst_guard.is_none());
                    *dst_guard = Some(Variable::ScalarArray(slice));
                }
                Instruction::LoadConstant { src, dst } => {
                    let constant = self.constant[src].clone();
                    let mut guard = self.variable[dst].write().unwrap();
                    assert!(guard.is_none());
                    *guard = Some(constant.value);
                }
                Instruction::AssembleTuple { vars, dst } => {
                    let mut assemble = Vec::new();
                    for var in vars.iter() {
                        let guard = self.variable[*var].read().unwrap();
                        assemble.push((*guard.as_ref().unwrap()).clone());
                    }
                    let mut guard = self.variable[dst].write().unwrap();
                    assert!(guard.is_none());
                    *guard = Some(Variable::Tuple(assemble));
                }
                Instruction::RemoveRegister { id } => {
                    let mut guard = self.variable[id].write().unwrap();
                    assert!(guard.is_some());
                    *guard = None;
                }
                Instruction::Blind { dst, start, end } => {
                    let mut guard = self.variable[dst].write().unwrap();
                    let poly = guard.as_mut().unwrap().unwrap_scalar_array_mut();
                    poly.blind(start, end, self.rng.clone());
                }
                Instruction::Return(var_id) => {
                    if !self.main_thread {
                        panic!("can only return from main thread");
                    }
                    let var = self.variable[var_id].write().unwrap().take().unwrap();
                    return Some(var);
                }
                Instruction::SetSliceMeta {
                    src,
                    dst,
                    offset,
                    len,
                } => {
                    let src_guard = self.variable[src].read().unwrap();
                    let poly = src_guard
                        .as_ref()
                        .unwrap()
                        .unwrap_scalar_array()
                        .set_slice_raw(offset, len);
                    if poly.is_none() {
                        panic!(
                            "set_slice_raw failed at thread {:?}, instruction {:?}",
                            _thread_id, i
                        );
                    }
                    drop(src_guard);
                    // println!("set dst {:?} meta to {:?}", dst.clone(), poly.clone());
                    let mut dst_guard = self.variable[dst].write().unwrap();
                    assert!(dst_guard.is_none());
                    *dst_guard = Some(Variable::ScalarArray(poly.unwrap()));
                }
                Instruction::GetScalarFromArray {
                    src,
                    dst,
                    idx,
                    stream,
                } => {
                    assert_ne!(src, dst);
                    let src_guard = self.variable[src].read().unwrap();
                    let mut dst_guard = self.variable[dst].write().unwrap();
                    let poly = src_guard.as_ref().unwrap().unwrap_scalar_array();
                    let scalar = dst_guard.as_mut().unwrap().unwrap_scalar_mut();
                    assert_eq!(poly.device, scalar.device);
                    match poly.device {
                        DeviceType::CPU => {
                            *scalar.as_mut() = poly[idx];
                        }
                        DeviceType::GPU { device_id } => {
                            let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                            let stream = stream_guard.as_ref().unwrap().unwrap_stream();
                            assert_eq!(stream.get_device(), device_id);
                            stream.memcpy_d2d(scalar.value, poly.get_ptr(idx), 1);
                        }
                        DeviceType::Disk => unreachable!("scalar can't be on disk"),
                    }
                }
                Instruction::LoadInput { src, dst } => {
                    let mut input_guard = self.inputs[src].lock().unwrap();
                    let input = input_guard.take().unwrap();
                    drop(input_guard);
                    let mut guard = self.variable[dst].write().unwrap();
                    assert!(guard.is_none());
                    *guard = Some(input);
                }
                Instruction::MoveRegister { src, dst } => {
                    if src == dst {
                        continue;
                    }
                    let mut src_guard = self.variable[src].write().unwrap();
                    let var = src_guard.take().unwrap();
                    drop(src_guard);
                    let mut dst_guard = self.variable[dst].write().unwrap();
                    *dst_guard = Some(var);
                }
                Instruction::AssertEq {
                    value: value_id,
                    expected: expected_id,
                } => {
                    let value_guard = self.variable[value_id].read().unwrap();
                    let expected_guard = self.variable[expected_id].read().unwrap();
                    let value = value_guard.as_ref().unwrap();
                    let expected = expected_guard.as_ref().unwrap();
                    if !assert_eq::assert_eq(value, expected) {
                        println!(
                            "assertion eq failed at thread {:?}: {:?} != {:?}",
                            _thread_id, value_id, expected_id,
                        );
                    } else {
                        println!(
                            "assertion eq passed at thread {:?}: {:?} == {:?}",
                            _thread_id, value_id, expected_id
                        );
                    }
                }
                Instruction::Print(value_id, label) => {
                    let value_guard = self.variable[value_id].read().unwrap();
                    let value = value_guard.as_ref().unwrap();
                    println!("{}({:?}) = {:?}", label, value_id, value)
                }
                Instruction::CopyRegister { src, dst } => {
                    let src_guard = self.variable[src].read().unwrap();
                    let var = src_guard.as_ref().unwrap().clone();
                    let mut dst_guard = self.variable[dst].write().unwrap();
                    *dst_guard = Some(var);
                }
            }
            if self.bench_start.is_some() {
                unsafe {
                    cuda_check!(cudaDeviceSynchronize());
                }
                let start_duration = start_time
                    .unwrap()
                    .saturating_duration_since(self.bench_start.clone().unwrap());
                let end_duration =
                    Instant::now().saturating_duration_since(self.bench_start.clone().unwrap());
                if let Some(func_name) = function_name {
                    println!(
                        "thread {:?} instruction FuncCall {} start: {:?} end: {:?}",
                        _thread_id,
                        func_name,
                        start_duration.as_micros(),
                        end_duration.as_micros()
                    );
                } else {
                    println!(
                        "thread {:?} instruction {:?} start: {:?} end: {:?}",
                        _thread_id,
                        instruct_copy.unwrap(),
                        start_duration.as_micros(),
                        end_duration.as_micros()
                    );
                }
            }
        }
        if !self.main_thread {
            epilogue
                .unwrap()
                .send(1)
                .expect("channel will be there waiting for the pool");
        }

        None
    }
}
