use std::{
    collections::{BTreeSet, HashMap},
    sync::{mpsc::Sender, Arc, Mutex},
    thread,
    time::Instant,
};

use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
pub use threadpool::ThreadPool;

use zkpoly_common::{devices::DeviceType, load_dynamic::Libs};

use crate::{
    args::{new_variable_table, ConstantTable, EntryTable, RuntimeType, Variable, VariableTable},
    async_rng::AsyncRng,
    debug::{DebugInfo, DebugInfoCollector},
    devices::{new_thread_table, Event, EventTable, ThreadTable},
    functions::{FuncMeta, FunctionTable},
    instructions::Instruction,
};

use zkpoly_cuda_api::{bindings::cudaDeviceSynchronize, cuda_check, mem::CudaAllocator};

use zkpoly_memory_pool::{static_allocator::CpuStaticAllocator, BuddyDiskPool};

pub mod alloc;
pub mod assert_eq;
pub mod transfer;

pub struct Runtime<T: RuntimeType> {
    instructions: Vec<Instruction>,
    gpu_mapping: Arc<dyn Fn(i32) -> i32 + Send + Sync>,
    variable: VariableTable<T>,
    constant: ConstantTable<T>,
    funcs: FunctionTable<T>,
    events: EventTable,
    threads: ThreadTable,
    pub mem_allocator: Option<CpuStaticAllocator>,
    gpu_allocator: HashMap<i32, CudaAllocator>,
    disk_allocator: Vec<BuddyDiskPool>,
    rng: AsyncRng,
    _libs: Libs,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum RuntimeDebug {
    RecordTime,
    BenchKernel,
    DebugInstruction,
    RecordTimeAsync,
    None,
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
        funcs: FunctionTable<T>,
        events: EventTable,
        n_threads: usize,
        mem_allocator: CpuStaticAllocator,
        gpu_allocator: HashMap<i32, CudaAllocator>,
        disk_allocator: Vec<BuddyDiskPool>,
        rng: AsyncRng,
        gpu_mapping: Arc<dyn Fn(i32) -> i32 + Send + Sync>,
        libs: Libs,
    ) -> Self {
        Self {
            instructions,
            variable: new_variable_table(n_variables),
            constant,
            funcs,
            events,
            threads: new_thread_table(n_threads),
            mem_allocator: Some(mem_allocator),
            gpu_allocator,
            disk_allocator,
            gpu_mapping,
            rng,
            _libs: libs,
        }
    }

    pub fn reset(&mut self) {
        self.events.0.par_iter_mut().for_each(|event| match event {
            Event::GpuEvent(e) => e.reset(),
            Event::ThreadEvent(e) => e.reset(),
        });
        for (i, var) in self.variable.0.iter_mut().enumerate() {
            if let Some(_) = var {
                println!("var {} is not eliminated", i);
            }
            *var = None;
        }
    }

    pub fn run(
        &mut self,
        input_table: &EntryTable<T>,
        debug_opt: RuntimeDebug,
    ) -> (
        (
            Option<Variable<T>>,
            Option<DebugInfoCollector>,
            RuntimeInfo<T>,
        ),
        CpuStaticAllocator,
    ) {
        let bench_start = if RuntimeDebug::RecordTime == debug_opt
            || RuntimeDebug::RecordTimeAsync == debug_opt
        {
            Some(Instant::now())
        } else {
            None
        };
        let executed_kernels = if RuntimeDebug::BenchKernel == debug_opt {
            Some(Arc::new(Mutex::new(BTreeSet::new())))
        } else {
            None
        };
        let debug_instruction = RuntimeDebug::DebugInstruction == debug_opt;
        let info = RuntimeInfo {
            gpu_mapping: self.gpu_mapping.clone(),
            variable: &mut self.variable as *mut VariableTable<T>,
            constant: &self.constant as *const ConstantTable<T>,
            inputs: input_table as *const EntryTable<T>,
            funcs: &mut self.funcs as *mut FunctionTable<T>,
            events: &self.events as *const EventTable,
            threads: &mut self.threads as *mut ThreadTable,
            rng: self.rng.clone(),
            main_thread: true,
            bench_start,
            executed_kernels,
            debug_instruction,
        };
        let (r, debug_info) = unsafe {
            info.run(
                self.instructions.clone(),
                Some(&mut self.mem_allocator.as_mut().unwrap()),
                Some(&mut self.gpu_allocator),
                Some(&mut self.disk_allocator),
                None,
                0,
                Arc::new(std::sync::Mutex::new(())),
                debug_opt,
            )
        };
        ((r, debug_info, info), self.mem_allocator.take().unwrap())
    }
}

#[derive(Clone)]
pub struct RuntimeInfo<T: RuntimeType> {
    pub gpu_mapping: Arc<dyn Fn(i32) -> i32 + Send + Sync>,
    pub variable: *mut VariableTable<T>,
    pub constant: *const ConstantTable<T>,
    pub inputs: *const EntryTable<T>,
    pub funcs: *mut FunctionTable<T>,
    pub events: *const EventTable,
    pub threads: *mut ThreadTable,
    pub rng: AsyncRng,
    pub main_thread: bool,
    pub bench_start: Option<Instant>,
    pub executed_kernels: Option<Arc<Mutex<BTreeSet<FuncMeta>>>>,
    debug_instruction: bool,
}

unsafe impl<T: RuntimeType> Send for RuntimeInfo<T> {}
unsafe impl<T: RuntimeType> Sync for RuntimeInfo<T> {}

impl<T: RuntimeType> RuntimeInfo<T> {
    #[allow(dangerous_implicit_autorefs)]
    pub unsafe fn run(
        &self,
        instructions: Vec<Instruction>,
        mut mem_allocator: Option<&mut CpuStaticAllocator>,
        mut gpu_allocator: Option<&mut HashMap<i32, CudaAllocator>>,
        mut disk_allocator: Option<&mut Vec<BuddyDiskPool>>,
        epilogue: Option<Sender<Option<DebugInfoCollector>>>,
        _thread_id: usize,
        global_mutex: Arc<Mutex<()>>,
        debug_option: RuntimeDebug,
    ) -> (Option<Variable<T>>, Option<DebugInfoCollector>) {
        let mut debug_infos = if debug_option == RuntimeDebug::RecordTimeAsync {
            Some(DebugInfoCollector {
                debug_info: Vec::new(),
                sub_thread_debug_info: HashMap::new(),
            })
        } else {
            None
        };
        for (i, instruction) in instructions.into_iter().enumerate() {
            let _guard: Option<std::sync::MutexGuard<'_, ()>> = if debug_option
                == RuntimeDebug::RecordTime
                || debug_option == RuntimeDebug::BenchKernel
                || debug_option == RuntimeDebug::DebugInstruction
            {
                Some(global_mutex.lock().unwrap())
            } else {
                None
            };

            let (start_time, instruct_copy) = if self.bench_start.is_some() {
                if debug_option == RuntimeDebug::RecordTime {
                    unsafe {
                        cuda_check!(cudaDeviceSynchronize()); // wait for all previous cuda calls
                    }
                }
                let instruct_copy = if let Instruction::AssertEq { .. } = &instruction {
                    None
                } else {
                    Some(instruction.clone())
                };
                (Some(Instant::now()), instruct_copy)
            } else {
                (None, None)
            };

            if self.debug_instruction {
                println!("{:?}", &instruction);
            }

            let mut function_name = None;
            match instruction {
                Instruction::Allocate {
                    device,
                    typ,
                    id,
                    alloc_method,
                } => {
                    // only main thread can allocate memory
                    assert!(self.main_thread);
                    let guard = &mut (*self.variable)[id];
                    assert!(guard.is_none());
                    *guard = Some(self.allocate(
                        device,
                        typ,
                        alloc_method,
                        &mut mem_allocator,
                        &mut gpu_allocator,
                        &mut disk_allocator,
                    ));
                }
                Instruction::Deallocate { id, alloc_method } => {
                    // only main thread can deallocate memory
                    assert!(self.main_thread);
                    let guard = &mut (*self.variable)[id];
                    if let Some(var) = guard.as_mut() {
                        self.deallocate(
                            var,
                            id,
                            alloc_method,
                            &mut mem_allocator,
                            &mut gpu_allocator,
                            &mut disk_allocator,
                        );
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
                    let src = (*self.variable)[src_id].as_ref().unwrap();
                    let dst = (*self.variable)[dst_id].as_mut().unwrap();
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
                        .map(|id| (*self.variable)[*id].clone())
                        .collect::<Vec<_>>();
                    let mut arg_mut_holder = arg_mut
                        .iter()
                        .map(|id| (*self.variable)[*id].clone())
                        .collect::<Vec<_>>();

                    let args_mut: Vec<_> = arg_mut_holder
                        .iter_mut()
                        .zip(arg_mut.iter())
                        .map(|(guard, r_mut)| {
                            guard
                                .as_mut()
                                .unwrap_or_else(|| panic!("{:?} is undefined", r_mut))
                        })
                        .collect();
                    let args: Vec<_> = arg_holder
                        .iter()
                        .zip(arg.iter())
                        .map(|(guard, r)| {
                            guard
                                .as_ref()
                                .unwrap_or_else(|| panic!("{:?} is undefined", r))
                        })
                        .collect();

                    function_name = Some((*self.funcs)[func_id].meta.name.clone());
                    if self.executed_kernels.is_some() {
                        let meta = (*self.funcs)[func_id].meta.clone();
                        let mut executed_kernels =
                            self.executed_kernels.as_ref().unwrap().lock().unwrap();
                        if executed_kernels.get(&meta).is_none() {
                            executed_kernels.insert(meta.clone());
                        } else {
                            continue;
                        }
                    }
                    let f = &mut (*self.funcs)[func_id].f;
                    f(args_mut, args, self.gpu_mapping.clone()).unwrap();
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
                    let ref event = (*self.events)[event_id];
                    match event {
                        Event::GpuEvent(cuda_event) => match slave {
                            DeviceType::CPU => {
                                cuda_event.sync();
                            }
                            DeviceType::GPU { .. } => {
                                let stream_guard = &(*self.variable)[stream.unwrap()];
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
                    let ref event = (*self.events)[event];
                    match event {
                        Event::GpuEvent(cuda_event) => {
                            let stream_guard = &(*self.variable)[stream.unwrap()];
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
                    (*self.threads)[new_thread].replace(rx);
                    let mut sub_info = self.clone();
                    sub_info.main_thread = false;
                    let global_mutex = global_mutex.clone();
                    thread::spawn(move || {
                        sub_info.run(
                            instructions,
                            None,
                            None,
                            None,
                            Some(tx),
                            new_thread.into(),
                            global_mutex,
                            debug_option.clone(),
                        );
                    });
                }
                Instruction::Join { thread } => {
                    let rx = (*self.threads)[thread].take().unwrap();
                    let sub_debug_info = rx.recv().unwrap();
                    if let Some(debug_infos) = &mut debug_infos {
                        debug_infos
                            .sub_thread_debug_info
                            .insert(thread, Box::new(sub_debug_info.unwrap()));
                    }
                }
                Instruction::Rotation { src, dst, shift } => {
                    let guard = &(*self.variable)[src];
                    let mut poly = guard.as_ref().unwrap().unwrap_scalar_array().clone();
                    poly.rotate(shift);
                    let guard = &mut (*self.variable)[dst];
                    assert!(guard.is_none());
                    *guard = Some(Variable::ScalarArray(poly));
                }
                Instruction::Slice {
                    src,
                    dst,
                    start,
                    end,
                } => {
                    let src_guard = &(*self.variable)[src];
                    let slice = src_guard
                        .as_ref()
                        .unwrap()
                        .unwrap_scalar_array()
                        .slice(start, end);
                    let dst_guard = &mut (*self.variable)[dst];
                    assert!(dst_guard.is_none());
                    *dst_guard = Some(Variable::ScalarArray(slice));
                }
                Instruction::LoadConstant { src, dst } => {
                    let constant = (*self.constant)[src].clone();
                    let guard = &mut (*self.variable)[dst];
                    assert!(guard.is_none());
                    *guard = Some(constant.value);
                }
                Instruction::AssembleTuple { vars, dst } => {
                    let mut assemble = Vec::new();
                    for var in vars.iter() {
                        let guard = &(*self.variable)[*var];
                        assemble.push((guard.as_ref().unwrap()).clone());
                    }
                    let guard = &mut (*self.variable)[dst];
                    assert!(guard.is_none());
                    *guard = Some(Variable::Tuple(assemble));
                }
                Instruction::RemoveRegister { id } => {
                    let guard = &mut (*self.variable)[id];
                    assert!(guard.is_some());
                    *guard = None;
                }
                Instruction::Blind { dst, start, end } => {
                    let guard = &mut (*self.variable)[dst];
                    let poly = guard.as_mut().unwrap().unwrap_scalar_array_mut();
                    poly.blind(start, end, self.rng.clone());
                }
                Instruction::Return(var_id) => {
                    if !self.main_thread {
                        panic!("can only return from main thread");
                    }
                    let var = (*self.variable)[var_id].take().unwrap();
                    return (Some(var), debug_infos);
                }
                Instruction::SetSliceMeta {
                    src,
                    dst,
                    offset,
                    len,
                } => {
                    let src_guard = &(*self.variable)[src];
                    let poly = src_guard
                        .as_ref()
                        .unwrap()
                        .unwrap_scalar_array()
                        .set_slice_raw(offset, len);
                    if poly.is_none() {
                        panic!(
                            "set_slice_raw failed at thread {:?}, instruction {:?}, for {:?} to {:?}",
                            _thread_id, i, src, dst
                        );
                    }
                    // println!("set dst {:?} meta to {:?}", dst.clone(), poly.clone());
                    let dst_guard = &mut (*self.variable)[dst];
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
                    let src_guard = &(*self.variable)[src];
                    let dst_guard = &mut (*self.variable)[dst];
                    let poly = src_guard.as_ref().unwrap().unwrap_scalar_array();
                    let scalar = dst_guard.as_mut().unwrap().unwrap_scalar_mut();
                    assert_eq!(poly.device, scalar.device);
                    match poly.device {
                        DeviceType::CPU => {
                            *scalar.as_mut() = poly[idx];
                        }
                        DeviceType::GPU { device_id } => {
                            let stream_guard = &(*self.variable)[stream.unwrap()];
                            let stream = stream_guard.as_ref().unwrap().unwrap_stream();
                            assert_eq!(stream.get_device(), device_id);
                            stream.memcpy_d2d(scalar.value, poly.get_ptr(idx), 1);
                        }
                        DeviceType::Disk => unreachable!("scalar can't be on disk"),
                    }
                }
                Instruction::LoadInput { src, dst } => {
                    let input_guard = &(*self.inputs)[src];
                    let input = input_guard.clone();
                    let guard = &mut (*self.variable)[dst];
                    assert!(guard.is_none());
                    *guard = Some(input); // the compiled instructions will ensure that the input is not modified
                }
                Instruction::MoveRegister { src, dst } => {
                    if src == dst {
                        continue;
                    }
                    let src_guard = &mut (*self.variable)[src];
                    let var = src_guard.take().unwrap();
                    let dst_guard = &mut (*self.variable)[dst];
                    *dst_guard = Some(var);
                }
                Instruction::AssertEq {
                    value: value_id,
                    expected: expected_id,
                    msg,
                } => {
                    let value_guard = &(*self.variable)[value_id];
                    let expected_guard = &(*self.variable)[expected_id];
                    let value = value_guard.as_ref().unwrap();
                    let expected = expected_guard.as_ref().unwrap();
                    if !assert_eq::assert_eq(value, expected) {
                        println!(
                            "assertion eq failed at thread {:?}: {:?} != {:?} {}",
                            _thread_id,
                            value_id,
                            expected_id,
                            msg.unwrap_or_default()
                        );
                    } else {
                        println!(
                            "assertion eq passed at thread {:?}: {:?} == {:?}",
                            _thread_id, value_id, expected_id
                        );
                    }
                }
                Instruction::Print(value_id, label) => {
                    let value_guard = &(*self.variable)[value_id];
                    let value = value_guard.as_ref().unwrap();
                    println!("{}({:?}) = {:?}", label, value_id, value)
                }
                Instruction::CopyRegister { src, dst } => {
                    let src_guard = &(*self.variable)[src];
                    let var = src_guard.as_ref().unwrap().clone();
                    let dst_guard = &mut (*self.variable)[dst];
                    *dst_guard = Some(var);
                }
            }
            if self.bench_start.is_some() && instruct_copy.is_some() {
                if debug_option == RuntimeDebug::RecordTime {
                    unsafe {
                        cuda_check!(cudaDeviceSynchronize());
                    }
                }
                let start_duration = start_time
                    .unwrap()
                    .saturating_duration_since(self.bench_start.clone().unwrap());
                let end_duration =
                    Instant::now().saturating_duration_since(self.bench_start.clone().unwrap());
                if debug_option == RuntimeDebug::RecordTime {
                    if let Some(func_name) = function_name {
                        println!(
                            "thread {:?} instruction FuncCall {} {:?} start: {:?} end: {:?}",
                            _thread_id,
                            func_name,
                            instruct_copy.unwrap(),
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
                } else {
                    debug_infos.as_mut().unwrap().debug_info.push(DebugInfo {
                        instruction: instruct_copy.unwrap(),
                        start_duration,
                        end_duration,
                    });
                }
            }
        }
        if !self.main_thread {
            epilogue
                .unwrap()
                .send(debug_infos)
                .expect("channel will be there waiting for the pool");
        }

        (None, None)
    }
}
