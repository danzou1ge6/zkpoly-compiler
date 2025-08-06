use std::{
    borrow::BorrowMut,
    collections::{BTreeSet, HashMap},
    ops::DerefMut,
    sync::{Arc, Mutex},
    thread,
};

use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
pub use threadpool::ThreadPool;

use zkpoly_common::{devices::DeviceType, load_dynamic::Libs, typ::Typ};

use crate::{
    args::{new_variable_table, ConstantTable, EntryTable, RuntimeType, Variable, VariableTable},
    async_rng::AsyncRng,
    debug,
    devices::{Event, EventTable},
    functions::{FuncMeta, FunctionTable},
    instructions::{Instruction, InstructionNode},
};

use zkpoly_cuda_api::{
    bindings::{cudaDeviceSynchronize, cudaSetDevice},
    cuda_check,
    mem::CudaAllocator,
    stream::{CudaEvent, CudaEventRaw},
};

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
    pub mem_allocator: Option<CpuStaticAllocator>,
    gpu_allocator: HashMap<i32, CudaAllocator>,
    disk_allocator: Arc<Mutex<Vec<BuddyDiskPool>>>,
    rng: AsyncRng,
    _libs: Libs,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct RuntimeDebug {
    pub bench_kernel: bool,
    pub print_instruction: bool,
    pub record_time: bool,
    pub serial_execution: bool,
}

impl RuntimeDebug {
    pub fn none() -> Self {
        Self {
            bench_kernel: false,
            print_instruction: false,
            record_time: false,
            serial_execution: false,
        }
    }

    pub fn with_bench_kernel(self, bench_mark: bool) -> Self {
        Self {
            bench_kernel: bench_mark,
            ..self
        }
    }

    pub fn with_print_instruction(self, print_instruction: bool) -> Self {
        Self {
            print_instruction,
            ..self
        }
    }

    pub fn with_record_time(self, record_time: bool) -> Self {
        Self {
            record_time,
            ..self
        }
    }

    pub fn with_serial_execution(self, serial_execution: bool) -> Self {
        Self {
            serial_execution,
            ..self
        }
    }
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
        disk_allocator: Arc<Mutex<Vec<BuddyDiskPool>>>,
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
        (Option<Variable<T>>, debug::Log, RuntimeInfo<T>),
        CpuStaticAllocator,
    ) {
        let executed_kernels = if debug_opt.bench_kernel {
            Some(Arc::new(Mutex::new(BTreeSet::new())))
        } else {
            None
        };

        let logger = debug::Logger::new();
        let logger_handle = logger.spawn();

        let info = RuntimeInfo {
            gpu_mapping: self.gpu_mapping.clone(),
            variable: &mut self.variable as *mut VariableTable<T>,
            constant: &self.constant as *const ConstantTable<T>,
            inputs: input_table as *const EntryTable<T>,
            funcs: &mut self.funcs as *mut FunctionTable<T>,
            events: &self.events as *const EventTable,
            rng: self.rng.clone(),
            main_thread: true,
            executed_kernels,
            debug_option: debug_opt,
            debug_writer: logger_handle.writer(),
            global_mutex: Arc::new(Mutex::new(())),
        };
        let r = unsafe {
            info.run(
                &self.instructions,
                Some(&mut self.mem_allocator.as_mut().unwrap()),
                Some(&mut self.gpu_allocator),
                Some(self.disk_allocator.as_ref()),
                0,
            )
        };

        let log = logger_handle.join();
        ((r, log, info), self.mem_allocator.take().unwrap())
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
    pub rng: AsyncRng,
    pub main_thread: bool,
    /// If this is Some, the runtime will only execute each kernel once
    pub executed_kernels: Option<Arc<Mutex<BTreeSet<FuncMeta>>>>,
    debug_option: RuntimeDebug,
    debug_writer: debug::Writer,
    global_mutex: Arc<Mutex<()>>,
}

unsafe impl<T: RuntimeType> Send for RuntimeInfo<T> {}
unsafe impl<T: RuntimeType> Sync for RuntimeInfo<T> {}

impl<T: RuntimeType> RuntimeInfo<T> {
    #[allow(dangerous_implicit_autorefs)]
    pub unsafe fn run(
        &self,
        instructions: &[Instruction],
        mut mem_allocator: Option<&mut CpuStaticAllocator>,
        mut gpu_allocator: Option<&mut HashMap<i32, CudaAllocator>>,
        disk_allocator: Option<&Mutex<Vec<BuddyDiskPool>>>,
        _thread_id: usize,
    ) -> Option<Variable<T>> {
        let mut disk2gpu_temp_buffer: Vec<*mut u8> = Vec::new();
        let mut gpu2disk_temp_buffer: Vec<*mut u8> = Vec::new();
        let temp_size: usize = 1024 * 1024 * 2; // 2MB temporary buffer size
        for (i, instruction) in instructions.into_iter().enumerate() {
            let _guard: Option<std::sync::MutexGuard<'_, ()>> =
                if self.debug_option.serial_execution {
                    Some(self.global_mutex.lock().unwrap())
                } else {
                    None
                };

            if self.debug_option.print_instruction {
                println!("{:?}", &instruction);
            }

            let ibg = if self.debug_option.record_time {
                Some(self.debug_writer.begin_instruction(instruction.clone()))
            } else {
                None
            };

            let mut function_meta = None;
            use InstructionNode::*;
            match instruction.node().clone() {
                Allocate {
                    device,
                    typ,
                    id,
                    alloc_method,
                } => {
                    // only main thread can allocate memory
                    assert!(self.main_thread);
                    let guard = &mut (*self.variable)[id];
                    assert!(guard.is_none());

                    let var = self.allocate(
                        device.clone(),
                        typ.clone(),
                        alloc_method,
                        &mut mem_allocator,
                        &mut gpu_allocator,
                        &mut disk_allocator.map(|m| m.lock().unwrap()),
                    );

                    if let Typ::Stream = &typ {
                        let stream = var.unwrap_stream();
                        let event = CudaEventRaw::new(stream.get_device());
                        event.record(stream);
                        self.debug_writer.new_stream(
                            instruction.unwrap_stream(),
                            stream.clone(),
                            event,
                        );
                    }

                    *guard = Some(var);
                }
                Deallocate { id, alloc_method } => {
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
                            &mut disk_allocator.map(|m| m.lock().unwrap()),
                        );
                        *guard = None;
                    } else {
                        panic!("deallocate a non-existing variable");
                    }
                }
                Transfer {
                    src_device,
                    dst_device,
                    stream,
                    src_id,
                    dst_id,
                } => {
                    let src = (*self.variable)[src_id].as_ref().unwrap();
                    let dst = (*self.variable)[dst_id].as_mut().unwrap();
                    self.transfer(src, dst, src_device.clone(), dst_device.clone(), stream, 
                                  &mut disk2gpu_temp_buffer, &mut gpu2disk_temp_buffer, temp_size);
                }
                FuncCall {
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

                    function_meta = Some((*self.funcs)[func_id].meta.clone());
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
                Wait {
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
                Record { stream, event } => {
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
                Fork {
                    new_thread,
                    instructions,
                } => {
                    let mut sub_info = self.clone();
                    sub_info.main_thread = false;
                    sub_info.debug_writer = sub_info.debug_writer.with_thread_id(new_thread);

                    thread::spawn(move || {
                        sub_info.run(&instructions, None, None, None, new_thread.into());
                    });
                }
                Join { .. } => {}
                Rotation { src, dst, shift } => {
                    let guard = &(*self.variable)[src];
                    let mut poly = guard.as_ref().unwrap().unwrap_scalar_array().clone();
                    poly.rotate(shift);
                    let guard = &mut (*self.variable)[dst];
                    assert!(guard.is_none());
                    *guard = Some(Variable::ScalarArray(poly));
                }
                Slice {
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
                LoadConstant { src, dst } => {
                    let constant = (*self.constant)[src].clone();
                    let guard = &mut (*self.variable)[dst];
                    assert!(guard.is_none());
                    *guard = Some(constant.value);
                }
                AssembleTuple { vars, dst } => {
                    let mut assemble = Vec::new();
                    for var in vars.iter() {
                        let guard = &(*self.variable)[*var];
                        assemble.push((guard.as_ref().unwrap()).clone());
                    }
                    let guard = &mut (*self.variable)[dst];
                    assert!(guard.is_none());
                    *guard = Some(Variable::Tuple(assemble));
                }
                RemoveRegister { id } => {
                    let guard = &mut (*self.variable)[id];
                    assert!(guard.is_some());
                    *guard = None;
                }
                Blind { dst, start, end } => {
                    let guard = &mut (*self.variable)[dst];
                    let poly = guard.as_mut().unwrap().unwrap_scalar_array_mut();
                    poly.blind(start, end, self.rng.clone());
                }
                Return(var_id) => {
                    if !self.main_thread {
                        panic!("can only return from main thread");
                    }
                    let var = (*self.variable)[var_id].take().unwrap();
                    return Some(var);
                }
                SetSliceMeta {
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
                GetScalarFromArray {
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
                LoadInput { src, dst } => {
                    let input_guard = &(*self.inputs)[src];
                    let input = input_guard.clone();
                    let guard = &mut (*self.variable)[dst];
                    assert!(guard.is_none());
                    *guard = Some(input); // the compiled instructions will ensure that the input is not modified
                }
                MoveRegister { src, dst } => {
                    if src == dst {
                        continue;
                    }
                    let src_guard = &mut (*self.variable)[src];
                    let var = src_guard.take().unwrap();
                    let dst_guard = &mut (*self.variable)[dst];
                    *dst_guard = Some(var);
                }
                AssertEq {
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
                Print(value_id, label) => {
                    let value_guard = &(*self.variable)[value_id];
                    let value = value_guard.as_ref().unwrap();
                    println!("{}({:?}) = {:?}", label, value_id, value)
                }
                CopyRegister { src, dst } => {
                    let src_guard = &(*self.variable)[src];
                    let var = src_guard.as_ref().unwrap().clone();
                    let dst_guard = &mut (*self.variable)[dst];
                    *dst_guard = Some(var);
                }
                SliceBuffer {
                    src,
                    dst,
                    offset,
                    len,
                } => {
                    let src_guard = &(*self.variable)[src];
                    let var = src_guard.as_ref().unwrap().clone();
                    let dst_guard = &mut (*self.variable)[dst];

                    let buf = var.unwrap_gpu_buffer().clone().sliced(offset, len);
                    *dst_guard = Some(Variable::GpuBuffer(buf))
                }
            }

            if self.debug_option.record_time {
                self.debug_writer
                    .end_instruction(ibg.unwrap(), function_meta);
            }
        }

        None
    }
}
