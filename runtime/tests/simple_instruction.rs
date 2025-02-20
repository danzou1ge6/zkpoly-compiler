use std::sync::RwLock;

use halo2curves::bn256;
use threadpool::ThreadPool;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_common::typ::{PolyMeta, Typ};
use zkpoly_core::poly::PolyAdd;
use zkpoly_cuda_api::mem::CudaAllocator;
use zkpoly_cuda_api::stream::CudaEvent;
use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_runtime::args::{ConstantTable, RuntimeType, Variable, VariableTable};
use zkpoly_runtime::async_rng::AsyncRng;
use zkpoly_runtime::devices::{DeviceType, Event, EventTable, ThreadTable};
use zkpoly_runtime::functions::*;
use zkpoly_runtime::instructions::Instruction;
use zkpoly_runtime::runtime::Runtime;
use zkpoly_runtime::scalar::ScalarArray;
use zkpoly_runtime::transcript::{Blake2bWrite, Challenge255};

#[derive(Debug, Clone)]
pub struct MyRuntimeType;

impl RuntimeType for MyRuntimeType {
    type Field = bn256::Fr;
    type PointAffine = bn256::G1Affine;
    type Challenge = Challenge255<bn256::G1Affine>;
    type Trans = Blake2bWrite<Vec<u8>, bn256::G1Affine, Challenge255<bn256::G1Affine>>;
}

pub type MyField = <MyRuntimeType as RuntimeType>::Field;

// a is constant poly
// b = a + a
// c = a + b
#[test]
fn test_add() {
    let k = 20;
    let len = 1 << k;

    let compiler_alloc = PinnedMemoryPool::new(k, size_of::<MyField>());
    let mut a_in = Variable::ScalarArray(ScalarArray::new(
        len,
        compiler_alloc.allocate(len),
        DeviceType::CPU,
    ));
    for iter in a_in.unwrap_scalar_array_mut().iter_mut() {
        *iter = MyField::one();
    }
    let c_out = Variable::ScalarArray(ScalarArray::new(
        len,
        compiler_alloc.allocate(len),
        DeviceType::CPU,
    ));

    let mut variable = VariableTable::<MyRuntimeType>::new();

    let ida = variable.push(RwLock::new(Some(a_in)));
    let idb = variable.push(RwLock::new(None));
    let idc = variable.push(RwLock::new(Some(c_out)));
    let ida_d = variable.push(RwLock::new(None));
    let idb_d = variable.push(RwLock::new(None));
    let idc_d = variable.push(RwLock::new(None));
    let stream_id = variable.push(RwLock::new(None));

    let mut libs = Libs::new();
    let mut funcs = FunctionTable::<MyRuntimeType>::new();
    let add_func_id = funcs.push(PolyAdd::new(&mut libs).get_fn());

    let mut events = EventTable::new();
    let end_event_id = events.push(Event::GpuEvent(CudaEvent::new()));

    let mut instructions = Vec::new();

    // allocate memory
    instructions.push(Instruction::Allocate {
        device: DeviceType::CPU,
        typ: Typ::scalar_array(len),
        id: idb,
        offset: None,
    });
    instructions.push(Instruction::Allocate {
        device: DeviceType::GPU { device_id: 0 },
        typ: Typ::scalar_array(len),
        id: ida_d,
        offset: Some(0),
    });
    instructions.push(Instruction::Allocate {
        device: DeviceType::GPU { device_id: 0 },
        typ: Typ::scalar_array(len),
        id: idb_d,
        offset: Some(len * size_of::<MyField>()),
    });
    instructions.push(Instruction::Allocate {
        device: DeviceType::GPU { device_id: 0 },
        typ: Typ::scalar_array(len),
        id: idc_d,
        offset: Some(2 * len * size_of::<MyField>()),
    });
    instructions.push(Instruction::Allocate {
        device: DeviceType::GPU { device_id: 0 },
        typ: Typ::Stream,
        id: stream_id,
        offset: None,
    });

    // copy a to device
    instructions.push(Instruction::Transfer {
        src_device: DeviceType::CPU,
        dst_device: DeviceType::GPU { device_id: 1 },
        stream: Some(stream_id),
        src_id: ida,
        dst_id: ida_d,
    });

    // b = a + a
    instructions.push(Instruction::FuncCall {
        func_id: add_func_id,
        arg_mut: vec![idb_d],
        arg: vec![ida_d, ida_d, stream_id],
    });

    // c = a + b
    instructions.push(Instruction::FuncCall {
        func_id: add_func_id,
        arg_mut: vec![idc_d],
        arg: vec![ida_d, idb_d, stream_id],
    });

    // copy b, c back to cpu
    instructions.push(Instruction::Transfer {
        src_device: DeviceType::GPU { device_id: 0 },
        dst_device: DeviceType::CPU,
        stream: Some(stream_id),
        src_id: idb_d,
        dst_id: idb,
    });
    instructions.push(Instruction::Transfer {
        src_device: DeviceType::GPU { device_id: 0 },
        dst_device: DeviceType::CPU,
        stream: Some(stream_id),
        src_id: idc_d,
        dst_id: idc,
    });

    // wait for copy to finish
    // record event
    instructions.push(Instruction::Record {
        stream: Some(stream_id),
        event: end_event_id,
    });

    // wait for event
    instructions.push(Instruction::Wait {
        slave: DeviceType::CPU,
        stream: None,
        event: end_event_id,
    });

    // deallocate memory
    instructions.push(Instruction::Deallocate { id: idb });
    instructions.push(Instruction::Deallocate { id: ida_d });
    instructions.push(Instruction::Deallocate { id: idb_d });
    instructions.push(Instruction::Deallocate { id: idc_d });
    instructions.push(Instruction::Deallocate { id: stream_id });

    let cpu_alloc = PinnedMemoryPool::new(k, size_of::<MyField>());
    let gpu_alloc = CudaAllocator::new(0, len * size_of::<MyField>() * 3);

    let runtime = Runtime::new(
        instructions,
        variable,
        ConstantTable::new(),
        ThreadPool::new(1),
        funcs,
        events,
        ThreadTable::new(),
        cpu_alloc,
        vec![gpu_alloc],
        AsyncRng::new(10),
        libs,
    );
    let info = runtime.run();
    let variable = info.variable;

    let binding_c = variable[idc].read().unwrap();

    for ci in binding_c.as_ref().unwrap().unwrap_scalar_array().iter() {
        assert_eq!(*ci, MyField::one() + MyField::one() + MyField::one());
    }

    drop(binding_c);

    compiler_alloc.free(
        variable[ida]
            .write()
            .unwrap()
            .as_mut()
            .unwrap()
            .unwrap_scalar_array_mut()
            .values,
    );
    compiler_alloc.free(
        variable[idc]
            .write()
            .unwrap()
            .as_mut()
            .unwrap()
            .unwrap_scalar_array_mut()
            .values,
    );
}
