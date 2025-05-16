use halo2curves::bn256;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_common::typ::Typ;
use zkpoly_core::poly::{PolyAdd, PolyZero};
use zkpoly_cuda_api::mem::CudaAllocator;
use zkpoly_cuda_api::stream::CudaEvent;
use zkpoly_memory_pool::CpuMemoryPool;
use zkpoly_runtime::args::{ConstantTable, EntryTable, RuntimeType, Variable, VariableTable};
use zkpoly_runtime::async_rng::AsyncRng;
use zkpoly_runtime::devices::{DeviceType, Event, EventTable};
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

    let mut compiler_alloc = CpuMemoryPool::new(k, size_of::<MyField>());
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

    let ida = variable.push(Some(a_in));
    let idb = variable.push(None);
    let idc = variable.push(Some(c_out));
    let ida_d = variable.push(None);
    let idb_d = variable.push(None);
    let idc_d = variable.push(None);
    let stream_id = variable.push(None);

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

    let cpu_alloc = CpuMemoryPool::new(k, size_of::<MyField>());
    let gpu_alloc = CudaAllocator::new(0, len * size_of::<MyField>() * 3, true);

    let mut runtime = Runtime::new(
        instructions,
        0,
        ConstantTable::new(),
        funcs,
        events,
        0,
        cpu_alloc,
        vec![gpu_alloc],
        AsyncRng::new(10),
        libs,
    )
    .with_variables(variable);
    let (_, info) = runtime.run(
        &mut EntryTable::new(),
        zkpoly_runtime::runtime::RuntimeDebug::None,
    );
    let variable = info.variable;

    let binding_c = unsafe { &(*variable)[idc] };

    for ci in binding_c.as_ref().unwrap().unwrap_scalar_array().iter() {
        assert_eq!(*ci, MyField::one() + MyField::one() + MyField::one());
    }

    unsafe {
        compiler_alloc.free(
            (*variable)[ida]
                .as_mut()
                .unwrap()
                .unwrap_scalar_array_mut()
                .values,
        );
        compiler_alloc.free(
            (*variable)[idc]
                .as_mut()
                .unwrap()
                .unwrap_scalar_array_mut()
                .values,
        );
    }
}

#[test]
fn test_extend() {
    let k = 3;
    let len = 1 << k;
    let half_len = len / 2;

    let mut compiler_alloc = CpuMemoryPool::new(k, size_of::<MyField>());
    let cpu_alloc = CpuMemoryPool::new(k, size_of::<MyField>());
    let gpu_alloc = CudaAllocator::new(0, 2usize.pow(20), true);

    let mut a_in = Variable::ScalarArray(ScalarArray::new(
        half_len,
        compiler_alloc.allocate(half_len),
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
    let ids = variable.push(None);
    let ida = variable.push(Some(a_in));
    let idc = variable.push(Some(c_out));
    let id0 = variable.push(None);
    let id1 = variable.push(None);
    let id2 = variable.push(None);
    let id3 = variable.push(None);
    let id4 = variable.push(None);
    let id5 = variable.push(None);

    let mut libs = Libs::new();
    let mut funcs = FunctionTable::<MyRuntimeType>::new();
    let f_poly_zero = funcs.push(PolyZero::new(&mut libs).get_fn());

    let events = EventTable::new();

    let instructions = vec![
        Instruction::Allocate {
            device: DeviceType::GPU { device_id: 0 },
            typ: Typ::Stream,
            id: ids,
            offset: None,
        },
        Instruction::Allocate {
            device: DeviceType::GPU { device_id: 0 },
            typ: Typ::scalar_array(half_len),
            id: id0,
            offset: Some(1024),
        },
        Instruction::Transfer {
            src_device: DeviceType::CPU,
            dst_device: DeviceType::GPU { device_id: 0 },
            stream: Some(ids),
            src_id: ida,
            dst_id: id0,
        },
        Instruction::Allocate {
            device: DeviceType::GPU { device_id: 0 },
            typ: Typ::scalar_array(len),
            id: id1,
            offset: Some(0),
        },
        Instruction::FuncCall {
            func_id: f_poly_zero,
            arg_mut: vec![id1],
            arg: vec![ids],
        },
        Instruction::MoveRegister { src: id1, dst: id2 },
        Instruction::SetSliceMeta {
            src: id2,
            dst: id3,
            offset: 0,
            len: half_len,
        },
        Instruction::Transfer {
            src_device: DeviceType::GPU { device_id: 0 },
            dst_device: DeviceType::GPU { device_id: 0 },
            stream: Some(ids),
            src_id: id0,
            dst_id: id3,
        },
        Instruction::MoveRegister { src: id3, dst: id4 },
        Instruction::SetSliceMeta {
            src: id4,
            dst: id5,
            offset: 0,
            len: len,
        },
        Instruction::Transfer {
            src_device: DeviceType::GPU { device_id: 0 },
            dst_device: DeviceType::CPU,
            stream: Some(ids),
            src_id: id5,
            dst_id: idc,
        },
    ];

    let mut runtime = Runtime::new(
        instructions,
        0,
        ConstantTable::new(),
        funcs,
        events,
        0,
        cpu_alloc,
        vec![gpu_alloc],
        AsyncRng::new(10),
        libs,
    )
    .with_variables(variable);
    let (_, info) = runtime.run(
        &mut EntryTable::new(),
        zkpoly_runtime::runtime::RuntimeDebug::None,
    );
    let variable = info.variable;
    unsafe {
        let binding_c = &(*variable)[idc];

        for (i, ci) in binding_c
            .as_ref()
            .unwrap()
            .unwrap_scalar_array()
            .iter()
            .enumerate()
        {
            if i < half_len {
                assert_eq!(*ci, MyField::one());
            } else {
                assert_eq!(*ci, MyField::zero());
            }
        }
    }
}
