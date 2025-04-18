mod common;
use common::*;

static K: u32 = 20;

use group::ff::BatchInvert;
use group::ff::Field;
use halo2_proofs::arithmetic::kate_division;
use rand_core::OsRng;
use rand_core::RngCore;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_core::poly::*;
use zkpoly_cuda_api::stream::CudaEvent;
use zkpoly_cuda_api::stream::CudaStream;
use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_runtime::args::Variable;
use zkpoly_runtime::devices::DeviceType;
use zkpoly_runtime::devices::Event;
use zkpoly_runtime::functions::RegisteredFunction;
use zkpoly_runtime::functions::*;
use zkpoly_runtime::gpu_buffer::GpuBuffer;
use zkpoly_runtime::runtime::transfer::Transfer;
use zkpoly_runtime::scalar::Scalar;
use zkpoly_runtime::scalar::ScalarArray;

fn test_binary(
    new: fn(&mut Libs) -> Box<dyn RegisteredFunction<MyRuntimeType>>,
    truth_func: fn(MyField, MyField) -> MyField,
) {
    let mut libs = Libs::new();
    let poly_add = new(&mut libs);
    let func = poly_add.get_fn();
    let f = match func {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => panic!("expected Fn"),
    };
    let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
    let stream = Variable::Stream(CudaStream::new(0));
    let len = 1 << K;
    let mut a = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut b = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut res = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut rng = XorShiftRng::from_seed([0; 16]);
    for i in 0..len {
        a[i] = MyField::random(&mut rng);
        b[i] = MyField::random(&mut rng);
    }
    let mut a_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));
    a_d.unwrap_scalar_array_mut().rotate(10); // test rotation
    let mut b_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));
    b_d.unwrap_scalar_array_mut().rotate(-10); // test rotation
    let mut res_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));
    res_d.unwrap_scalar_array_mut().rotate(3); // test rotation
    a.cpu2gpu(a_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
    b.cpu2gpu(b_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
    f(vec![&mut res_d], vec![&a_d, &b_d, &stream]).unwrap();
    res_d
        .unwrap_scalar_array_mut()
        .gpu2cpu(&mut res, stream.unwrap_stream());
    stream
        .unwrap_stream()
        .free(a_d.unwrap_scalar_array().values);
    stream
        .unwrap_stream()
        .free(b_d.unwrap_scalar_array().values);
    stream
        .unwrap_stream()
        .free(res_d.unwrap_scalar_array().values);
    stream.unwrap_stream().sync();
    for ((a, b), r) in a.iter().zip(b.iter()).zip(res.iter()) {
        assert_eq!(*r, truth_func(*a, *b));
    }
}

#[test]
fn test_add() {
    test_binary(
        |libs| -> Box<dyn RegisteredFunction<MyRuntimeType>> {
            Box::new(PolyAdd::<MyRuntimeType>::new(libs))
        },
        |a, b| a + b,
    );
}

#[test]
fn test_sub() {
    test_binary(
        |libs| -> Box<dyn RegisteredFunction<MyRuntimeType>> {
            Box::new(PolySub::<MyRuntimeType>::new(libs))
        },
        |a, b| a - b,
    );
}

#[test]
fn test_mul() {
    test_binary(
        |libs| -> Box<dyn RegisteredFunction<MyRuntimeType>> {
            Box::new(PolyMul::<MyRuntimeType>::new(libs))
        },
        |a, b| a * b,
    );
}

#[test]
fn test_eval() {
    let len = 1 << K;
    let mut libs = Libs::new();
    let poly_eval = PolyEval::<MyRuntimeType>::new(&mut libs);
    let func = match poly_eval.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => unreachable!(),
    };
    let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
    let stream = Variable::Stream(CudaStream::new(0));
    let mut poly = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut x = Scalar::new_cpu();
    let mut res = Scalar::new_cpu();
    let mut rng = XorShiftRng::from_seed([0; 16]);
    for i in 0..len {
        poly[i] = MyField::random(&mut rng);
    }
    *x.as_mut() = MyField::random(&mut rng);
    let mut poly_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));
    poly_d.unwrap_scalar_array_mut().rotate(10); // test rotation
    let mut x_d = Variable::Scalar(Scalar::new_gpu(stream.unwrap_stream().allocate(1), 0));
    let mut res_d = Variable::Scalar(Scalar::new_gpu(stream.unwrap_stream().allocate(1), 0));
    let buf_size = poly_eval.get_buffer_size(len);
    let mut temp_buf = Variable::GpuBuffer(GpuBuffer::new(
        stream.unwrap_stream().allocate(buf_size),
        buf_size,
    ));

    poly.cpu2gpu(poly_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
    x.cpu2gpu(x_d.unwrap_scalar_mut(), stream.unwrap_stream());
    func(
        vec![&mut temp_buf, &mut res_d],
        vec![&poly_d, &x_d, &stream],
    )
    .unwrap();
    res_d
        .unwrap_scalar_mut()
        .gpu2cpu(&mut res, stream.unwrap_stream());
    stream
        .unwrap_stream()
        .free(poly_d.unwrap_scalar_array().values);
    stream.unwrap_stream().free(x_d.unwrap_scalar().value);
    stream.unwrap_stream().free(res_d.unwrap_scalar().value);
    stream
        .unwrap_stream()
        .free(temp_buf.unwrap_gpu_buffer().ptr);
    stream.unwrap_stream().sync();
    let mut truth = MyField::zero();
    for i in (0..len).rev() {
        truth = truth * *x.as_ref() + poly[i];
    }
    assert_eq!(*res.as_ref(), truth);
}

#[test]
fn test_kate() {
    let log_len = K;
    let len = 1 << K;
    let mut libs = Libs::new();
    let kate = KateDivision::<MyRuntimeType>::new(&mut libs);
    let func = match kate.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => unreachable!(),
    };
    let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
    let stream = Variable::Stream(CudaStream::new(0));
    let mut poly = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut b = Scalar::new_cpu();
    let mut res = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut rng = XorShiftRng::from_seed([0; 16]);
    for i in 0..len {
        poly[i] = MyField::random(&mut rng);
    }
    *b.as_mut() = MyField::random(&mut rng);
    let mut poly_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));
    poly_d.unwrap_scalar_array_mut().rotate(2);
    let mut b_d = Variable::Scalar(Scalar::new_gpu(stream.unwrap_stream().allocate(1), 0));
    let mut res_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));
    res_d.unwrap_scalar_array_mut().rotate(-2);
    let buf_size = kate.get_buffer_size(log_len);
    let mut temp_buf = Variable::GpuBuffer(GpuBuffer::new(
        stream.unwrap_stream().allocate(buf_size),
        buf_size,
    ));

    poly.cpu2gpu(poly_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
    b.cpu2gpu(b_d.unwrap_scalar_mut(), stream.unwrap_stream());
    func(
        vec![&mut temp_buf, &mut res_d],
        vec![&poly_d, &b_d, &stream],
    )
    .unwrap();
    res_d
        .unwrap_scalar_array()
        .gpu2cpu(&mut res, stream.unwrap_stream());
    stream
        .unwrap_stream()
        .free(poly_d.unwrap_scalar_array().values);
    stream.unwrap_stream().free(b_d.unwrap_scalar().value);
    stream
        .unwrap_stream()
        .free(res_d.unwrap_scalar_array().values);
    stream
        .unwrap_stream()
        .free(temp_buf.unwrap_gpu_buffer().ptr);
    stream.unwrap_stream().sync();

    let truth = kate_division(poly.as_ref(), *b.as_ref());
    for i in 0..len - 1 {
        assert_eq!(res[i], truth[i]);
    }
}

#[test]
fn test_zero_one() {
    let len = 1 << K;
    let mut libs = Libs::new();
    let zero = PolyZero::<MyRuntimeType>::new(&mut libs);
    let func = match zero.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => unreachable!(),
    };
    let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
    let stream = Variable::Stream(CudaStream::new(0));
    let mut poly = Variable::ScalarArray(ScalarArray::new(
        len,
        cpu_pool.allocate(len),
        DeviceType::CPU,
    ));

    let mut poly_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));

    func(vec![&mut poly_d], vec![&stream]).unwrap();
    poly_d
        .unwrap_scalar_array()
        .gpu2cpu(poly.unwrap_scalar_array_mut(), stream.unwrap_stream());

    stream.unwrap_stream().sync();

    let truth: Vec<_> = (0..len).into_iter().map(|_| MyField::ZERO).collect();
    for i in 0..len {
        assert_eq!(poly.unwrap_scalar_array_mut()[i], truth[i]);
    }

    // cpu ver
    func(vec![&mut poly], vec![]).unwrap();
    for i in 0..len {
        assert_eq!(poly.unwrap_scalar_array_mut()[i], truth[i]);
    }

    let one = PolyOneLagrange::<MyRuntimeType>::new(&mut libs);
    let func = match one.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => unreachable!(),
    };

    func(vec![&mut poly_d], vec![&stream]).unwrap();
    poly_d
        .unwrap_scalar_array()
        .gpu2cpu(poly.unwrap_scalar_array_mut(), stream.unwrap_stream());

    stream.unwrap_stream().sync();

    let truth: Vec<_> = (0..len).into_iter().map(|_| MyField::ONE).collect();
    for i in 0..len {
        assert_eq!(poly.unwrap_scalar_array_mut()[i], truth[i]);
    }

    // cpu ver
    func(vec![&mut poly], vec![]).unwrap();
    for i in 0..len {
        assert_eq!(poly.unwrap_scalar_array_mut()[i], truth[i]);
    }

    let one_coef = PolyOneCoef::<MyRuntimeType>::new(&mut libs);
    let func = match one_coef.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => unreachable!(),
    };

    func(vec![&mut poly_d], vec![&stream]).unwrap();
    poly_d
        .unwrap_scalar_array()
        .gpu2cpu(poly.unwrap_scalar_array_mut(), stream.unwrap_stream());

    stream.unwrap_stream().sync();

    let mut truth: Vec<_> = (0..len).into_iter().map(|_| MyField::ZERO).collect();
    truth[0] = MyField::ONE;
    for i in 0..len {
        assert_eq!(poly.unwrap_scalar_array_mut()[i], truth[i]);
    }

    // cpu ver
    func(vec![&mut poly], vec![]).unwrap();
    for i in 0..len {
        assert_eq!(poly.unwrap_scalar_array_mut()[i], truth[i]);
    }

    stream
        .unwrap_stream()
        .free(poly_d.unwrap_scalar_array().values);
}

#[test]
fn test_scan() {
    let len = 1 << K;
    let mut libs = Libs::new();
    let scan = PolyScan::<MyRuntimeType>::new(&mut libs);
    let func = match scan.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => unreachable!(),
    };
    let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
    let stream = Variable::Stream(CudaStream::new(0));
    let mut poly = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut x0 = Scalar::new_cpu();
    let mut res = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut rng = XorShiftRng::from_seed([0; 16]);
    for i in 0..len {
        poly[i] = MyField::random(&mut rng);
    }
    *x0.as_mut() = MyField::random(&mut rng);
    let mut poly_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));
    poly_d.unwrap_scalar_array_mut().rotate(3);
    let mut x0_d = Variable::Scalar(Scalar::new_gpu(stream.unwrap_stream().allocate(1), 0));
    let mut res_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));
    res_d.unwrap_scalar_array_mut().rotate(-5);
    let buf_size = scan.get_buffer_size(len.try_into().unwrap());
    let mut temp_buf = Variable::GpuBuffer(GpuBuffer::new(
        stream.unwrap_stream().allocate(buf_size),
        buf_size,
    ));

    poly.cpu2gpu(poly_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
    x0.cpu2gpu(x0_d.unwrap_scalar_mut(), stream.unwrap_stream());
    // add timer
    let start = CudaEvent::new();
    let end = CudaEvent::new();
    stream.unwrap_stream().record(&start);
    func(
        vec![&mut temp_buf, &mut res_d],
        vec![&poly_d, &x0_d, &stream],
    )
    .unwrap();
    stream.unwrap_stream().record(&end);
    end.sync();
    let elapsed = start.elapsed(&end);
    println!("Elapsed time: {:?} ms", elapsed);
    res_d
        .unwrap_scalar_array()
        .gpu2cpu(&mut res, stream.unwrap_stream());
    stream
        .unwrap_stream()
        .free(poly_d.unwrap_scalar_array().values);
    stream.unwrap_stream().free(x0_d.unwrap_scalar().value);
    stream
        .unwrap_stream()
        .free(res_d.unwrap_scalar_array().values);
    stream
        .unwrap_stream()
        .free(temp_buf.unwrap_gpu_buffer().ptr);
    stream.unwrap_stream().sync();

    assert_eq!(res[0], *x0.as_ref());
    for i in 1..len {
        assert_eq!(res[i], res[i - 1] * poly[i - 1]);
    }
}

#[test]
fn test_invert() {
    let len = 1 << K;
    let mut libs = Libs::new();
    let invert = PolyInvert::<MyRuntimeType>::new(&mut libs);
    let func = match invert.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => unreachable!(),
    };
    let cpu_pool = PinnedMemoryPool::new(K, size_of::<MyField>());
    let stream = Variable::Stream(CudaStream::new(0));
    let mut poly = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut res = ScalarArray::new(len, cpu_pool.allocate(len), DeviceType::CPU);
    let mut rng = XorShiftRng::from_seed([0; 16]);
    for i in 0..len {
        poly[i] = if i % 17 == 0 {
            MyField::ZERO
        } else {
            MyField::random(&mut rng)
        };
    }
    let mut poly_d = Variable::ScalarArray(ScalarArray::new(
        len,
        stream.unwrap_stream().allocate(len),
        DeviceType::GPU { device_id: 0 },
    ));

    let buf_size = invert.get_buffer_size(len.try_into().unwrap());
    let mut temp_buf = Variable::GpuBuffer(GpuBuffer::new(
        stream.unwrap_stream().allocate(buf_size),
        buf_size,
    ));

    poly.cpu2gpu(poly_d.unwrap_scalar_array_mut(), stream.unwrap_stream());
    func(vec![&mut temp_buf, &mut poly_d], vec![&stream]).unwrap();
    poly_d
        .unwrap_scalar_array()
        .gpu2cpu(&mut res, stream.unwrap_stream());
    stream
        .unwrap_stream()
        .free(poly_d.unwrap_scalar_array().values);
    stream
        .unwrap_stream()
        .free(temp_buf.unwrap_gpu_buffer().ptr);
    stream.unwrap_stream().sync();

    let _ = poly.as_mut().batch_invert();

    for i in 0..len - 1 {
        assert_eq!(res[i], poly[i]);
    }
}

#[test]
fn test_invert_scalar() {
    let mut libs = Libs::new();
    let invert = ScalarInv::<MyRuntimeType>::new(&mut libs);
    let f = match invert.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => panic!(),
    };
    let mut scalar = Variable::<MyRuntimeType>::Scalar(Scalar::new_cpu());

    let mut rand_scalar = MyField::random(OsRng);
    while rand_scalar == MyField::ZERO {
        rand_scalar = MyField::random(OsRng);
    }
    let truth = rand_scalar.invert().unwrap();
    *scalar.unwrap_scalar_mut().as_mut() = rand_scalar;

    let stream = CudaStream::new(0);

    let mut gpu_scalar = Variable::<MyRuntimeType>::Scalar(Scalar::new_gpu(stream.allocate(1), 0));

    scalar
        .unwrap_scalar()
        .cpu2gpu(gpu_scalar.unwrap_scalar_mut(), &stream);

    stream.sync();

    f(vec![&mut scalar], vec![]).unwrap();

    assert_eq!(*scalar.unwrap_scalar().as_ref(), truth);

    *scalar.unwrap_scalar_mut().as_mut() = MyField::ZERO;

    f(
        vec![&mut gpu_scalar],
        vec![&Variable::Stream(stream.clone())],
    )
    .unwrap();

    gpu_scalar
        .unwrap_scalar()
        .gpu2cpu(scalar.unwrap_scalar_mut(), &stream);
    stream.sync();

    assert_eq!(*scalar.unwrap_scalar().as_ref(), truth);
}

#[test]
fn test_pow_scalar() {
    let mut libs = Libs::new();
    let exp_u64 = OsRng.next_u64();

    let pow = ScalarPow::<MyRuntimeType>::new(&mut libs, exp_u64);
    let f = match pow.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => panic!(),
    };
    let mut scalar = Variable::<MyRuntimeType>::Scalar(Scalar::new_cpu());
    let mut rand_scalar = MyField::random(OsRng);
    while rand_scalar == MyField::ZERO {
        rand_scalar = MyField::random(OsRng);
    }
    let truth = rand_scalar.pow(vec![exp_u64]);

    *scalar.unwrap_scalar_mut().as_mut() = rand_scalar;

    let stream = CudaStream::new(0);

    let mut gpu_scalar = Variable::<MyRuntimeType>::Scalar(Scalar::new_gpu(stream.allocate(1), 0));

    scalar
        .unwrap_scalar()
        .cpu2gpu(gpu_scalar.unwrap_scalar_mut(), &stream);

    stream.sync();

    f(vec![&mut scalar], vec![]).unwrap();

    assert_eq!(*scalar.unwrap_scalar().as_ref(), truth);

    *scalar.unwrap_scalar_mut().as_mut() = MyField::ZERO;

    f(
        vec![&mut gpu_scalar],
        vec![&Variable::Stream(stream.clone())],
    )
    .unwrap();

    gpu_scalar
        .unwrap_scalar()
        .gpu2cpu(scalar.unwrap_scalar_mut(), &stream);
    stream.sync();

    assert_eq!(*scalar.unwrap_scalar().as_ref(), truth);
}
