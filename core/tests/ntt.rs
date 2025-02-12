mod common;
use common::*;

static MAX_K: u32 = 20;

use group::ff::Field;
use halo2_proofs::arithmetic;
use rand_core::OsRng;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_core::ntt::*;
use zkpoly_cuda_api::stream::CudaStream;
use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_runtime::args::Variable;
use zkpoly_runtime::devices::DeviceType;
use zkpoly_runtime::functions::*;
use zkpoly_runtime::runtime::transfer::Transfer;
use zkpoly_runtime::scalar::ScalarArray;

#[test]
fn test_ssip_ntt() {
    let mut libs = Libs::new();
    let ntt = SsipNtt::<MyRuntimeType>::new(&mut libs);
    let precompute = SsipPrecompute::<MyRuntimeType>::new(&mut libs);
    let ntt_fn = match ntt.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => panic!("expected Fn"),
    };

    let precompute_fn = precompute.get_fn();
    let cpu_alloc = PinnedMemoryPool::new(MAX_K, size_of::<MyField>());
    let stream = Variable::<MyRuntimeType>::Stream(CudaStream::new(0));

    for k in 1..=MAX_K {
        println!("generating data for k = {k}...");
        let mut data_rust: Vec<_> = (0..(1 << k))
            .into_iter()
            .map(|_| MyField::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();

        let mut poly_cpu =
            ScalarArray::<MyField>::new(1 << k, cpu_alloc.allocate(1 << k), DeviceType::CPU);

        unsafe {
            std::ptr::copy_nonoverlapping(data_rust.as_ptr(), poly_cpu.values, 1 << k);
        }

        let omega = MyField::random(XorShiftRng::from_rng(OsRng).unwrap()); // would be weird if this mattered

        println!("testing on cpu for k = {k}...");
        arithmetic::best_fft(&mut data_rust, omega, k as u32);

        println!("precomputing twiddle factors for k = {k}...");

        let mut twiddle = ScalarArray::<MyField>::new(
            1 << (k - 1),
            cpu_alloc.allocate(1 << (k - 1)),
            DeviceType::CPU,
        );
        precompute_fn(&mut twiddle, &omega).unwrap();

        println!("testing on gpu for k = {k}...");

        let start = std::time::Instant::now();

        let ptr_data: *mut MyField = stream.unwrap_stream().allocate(1 << k);
        let ptr_twiddle: *mut MyField = stream.unwrap_stream().allocate(1 << (k - 1));

        let mut poly_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
            1 << k,
            ptr_data,
            DeviceType::GPU { device_id: 0 },
        ));
        poly_gpu.unwrap_scalar_array_mut().rotate(3);
        let mut twiddle_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
            1 << (k - 1),
            ptr_twiddle,
            DeviceType::GPU { device_id: 0 },
        ));

        poly_cpu.cpu2gpu(poly_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());
        twiddle.cpu2gpu(
            twiddle_gpu.unwrap_scalar_array_mut(),
            stream.unwrap_stream(),
        );

        ntt_fn(vec![&mut poly_gpu], vec![&twiddle_gpu, &stream]).unwrap();

        poly_gpu
            .unwrap_scalar_array()
            .gpu2cpu(&mut poly_cpu, stream.unwrap_stream());

        let end1 = std::time::Instant::now();

        stream.unwrap_stream().sync();

        let end2 = std::time::Instant::now();
        let dur1 = end1 - start;
        let dur2 = end2 - start;

        println!(
            "time for k = {k}: {:?} (launch) + {:?} (actual)",
            dur1, dur2
        );

        stream.unwrap_stream().free(ptr_data);
        stream.unwrap_stream().free(ptr_twiddle);

        println!("comparing results for k = {k}...");

        let data_cuda = unsafe { std::slice::from_raw_parts(poly_cpu.values, 1 << k) };

        data_cuda.iter().zip(data_rust.iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });

        cpu_alloc.free(poly_cpu.values);
        cpu_alloc.free(twiddle.values);
    }
}

#[test]
fn test_recompute_ntt() {
    let mut libs = Libs::new();
    let ntt = RecomputeNtt::<MyRuntimeType>::new(&mut libs);
    let precompute = GenPqOmegas::<MyRuntimeType>::new(&mut libs);
    let ntt_fn = match ntt.get_fn() {
        Function {
            f: FunctionValue::Fn(func),
            ..
        } => func,
        _ => panic!("expected Fn"),
    };

    let precompute_fn = precompute.get_fn();
    let cpu_alloc = PinnedMemoryPool::new(MAX_K, size_of::<MyField>());
    let stream = Variable::<MyRuntimeType>::Stream(CudaStream::new(0));

    for k in 1..=MAX_K {
        println!("generating data for k = {k}...");
        let mut data_rust: Vec<_> = (0..(1 << k))
            .into_iter()
            .map(|_| MyField::random(XorShiftRng::from_rng(OsRng).unwrap()))
            .collect();

        let mut poly_cpu =
            ScalarArray::<MyField>::new(1 << k, cpu_alloc.allocate(1 << k), DeviceType::CPU);

        unsafe {
            std::ptr::copy_nonoverlapping(data_rust.as_ptr(), poly_cpu.values, 1 << k);
        }

        let omega = MyField::random(XorShiftRng::from_rng(OsRng).unwrap()); // would be weird if this mattered

        println!("testing on cpu for k = {k}...");
        arithmetic::best_fft(&mut data_rust, omega, k as u32);

        println!("precomputing twiddle factors for k = {k}...");

        let mut pq = ScalarArray::<MyField>::new(
            precompute.get_pq_len(k),
            cpu_alloc.allocate(precompute.get_pq_len(k)),
            DeviceType::CPU,
        );
        let mut omegas = ScalarArray::<MyField>::new(32, cpu_alloc.allocate(32), DeviceType::CPU);
        precompute_fn(&mut pq, &mut omegas, 1 << k, &omega).unwrap();

        println!("transferring data to gpu for k = {k}...");

        let ptr_data: *mut MyField = stream.unwrap_stream().allocate(1 << k);
        let ptr_pq: *mut MyField = stream.unwrap_stream().allocate(precompute.get_pq_len(k));
        let ptr_omegas: *mut MyField = stream.unwrap_stream().allocate(32);

        let mut poly_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
            1 << k,
            ptr_data,
            DeviceType::GPU { device_id: 0 },
        ));
        poly_gpu.unwrap_scalar_array_mut().rotate(-2);
        let mut pq_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
            precompute.get_pq_len(k),
            ptr_pq,
            DeviceType::GPU { device_id: 0 },
        ));
        let mut omegas_gpu = Variable::ScalarArray(ScalarArray::<MyField>::new(
            32,
            ptr_omegas,
            DeviceType::GPU { device_id: 0 },
        ));

        println!("testing on gpu for k = {k}...");

        poly_cpu.cpu2gpu(poly_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());
        pq.cpu2gpu(pq_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());
        omegas.cpu2gpu(omegas_gpu.unwrap_scalar_array_mut(), stream.unwrap_stream());

        ntt_fn(vec![&mut poly_gpu], vec![&pq_gpu, &omegas_gpu, &stream]).unwrap();

        poly_gpu
            .unwrap_scalar_array()
            .gpu2cpu(&mut poly_cpu, stream.unwrap_stream());

        stream.unwrap_stream().sync();
        stream.unwrap_stream().free(ptr_data);
        stream.unwrap_stream().free(ptr_pq);
        stream.unwrap_stream().free(ptr_omegas);

        println!("comparing results for k = {k}...");

        let data_cuda = unsafe { std::slice::from_raw_parts(poly_cpu.values, 1 << k) };

        data_cuda.iter().zip(data_rust.iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });
        cpu_alloc.free(poly_cpu.values);
        cpu_alloc.free(pq.values);
        cpu_alloc.free(omegas.values);
    }
}
