mod common;
use std::ptr::null_mut;
use std::time::SystemTime;
use std::{os::raw::c_void, sync::Arc};

use common::*;

static MAX_K: u8 = 30;
static BATCHES: u32 = 4;

pub type MyCurve = <MyRuntimeType as RuntimeType>::PointAffine;

use group::{
    ff::{Field, PrimeField},
    prime::PrimeCurveAffine,
};
mod msm_halo2;
use ::halo2curves::bn256::{Fr as Scalar, G1Affine as Point};
use rand_core::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::{current_thread_index, prelude::*};
use zkpoly_common::{devices::DeviceType, load_dynamic::Libs};
use zkpoly_core::msm::*;
use zkpoly_cuda_api::bindings::{cudaFree, cudaMalloc};
use zkpoly_memory_pool::CpuMemoryPool;
use zkpoly_runtime::{
    args::{RuntimeType, Variable},
    functions::*,
    gpu_buffer::GpuBuffer,
    point::PointArray,
    scalar::ScalarArray,
};

const SEED: [u8; 16] = [
    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc, 0xe5,
];

fn generate_curvepoints(k: u8) -> Vec<Point> {
    if k <= 20 {
        let n: u64 = {
            assert!(k < 64);
            1 << k
        };

        println!("Generating 2^{k} = {n} curve points..",);
        let timer = SystemTime::now();
        let bases = (0..n)
            .into_par_iter()
            .map_init(
                || {
                    let mut thread_seed = SEED;
                    let uniq = current_thread_index().unwrap().to_ne_bytes();
                    assert!(std::mem::size_of::<usize>() == 8);
                    for i in 0..uniq.len() {
                        thread_seed[i] += uniq[i];
                        thread_seed[i + 8] += uniq[i];
                    }
                    XorShiftRng::from_seed(thread_seed)
                },
                |rng, _| Point::random(rng),
            )
            .collect();
        let end = timer.elapsed().unwrap();
        println!(
            "Generating 2^{k} = {n} curve points took: {} sec.\n\n",
            end.as_secs()
        );
        bases
    } else {
        let data = generate_curvepoints(k - 1);
        data.iter().chain(data.iter()).cloned().collect()
    }
}

fn generate_coefficients(k: u8, bits: usize) -> Vec<Scalar> {
    if k <= 20 {
        let n: u64 = {
            assert!(k < 64);
            1 << k
        };
        let max_val: Option<u128> = match bits {
            1 => Some(1),
            8 => Some(0xff),
            16 => Some(0xffff),
            32 => Some(0xffff_ffff),
            64 => Some(0xffff_ffff_ffff_ffff),
            128 => Some(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff),
            256 => None,
            _ => panic!("unexpected bit size {}", bits),
        };

        println!("Generating 2^{k} = {n} coefficients..",);
        let timer = SystemTime::now();
        let coeffs = (0..n)
            .into_par_iter()
            .map_init(
                || {
                    let mut thread_seed = SEED;
                    let uniq = current_thread_index().unwrap().to_ne_bytes();
                    assert!(std::mem::size_of::<usize>() == 8);
                    for i in 0..uniq.len() {
                        thread_seed[i] += uniq[i];
                        thread_seed[i + 8] += uniq[i];
                    }
                    XorShiftRng::from_seed(thread_seed)
                },
                |rng, _| {
                    if let Some(max_val) = max_val {
                        let v_lo = rng.next_u64() as u128;
                        let v_hi = rng.next_u64() as u128;
                        let mut v = v_lo + (v_hi << 64);
                        v &= max_val; // Mask the 128bit value to get a lower number of bits
                        Scalar::from_u128(v)
                    } else {
                        Scalar::random(rng)
                    }
                },
            )
            .collect();
        let end = timer.elapsed().unwrap();
        println!(
            "Generating 2^{k} = {n} coefficients took: {} sec.\n\n",
            end.as_secs()
        );
        coeffs
    } else {
        let data = generate_coefficients(k - 1, bits);
        data.iter().chain(data.iter()).cloned().collect()
    }
}

#[test]
fn test_msm() {
    let mut libs = Libs::new();

    let mut cpu_alloc = CpuMemoryPool::new((MAX_K + 1) as u32, size_of::<MyField>());

    for k in (20..=MAX_K).step_by(2) {
        let msm_config = get_best_config::<MyRuntimeType>(1 << k, BATCHES, 4 * (1 << 30));
        let msm = MSM::<MyRuntimeType>::new(&mut libs, msm_config.clone());
        let msm_precompute = MSMPrecompute::<MyRuntimeType>::new(&mut libs, msm_config.clone());

        let msm_fn = msm.get_fn().f;

        let precompute_fn = msm_precompute.get_fn();

        println!("generating data for k = {k}...");
        let bases: Vec<Point> = generate_curvepoints(k);
        let bits = [256];
        let coeffs: Vec<_> = bits.iter().map(|b| generate_coefficients(k, *b)).collect();

        println!("testing for k = {k}:");
        let n: usize = 1 << k;

        let start = std::time::Instant::now();
        let cpu_result = msm_halo2::best_multiexp(&coeffs[0][..n], &bases[..n]);
        let end = std::time::Instant::now();
        println!("cpu time for k = {k}: {:?}", end - start);

        let n_precompute = msm_config.get_precompute();
        println!("n_precompute: {n_precompute}");
        let mut var = (0..n_precompute)
            .map(|_| {
                Variable::PointArray::<MyRuntimeType>(PointArray::new(
                    n,
                    cpu_alloc.allocate(n),
                    DeviceType::CPU,
                ))
            })
            .collect::<Vec<_>>();

        unsafe {
            std::ptr::copy_nonoverlapping(
                bases.as_ptr(),
                var[0].unwrap_point_array_mut().values,
                n,
            );
        }

        let scalars: *mut MyField = cpu_alloc.allocate(n);
        unsafe {
            std::ptr::copy_nonoverlapping(coeffs[0].as_ptr(), scalars, n);
        }

        let mut mut_var = Vec::new();

        let buffer_sizes = msm.get_buffer_size(n);

        for i in 0..msm_config.cards.len() {
            let buffer_size = buffer_sizes[i];
            let mut ptr = null_mut();
            unsafe {
                cudaMalloc(&mut ptr as *mut *mut u8 as *mut *mut c_void, buffer_size);
            }
            mut_var.push(Variable::GpuBuffer::<MyRuntimeType>(GpuBuffer::new(
                ptr,
                buffer_size,
                DeviceType::GPU { device_id: 0 },
            )));
        }

        for _i in 0..BATCHES {
            mut_var.push(Variable::Point::<MyRuntimeType>(
                zkpoly_runtime::point::Point::new(MyCurve::identity()),
            ));
        }

        println!("precomputing for k = {k}...");
        precompute_fn(
            var.iter_mut().map(|p| p.unwrap_point_array_mut()).collect(),
            n,
            4,
        )
        .unwrap();

        for _i in 0..BATCHES {
            var.push(Variable::<MyRuntimeType>::ScalarArray(ScalarArray::new(
                n,
                scalars,
                DeviceType::CPU,
            )));
        }

        println!("running msm for k = {k}...");
        let start = std::time::Instant::now();
        msm_fn(
            mut_var.iter_mut().map(|v| v).collect::<Vec<_>>(),
            var.iter().map(|v| v).collect::<Vec<_>>(),
            Arc::new(|x| x),
        )
        .unwrap();
        let end = std::time::Instant::now();

        println!("gpu time for k = {k}: {:?}", end - start);

        println!("checking for k = {k}...");

        for i in 0..BATCHES {
            let gpu_result = mut_var[msm_config.cards.len() + i as usize]
                .unwrap_point()
                .as_ref();
            let x1 = cpu_result.x;
            let y1 = cpu_result.y;
            let x2 = gpu_result.x * cpu_result.z;
            let y2 = gpu_result.y * cpu_result.z;
            assert_eq!(x1, x2);
            assert_eq!(y1, y2);
        }

        cpu_alloc.free(scalars);
        for i in 0..n_precompute {
            cpu_alloc.free(var[i as usize].unwrap_point_array_mut().values);
        }
        for i in 0..msm_config.cards.len() {
            unsafe {
                cudaFree(mut_var[i].unwrap_gpu_buffer_mut().ptr as *mut c_void);
            }
        }
    }
}

#[test]
fn test_cost_model() {
    for i in (5..=20).rev() {
        println!("testing for k = {i}...");
        let config = get_best_config::<MyRuntimeType>(2_usize.pow(i), 4, 4 * (1 << 30));
        println!("get config {:?}", config);
    }
}
