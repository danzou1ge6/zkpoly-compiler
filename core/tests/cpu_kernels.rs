mod common;
use std::sync::Arc;

use common::*;

static K: u32 = 5;

use group::ff::Field;
use halo2_proofs::arithmetic::lagrange_interpolate;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;
use zkpoly_core::cpu_kernels::InterpolateKernel;

use zkpoly_common::devices::DeviceType;
use zkpoly_memory_pool::CpuMemoryPool;
use zkpoly_runtime::args::Variable;
use zkpoly_runtime::functions::RegisteredFunction;
use zkpoly_runtime::scalar::{Scalar, ScalarArray};

#[test]
fn test_interpolate() {
    let inter = InterpolateKernel::<MyRuntimeType>::new();
    let func = inter.get_fn();
    let f = func.f;
    let mut cpu_pool = CpuMemoryPool::new(K, size_of::<MyField>());
    let len = 1 << K;
    let mut a = (0..len)
        .into_iter()
        .map(|_| Variable::Scalar(Scalar::new_cpu()))
        .collect::<Vec<_>>();
    let mut b = (0..len)
        .into_iter()
        .map(|_| Variable::Scalar(Scalar::new_cpu()))
        .collect::<Vec<_>>();

    let mut res = Variable::ScalarArray(ScalarArray::new(
        len,
        cpu_pool.allocate(len),
        DeviceType::CPU,
    ));
    let mut rng = XorShiftRng::from_seed([0; 16]);
    let mut a_t: Vec<MyField> = Vec::new();
    let mut b_t: Vec<MyField> = Vec::new();
    for i in 0..len {
        let ai = MyField::random(&mut rng);
        let bi = MyField::random(&mut rng);
        *a[i].unwrap_scalar_mut().as_mut() = ai.clone();
        *b[i].unwrap_scalar_mut().as_mut() = bi.clone();
        a_t.push(ai);
        b_t.push(bi);
    }

    let mut vars = Vec::new();

    for i in 0..len {
        vars.push(&a[i]);
    }
    for i in 0..len {
        vars.push(&b[i]);
    }

    f(vec![&mut res], vars, Arc::new(|x: i32| x)).unwrap();

    let truth = lagrange_interpolate(&a_t, &b_t);

    for (r, t) in res.unwrap_scalar_array_mut().iter().zip(truth.iter()) {
        assert_eq!(*r, *t);
    }

    cpu_pool.free(res.unwrap_scalar_array_mut().values);
}
