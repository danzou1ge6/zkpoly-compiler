mod common;
use common::*;
use group::ff::Field;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_core::tutorial::SimpleFunc;
use zkpoly_runtime::{
    args::Variable,
    functions::{FunctionValue, RegisteredFunction},
    scalar::Scalar,
};

#[test]
fn test_simple_func() {
    let mut libs = Libs::new();
    let simple_func = SimpleFunc::<MyRuntimeType>::new(&mut libs);
    let f = simple_func.get_fn();

    let mut a = Variable::Scalar(Scalar::new_cpu());
    let mut b = Variable::Scalar(Scalar::new_cpu());
    let mut c = Variable::Scalar(Scalar::new_cpu());

    let f = match f.f {
        FunctionValue::Fn(f) => f,
        _ => unreachable!(),
    };

    for _ in 0..100 {
        let a_in = MyField::random(rand_core::OsRng);
        let b_in = MyField::random(rand_core::OsRng);

        *a.unwrap_scalar_mut().as_mut() = a_in.clone();
        *b.unwrap_scalar_mut().as_mut() = b_in.clone();

        f(vec![&mut c], vec![&a, &b]).unwrap();

        assert_eq!(*c.unwrap_scalar().as_ref(), a_in + b_in);
    }
}
