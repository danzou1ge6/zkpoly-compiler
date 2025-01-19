// this file is used to show how to insert a new function into the runtime

use std::{any::type_name, marker::PhantomData, os::raw::c_uint};

use libloading::Symbol;

use crate::{
    args::{RuntimeType, Variable},
    error::RuntimeError,
};

use super::{
    build_func::{resolve_type, xmake_config, xmake_run},
    load_dynamic::Libs,
    Function, FunctionValue, RegisteredFunction,
};

// c = a + b
pub struct SimpleFunc<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func:
        Symbol<'static, unsafe extern "C" fn(a: *const c_uint, b: *const c_uint, c: *mut c_uint)>,
}

impl<T: RuntimeType> RegisteredFunction<T> for SimpleFunc<T> {
    fn new(libs: &mut Libs) -> Self {
        // compile the dynamic library according to the template
        let field_type = resolve_type(type_name::<T::Field>());
        xmake_config("SIMPLE_ADD_FIELD", field_type);
        xmake_run("simple_add");

        // load the dynamic library
        let lib = libs.load("../lib/libsimple_add.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"simple_add\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }

    fn get_fn(&self) -> super::Function<T> {
        let c_func = self.c_func.clone();

        // define the rust side wrapper function
        let rust_func = move |mut mut_var: Vec<&mut Variable<T>>,
                              var: Vec<&Variable<T>>|
              -> Result<(), RuntimeError> {
            assert!(mut_var.len() == 1);
            assert!(var.len() == 2);
            let c = mut_var[0].unwrap_scalar_mut();
            let a = var[0].unwrap_scalar();
            let b = var[1].unwrap_scalar();
            unsafe {
                (c_func)(
                    a.value as *const c_uint,
                    b.value as *const c_uint,
                    c.value as *mut c_uint,
                );
            }
            Ok(())
        };
        Function {
            name: "simple_add".to_string(),
            f: FunctionValue::Fn(Box::new(rust_func)),
        }
    }
}

#[test]
fn test_simple_func() {
    use crate::args::Variable;
    use crate::scalar::Scalar;
    use crate::transcript::{Blake2bWrite, Challenge255};
    use group::ff::Field;
    use halo2curves::bn256;

    #[derive(Debug, Clone)]
    struct MyRuntimeType;

    impl RuntimeType for MyRuntimeType {
        type Field = bn256::Fr;
        type PointAffine = bn256::G1Affine;
        type Challenge = Challenge255<bn256::G1Affine>;
        type Trans = Blake2bWrite<Vec<u8>, bn256::G1Affine, Challenge255<bn256::G1Affine>>;
    }

    type MyField = <MyRuntimeType as RuntimeType>::Field;

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
