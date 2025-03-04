// this file is used to show how to insert a new function into the runtime

use std::{any::type_name, marker::PhantomData, os::raw::c_uint};

use libloading::Symbol;

use zkpoly_runtime::{
    args::{RuntimeType, Variable},
    error::RuntimeError,
};

use zkpoly_runtime::functions::{Function, FunctionValue, RegisteredFunction};

use super::build_func::{resolve_type, xmake_config, xmake_run};

use zkpoly_common::load_dynamic::Libs;

// c = a + b
pub struct SimpleFunc<T: RuntimeType> {
    _marker: PhantomData<T>,
    c_func:
        Symbol<'static, unsafe extern "C" fn(a: *const c_uint, b: *const c_uint, c: *mut c_uint)>,
}

impl<T: RuntimeType> SimpleFunc<T> {
    pub fn new(libs: &mut Libs) -> Self {
        if !libs.contains("libsimple_add.so") {
            // compile the dynamic library according to the template
            let field_type = resolve_type(type_name::<T::Field>());
            xmake_config("SIMPLE_ADD_FIELD", field_type);
            xmake_run("simple_add");
        }

        // load the dynamic library
        let lib = libs.load("libsimple_add.so");
        // get the function pointer
        let c_func = unsafe { lib.get(b"simple_add\0") }.unwrap();
        Self {
            _marker: PhantomData,
            c_func,
        }
    }
}

impl<T: RuntimeType> RegisteredFunction<T> for SimpleFunc<T> {
    fn get_fn(&self) -> Function<T> {
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
