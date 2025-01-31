use crate::args::{RuntimeType, Variable};
use crate::error::RuntimeError;
use std::sync::Mutex;
use zkpoly_common::heap;

zkpoly_common::define_usize_id!(FunctionId);
pub type FunctionTable<T> = heap::Heap<FunctionId, Function<T>>;

pub trait RegisteredFunction<T: RuntimeType> {
    fn get_fn(&self) -> Function<T>;
}

pub enum FunctionValue<T: RuntimeType> {
    FnOnce(
        Mutex<
            Option<
                Box<
                    dyn FnOnce(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                        + Sync
                        + Send
                        + 'static,
                >,
            >,
        >,
    ),
    FnMut(
        Mutex<
            Box<
                dyn FnMut(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                    + Sync
                    + Send
                    + 'static,
            >,
        >,
    ),
    Fn(
        Box<
            dyn Fn(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ),
}

pub struct Function<T: RuntimeType> {
    pub name: String,
    pub f: FunctionValue<T>,
}

impl<T: RuntimeType> std::fmt::Debug for Function<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut st = f.debug_struct("Function");

        if let FunctionValue::FnOnce(_) = &self.f {
            st.field("mutability", &"FnOnce");
        } else {
            st.field("mutability", &"FnMut");
        }
        st.finish()
    }
}

impl<T: RuntimeType> Function<T> {
    pub fn new_once(
        name: String,
        f: Box<
            dyn FnOnce(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ) -> Self {
        Self {
            name,
            f: FunctionValue::FnOnce(Mutex::new(Some(f))),
        }
    }

    pub fn new_mut(
        name: String,
        f: Box<
            dyn FnMut(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ) -> Self {
        Self {
            name,
            f: FunctionValue::FnMut(Mutex::new(f)),
        }
    }

    pub fn new(
        name: String,
        f: Box<
            dyn Fn(Vec<&mut Variable<T>>, Vec<&Variable<T>>) -> Result<(), RuntimeError>
                + Sync
                + Send
                + 'static,
        >,
    ) -> Self {
        Self {
            name,
            f: FunctionValue::Fn(f),
        }
    }
}
