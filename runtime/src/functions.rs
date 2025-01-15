use crate::args::{RuntimeType, Variable};
use crate::error::RuntimeError;
use crate::typ::{self, Typ};
use std::any;
use std::sync::Mutex;
use zkpoly_common::digraph::external::VertexId;
use zkpoly_common::heap;

zkpoly_common::define_usize_id!(FunctionId);

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
    typ_mut: Vec<Typ>,
    typ: Vec<Typ>,
}

impl<T: RuntimeType> std::fmt::Debug for Function<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut st = f.debug_struct("Function");
        st.field("name", &self.name).field("typ", &self.typ);

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
        typ_mut: Vec<Typ>,
        typ: Vec<Typ>,
    ) -> Self {
        Self {
            name,
            f: FunctionValue::FnOnce(Mutex::new(Some(f))),
            typ_mut,
            typ,
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
        typ_mut: Vec<Typ>,
        typ: Vec<Typ>,
    ) -> Self {
        Self {
            name,
            f: FunctionValue::FnMut(Mutex::new(f)),
            typ_mut,
            typ,
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
        typ_mut: Vec<Typ>,
        typ: Vec<Typ>,
    ) -> Self {
        Self {
            name,
            f: FunctionValue::Fn(f),
            typ_mut,
            typ,
        }
    }
}

pub type FunctionTable<T> = heap::Heap<FunctionId, Function<T>>;
