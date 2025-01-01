use crate::error::RuntimeError;
use crate::typ::Typ;
use std::any;
use zkpoly_common::heap;

zkpoly_common::define_usize_id!(FunctionId);

pub enum FunctionValue {
    FnOnce(Box<dyn FnOnce(Vec<&dyn any::Any>) -> Result<Box<dyn any::Any>, RuntimeError>>),
    FnMut(Box<dyn FnMut(Vec<&dyn any::Any>) -> Result<Box<dyn any::Any>, RuntimeError>>),
}

pub struct Function {
    name: String,
    f: FunctionValue,
    typ: (Vec<Typ>, Typ),
}

impl Function {
    pub fn ret_typ(&self) -> &Typ {
        &self.typ.1
    }
}

impl std::fmt::Debug for Function {
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

impl Function {
    pub fn new_once(
        name: String,
        f: Box<dyn FnOnce(Vec<&dyn any::Any>) -> Result<Box<dyn any::Any>, RuntimeError>>,
        typ: (Vec<Typ>, Typ),
    ) -> Self {
        Self {
            name,
            f: FunctionValue::FnOnce(f),
            typ,
        }
    }

    pub fn new_mut(
        name: String,
        f: Box<dyn FnMut(Vec<&dyn any::Any>) -> Result<Box<dyn any::Any>, RuntimeError>>,
        typ: (Vec<Typ>, Typ),
    ) -> Self {
        Self {
            name,
            f: FunctionValue::FnMut(f),
            typ,
        }
    }
}

pub type FunctionTable = heap::Heap<FunctionId, Function>;
