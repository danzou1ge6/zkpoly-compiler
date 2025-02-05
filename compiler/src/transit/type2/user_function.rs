use super::typ::Typ;
use zkpoly_common::define_usize_id;
use zkpoly_common::heap::Heap;
use zkpoly_runtime::args::{RuntimeType, Variable};
use zkpoly_runtime::error::RuntimeError;

pub type UserFnOnce<Rt: RuntimeType> = Box<
    dyn FnOnce(Vec<&mut Variable<Rt>>, Vec<&Variable<Rt>>) -> Result<(), RuntimeError>
        + Sync
        + Send
        + 'static,
>;
pub type UserFnMut<Rt: RuntimeType> = Box<
    dyn FnMut(Vec<&mut Variable<Rt>>, Vec<&Variable<Rt>>) -> Result<(), RuntimeError>
        + Sync
        + Send
        + 'static,
>;

pub enum Value<Rt: RuntimeType> {
    FnOnce(UserFnOnce<Rt>),
    FnMut(UserFnMut<Rt>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mutability {
    Mutable,
    Immutable,
}

#[derive(Debug, Clone)]
pub struct FunctionType<Rt: RuntimeType> {
    pub(crate) args: Vec<(Typ<Rt>, Mutability)>,
    pub(crate) ret: Typ<Rt>,
    pub(crate) ret_inplace: Vec<Option<usize>>,
}

pub struct Function<Rt: RuntimeType> {
    pub name: String,
    pub f: Value<Rt>,
    pub typ: FunctionType<Rt>,
}

impl<Rt: RuntimeType + std::fmt::Debug> std::fmt::Debug for Function<Rt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut st = f.debug_struct("Function");
        st.field("name", &self.name).field("typ", &self.typ);

        if let Value::FnOnce(_) = &self.f {
            st.field("mutability", &"FnOnce");
        } else {
            st.field("mutability", &"FnMut");
        }
        st.finish()
    }
}

define_usize_id!(Id);
pub type Table<Rt: RuntimeType> = Heap<Id, Function<Rt>>;
