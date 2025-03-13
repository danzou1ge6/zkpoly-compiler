use super::typ::Typ;
use crate::ast::{self, lowering::UserFunctionId, user_function::Value};
use zkpoly_common::define_usize_id;
use zkpoly_common::heap::Heap;
use zkpoly_runtime::args::{RuntimeType, Variable};
use zkpoly_runtime::error::RuntimeError;

#[derive(Debug, Clone)]
pub struct FunctionType {
    pub(crate) args: Vec<Mutability>,
    pub(crate) ret_inplace: Vec<Option<usize>>,
}

#[derive(Debug)]
pub struct Function<Rt: RuntimeType> {
    pub f: ast::user_function::Function<Rt>,
    pub typ: FunctionType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Mutable,
    Immutable,
}

// impl<Rt: RuntimeType + std::fmt::Debug> std::fmt::Debug for Function<Rt> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let mut st = f.debug_struct("Function");
//         st.field("name", &self.f.name).field("typ", &self.typ);

//         match &self.f.value {
//             Value::Fn(..) => st.field("value", &"Fn"),
//             Value::Mut(..) => st.field("value", &"Mut"),
//             Value::Once(..) => st.field("value", &"Once"),
//         };
//         st.finish()
//     }
// }

pub type Id = UserFunctionId;
pub type Table<Rt> = Heap<Id, Function<Rt>>;
