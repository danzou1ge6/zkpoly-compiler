use std::collections::BTreeMap;
use zkpoly_common::{define_usize_id, digraph::internal::Digraph, heap::Heap, typ::PolyType};
use zkpoly_runtime::{args::{ConstantId, RuntimeType, Variable}, functions::FunctionValue};

use super::{
    transit::type2::{self, partial_typed, VertexId},
    Function, FunctionUntyped,
};

pub type Typ<Rt: RuntimeType> = type2::typ::template::Typ<Rt, (PolyType, Option<u64>)>;

impl<Rt: RuntimeType> Typ<Rt> {
    pub fn lagrange_with_deg(deg: u64) -> Self {
        Self::Poly((PolyType::Lagrange, Some(deg)))
    }
    pub fn lagrange() -> Self {
        Self::Poly((PolyType::Lagrange, None))
    }
    pub fn coef_with_deg(deg: u64) -> Self {
        Self::Poly((PolyType::Coef, Some(deg)))
    }
    pub fn coef() -> Self {
        Self::Poly((PolyType::Coef, None))
    }
}

pub type Vertex<'s, Rt: RuntimeType> = partial_typed::Vertex<'s, Option<Typ<Rt>>>;

pub struct Constant<Rt: RuntimeType> {
    pub(crate) name: Option<String>,
    pub(crate) value: Variable<Rt>,
}

pub type ConstantTable<Rt: RuntimeType> = Heap<ConstantId, Constant<Rt>>;

define_usize_id!(UserFunctionId);

pub type UserFunctionTable<Rt: RuntimeType> = Heap<UserFunctionId, Function<Rt>>;

pub struct Cg<'s, Rt: RuntimeType> {
    pub(crate) g: Digraph<VertexId, Vertex<'s, Rt>>,
    pub(crate) mapping: BTreeMap<*const u8, VertexId>,
    pub(crate) constant_table: ConstantTable<Rt>,
    pub(crate) user_function_table: UserFunctionTable<Rt>,
    pub(crate) user_function_id_mapping: BTreeMap<*const u8, UserFunctionId>,
}

impl<'s, Rt: RuntimeType> Cg<'s, Rt> {
    pub fn lookup_or_insert_with(
        &mut self,
        ptr: *const u8,
        f: impl FnOnce(&mut Self) -> Vertex<'s, Rt>,
    ) -> VertexId {
        if let Some(id) = self.mapping.get(&ptr) {
            return id.clone();
        }
        let v = f(self);
        let id = self.g.add_vertex(v);
        self.mapping.insert(ptr, id.clone());
        id
    }

    pub fn add_constant(&mut self, value: Variable<Rt>, name: Option<String>) -> ConstantId {
        let id = self.constant_table.push(Constant { name, value });
        id
    }

    pub fn add_function(&mut self, f: FunctionUntyped<Rt>) -> UserFunctionId {
        let ptr = f.as_ptr();
        self.user_function_id_mapping
            .entry(ptr)
            .or_insert_with(|| {
                self.user_function_table
                    .push(f.inner.t.take())
            })
            .clone()
    }

}

