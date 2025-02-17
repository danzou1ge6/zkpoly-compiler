use std::collections::BTreeMap;
use zkpoly_common::{digraph::internal::Digraph, heap::Heap, typ::PolyType};
use zkpoly_runtime::args::{ConstantId, RuntimeType, Variable};

use super::transit::type2::{self, partial_typed, VertexId};

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
    name: Option<String>,
    value: Variable<Rt>,
}

pub type ConstantTable<Rt: RuntimeType> = Heap<ConstantId, Constant<Rt>>;

pub struct Cg<'s, Rt: RuntimeType> {
    pub(crate) g: Digraph<VertexId, Vertex<'s, Rt>>,
    pub(crate) mapping: BTreeMap<*const u8, VertexId>,
    pub(crate) constant_table: ConstantTable<Rt>,
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
}
