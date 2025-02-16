use std::collections::BTreeMap;

use zkpoly_common::digraph::internal::Digraph;
use zkpoly_runtime::args::RuntimeType;

use super::transit::type2::{partial_typed::Vertex, VertexId};

pub struct Cg<'s, Rt: RuntimeType> {
    pub(crate) g: Digraph<VertexId, Vertex<'s, Rt>>,
    pub(crate) mapping: BTreeMap<*const u8, VertexId>
}

impl <'s, Rt: RuntimeType> Cg<'s, Rt> {
    pub fn lookup_or_insert_with(&mut self, ptr: *const u8, f: impl FnOnce(&mut Self) -> Vertex<'s, Rt>) -> VertexId {
        if let Some(id) = self.mapping.get(&ptr) {
            return id.clone();
        }
        let v = f(self);
        let id = self.g.add_vertex(v);
        self.mapping.insert(ptr, id.clone());
        id
    }
}
