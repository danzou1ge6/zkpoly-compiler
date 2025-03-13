use std::collections::BTreeMap;
use zkpoly_common::{
    arith::{Arith, BinOp, UnrOp},
    define_usize_id,
    digraph::internal::Digraph,
    heap::Heap,
    typ::PolyType,
};
use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_runtime as rt;
pub use zkpoly_runtime::args::ConstantId;
use zkpoly_runtime::args::{RuntimeType, Variable};

use super::{
    transit::type2::{self, partial_typed, VertexId},
    transit::SourceInfo,
    user_function::Function,
    FunctionUntyped, RuntimeCorrespondance,
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

    pub fn try_to_type2(self) -> Option<type2::Typ<Rt>> {
        use type2::typ::template::Typ::*;
        match self {
            Poly((ptyp, deg)) => Some(Poly((ptyp, deg?))),
            PointBase { log_n } => Some(PointBase { log_n }),
            Scalar => Some(Scalar),
            Transcript => Some(Transcript),
            Point => Some(Point),
            Tuple(elements) => Some(Tuple(
                elements
                    .into_iter()
                    .map(|x| x.try_to_type2())
                    .collect::<Option<Vec<_>>>()?,
            )),
            Array(t, len) => Some(Array(Box::new(t.try_to_type2()?), len)),
            Any(tid, size) => Some(Any(tid, size)),
            _Phantom(..) => unreachable!(),
        }
    }

    pub fn from_type2(t2typ: type2::Typ<Rt>) -> Self {
        use type2::typ::template::Typ::*;
        match t2typ {
            Poly((ptyp, deg)) => Self::Poly((ptyp, Some(deg))),
            PointBase { log_n } => Self::PointBase { log_n },
            Scalar => Self::Scalar,
            Transcript => Self::Transcript,
            Point => Self::Point,
            Tuple(elements) => Self::Tuple(
                elements
                    .into_iter()
                    .map(|x| Self::from_type2(x))
                    .collect::<Vec<_>>(),
            ),
            Array(t, len) => Self::Array(Box::new(Self::from_type2(*t)), len),
            Any(tid, size) => Self::Any(tid, size),
            _Phantom(..) => unreachable!(),
        }
    }

    pub fn compatible_with_type2(&self, t2typ: &type2::Typ<Rt>) -> bool {
        use type2::typ::template::Typ::*;
        match (self, t2typ) {
            (Poly((ptyp, deg1)), Poly((ptyp2, deg2))) => {
                ptyp == ptyp2 && deg1.is_none() || deg1.unwrap() == *deg2
            }
            (PointBase { log_n: log_n1 }, PointBase { log_n: log_n2 }) => log_n1 == log_n2,
            (Scalar, Scalar) => true,
            (Transcript, Transcript) => true,
            (Point, Point) => true,
            (Tuple(elements1), Tuple(elements2)) => {
                elements1.len() == elements2.len()
                    && elements1
                        .iter()
                        .zip(elements2.iter())
                        .all(|(x, y)| x.compatible_with_type2(y))
            }
            (Array(t1, len1), Array(t2, len2)) => t1.compatible_with_type2(t2) && len1 == len2,
            (Any(tid1, size1), Any(tid2, size2)) => tid1 == tid2 && size1 == size2,
            _ => false,
        }
    }
}

pub type Vertex<'s, Rt: RuntimeType> = partial_typed::Vertex<'s, Option<Typ<Rt>>>;

impl<'s, Rt: RuntimeType> Vertex<'s, Rt> {
    pub fn try_to_type2_typ(&self) -> Option<type2::Typ<Rt>> {
        self.typ()
            .as_ref()
            .map(|typ| typ.clone().try_to_type2())
            .flatten()
    }
}

#[derive(Debug, Clone)]
pub struct Constant<Rt: RuntimeType> {
    pub(crate) name: Option<String>,
    pub(crate) value: Variable<Rt>,
}

impl<Rt: RuntimeType> Constant<Rt> {
    pub fn new(value: Variable<Rt>, name: String) -> Self {
        Self {
            name: Some(name),
            value,
        }
    }
}

pub type ConstantTable<Rt: RuntimeType> = Heap<ConstantId, Constant<Rt>>;

define_usize_id!(UserFunctionId);

pub type UserFunctionTable<Rt: RuntimeType> = Heap<UserFunctionId, Function<Rt>>;

mod type_inferer;
pub use type_inferer::{Error, ErrorNode};

pub struct Cg<'s, Rt: RuntimeType> {
    pub(crate) g: Digraph<VertexId, Vertex<'s, Rt>>,
    pub(crate) mapping: BTreeMap<*const u8, VertexId>,
    pub(crate) constant_table: ConstantTable<Rt>,
    pub(crate) user_function_table: UserFunctionTable<Rt>,
    pub(crate) user_function_id_mapping: BTreeMap<*const u8, UserFunctionId>,
    pub(crate) allocator: PinnedMemoryPool,
    pub(crate) one: ConstantId,
    pub(crate) zero: ConstantId,
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
            .or_insert_with(|| self.user_function_table.push(f.inner.t.take()))
            .clone()
    }

    pub fn allocator(&mut self) -> &mut PinnedMemoryPool {
        &mut self.allocator
    }

    pub fn lower(
        self,
        output: VertexId,
    ) -> Result<
        type2::Program<'s, Rt>,
        (
            type_inferer::Error<'s, Rt>,
            Digraph<VertexId, type2::partial_typed::Vertex<'s, Option<type2::Typ<Rt>>>>,
        ),
    > {
        let mut inferer = type_inferer::TypeInferer::new();
        match self.g.map_by_ref_result(|vid, _| inferer.infer(&self, vid)) {
            Ok(types) => {
                let g: Digraph<VertexId, _> = self
                    .g
                    .map(&mut |vid, v| v.map_typ(|_| types.vertex(vid).clone()));

                let uf_table: Heap<UserFunctionId, _> =
                    self.user_function_table.map(&mut |_, f| {
                        let typ = type2::user_function::FunctionType {
                            args: vec![type2::user_function::Mutability::Immutable; f.n_args],
                            ret_inplace: f.ret_typ.iter().map(|_| None).collect(),
                        };
                        type2::user_function::Function { typ, f }
                    });

                Ok(type2::Program {
                    cg: crate::transit::Cg { output, g },
                    user_function_table: uf_table,
                    consant_table: self.constant_table,
                    memory_pool: self.allocator,
                })
            }
            Err(e) => {
                let g: Digraph<VertexId, _> = self
                    .g
                    .map(&mut |vid, v| v.map_typ(|_| inferer.get_typ(vid).cloned()));

                Err((e, g))
            }
        }
    }

    pub fn empty(allocator: PinnedMemoryPool) -> Self {
        let mut constant_table = Heap::new();
        let one = constant_table.push(Constant::new(
            super::Scalar::to_variable(rt::scalar::Scalar::from_ff(
                &<Rt::Field as group::ff::Field>::ONE,
            )),
            "scalar_one".to_string(),
        ));
        let zero = constant_table.push(Constant::new(
            super::Scalar::to_variable(rt::scalar::Scalar::from_ff(
                &<Rt::Field as group::ff::Field>::ZERO,
            )),
            "scalar_zero".to_string(),
        ));
        Self {
            g: Digraph::new(),
            mapping: BTreeMap::new(),
            user_function_table: Heap::new(),
            constant_table,
            user_function_id_mapping: BTreeMap::new(),
            allocator,
            one,
            zero,
        }
    }

    pub fn new(
        output_v: impl super::TypeEraseable<Rt>,
        allocator: PinnedMemoryPool,
    ) -> (Self, VertexId) {
        let mut cg = Self::empty(allocator);
        let output_vid = output_v.erase(&mut cg);
        let src_info = cg.g.vertex(output_vid).src().clone();
        let return_vid = cg.g.add_vertex(Vertex::new(
            type2::VertexNode::Return(output_vid),
            None,
            src_info,
        ));
        (cg, return_vid)
    }
}
