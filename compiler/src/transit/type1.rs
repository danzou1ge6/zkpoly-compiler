//! Data structures for Stage 1 Transit IR.
//! This type of IR has not have automatically inserted NTT's yet,
//! and vertices of the computation graph still contain expression trees.

use crate::transit::{self, type2, SourceInfo};
use zkpoly_common::digraph::{self, internal::Predecessors};
use zkpoly_common::heap;

#[derive(Debug, Clone)]
pub enum Typ {
    /// Factor ring F_q[x]/<x^n>
    /// Representation (coefficients/Lagrange) is unspecified.
    PolyFr(u64),
    /// Polynomial ring F_q[x] with degree bound.
    /// They will be represented with F_q[x]/<x^n'> for some big enough n'.
    Poly(u64),
    Scalar,
}

zkpoly_common::define_usize_id!(ExprId);

pub type Arith = transit::Arith<ExprId>;

#[derive(Debug, Clone)]
pub enum VertexNode {
    Arith(Arith),
    Input,
    Return,
    /// A small scalar. To accomodate big scalars, use global constants.
    LiteralScalar(usize),
    /// Returns the `n`-degree polynomial obtained from Lagrange Interplotation,
    /// where `n` is length of both [`xs`] and [`ys`]
    Interplote {
        xs: Vec<ExprId>,
        ys: Vec<ExprId>,
    },
    /// Assmeble a `.1`-degree polynomial from coefficient scalars in `tree::.2`
    AssmblePoly(u64, Vec<ExprId>),
}

pub type Vertex<'s> = transit::Vertex<VertexNode, Typ, SourceInfo<'s>>;

impl<'s> digraph::internal::Predecessors<ExprId> for Vertex<'s> {
    #[allow(refining_impl_trait)]
    fn predecessors<'a>(&'a self) -> Box<dyn Iterator<Item = ExprId> + 'a> {
        use VertexNode::*;
        match self.node() {
            Arith(transit::Arith::Bin(_, lhs, rhs)) => Box::new([*lhs, *rhs].into_iter()),
            Arith(transit::Arith::Unr(_, x)) => Box::new([*x].into_iter()),
            Interplote { xs, ys } => Box::new(xs.iter().copied().chain(ys.iter().copied())),
            AssmblePoly(_, es) => Box::new(es.iter().copied()),
            _ => Box::new([].into_iter()),
        }
    }
}

/// Invariants
/// - Acyclic
/// - Types are correct
pub type Cg<'s> = transit::Cg<ExprId, Vertex<'s>>;

// impl<'s> Cg<'s> {
//     // The biggest degree of polynomial n'.
//     pub fn max_poly_degree(&self) -> u64 {
//         let iter = self
//             .outputs
//             .iter()
//             .copied()
//             .fold(self.g.dfs(), |it, begin| it.add_begin(begin));

//         iter.filter_map(|(_, v)| {
//             let transit::Vertex(_, typ, _) = v;
//             match typ {
//                 Typ::Poly(bound) => Some(*bound),
//                 _ => None,
//             }
//         })
//         .max()
//         .unwrap_or(0)
//     }

//     pub fn to_tree2(self) -> type2::Cg<'s> {
//         let fr_degree_bound = self.max_poly_degree();
//         let vprs: Heap<type2::ExprId, _> =
//             to_type2::infer_poly_representations(&self, fr_degree_bound).map(&mut |_, x| x);

//         // Clone tree1 graph
//         let mut g2 = self
//             .g
//             .map(&mut |i, v| to_type2::lower_vertex(v, &vprs[type2::ExprId::from_type1(i)]));

//         let mut repr_stored_in: BTreeMap<(type2::ExprId, type2::PolyRepr), type2::ExprId> =
//             BTreeMap::new();

//         // Add Ntt vertex when needed, and connect them to the vertices they use
//         let old_vertices: Vec<type2::ExprId> = g2.vertices().collect();
//         for i in old_vertices.iter().copied() {
//             let vpr = &vprs[i];
//             for requested_repr in vpr.requested.iter() {
//                 // If some representation is requested, then there must be some representation
//                 // that can be provided
//                 let provided_repr = vpr.provides.as_ref().unwrap();

//                 if requested_repr == provided_repr {
//                     repr_stored_in.insert((i, provided_repr.clone()), i);
//                 } else {
//                     let v_node = type2::VertexNode::Ntt {
//                         s: i,
//                         to: requested_repr.clone(),
//                         from: provided_repr.clone(),
//                     };
//                     let vertex = type2::Vertex::new(
//                         v_node,
//                         type2::Typ::Poly(requested_repr.clone()),
//                         g2.vertex(i).src().clone(),
//                     );
//                     let new_i = g2.add_vertex(vertex);

//                     repr_stored_in.insert((i, requested_repr.clone()), new_i);
//                 }
//             }
//         }

//         // Modify uses of each vertex to locals having correct representation
//         for (i, need) in old_vertices
//             .iter()
//             .copied()
//             .filter_map(|i| vprs[i].needs.as_ref().map(|need| (i, need)))
//         {
//             let predecessors = g2.vertex(i).predecessors().collect::<Vec<_>>();

//             let mut local_subs = BTreeMap::new();

//             for j in predecessors.into_iter() {
//                 if let Some(j_new) = repr_stored_in.get(&(j, need.clone())) {
//                     // If the needed representation of j is registered, v_j should define a
//                     // polynomial
//                     assert!(matches!(g2.vertex(j).typ(), type2::Typ::Poly(..)));
//                     if *j_new != j {
//                         local_subs.insert(j, *j_new);
//                     }
//                 } else {
//                     assert!(!matches!(g2.vertex(j).typ(), type2::Typ::Poly(..)));
//                 }
//             }

//             g2.vertex_mut(i).modify_locals(&mut |local_id| {
//                 local_subs.get(&local_id).copied().unwrap_or(local_id)
//             })
//         }

//         type2::Cg {
//             inputs: self
//                 .inputs
//                 .into_iter()
//                 .map(type2::ExprId::from_type1)
//                 .collect(),
//             outputs: self
//                 .outputs
//                 .into_iter()
//                 .map(type2::ExprId::from_type1)
//                 .collect(),
//             g: g2,
//         }
//     }
// }

// mod to_type2 {
//     use std::collections::{BTreeMap, BTreeSet};

//     use digraph::internal::Predecessors;

//     use super::*;

//     #[derive(Clone, Default)]
//     pub struct VertexPolyReprsentation {
//         /// The needed representation of uses of the vertex
//         pub needs: Option<type2::PolyRepr>,
//         /// The representation the vertex can provide
//         pub provides: Option<type2::PolyRepr>,
//         /// Representations of the vertex's def requested by sucessors of the vertex
//         pub requested: BTreeSet<type2::PolyRepr>,
//     }

//     fn decide_need_and_provides<'s, 'a>(
//         expr_id: ExprId,
//         vertex: &Vertex<'s>,
//         local_typs: &impl Fn(ExprId) -> &'a Typ,
//         vpr: &mut VertexPolyReprsentation,
//         fr_deg_bound: u64,
//     ) {
//         use Typ::*;
//         use VertexNode::*;
//         match &vertex.0 {
//             Arith(..) => match local_typs(expr_id) {
//                 Poly(deg) => {
//                     vpr.needs = Some(type2::PolyRepr::Lagrange(*deg));
//                     vpr.provides = Some(type2::PolyRepr::Lagrange(*deg));
//                 }
//                 PolyFr(_) => {
//                     vpr.needs = Some(type2::PolyRepr::ExtendedLagrange(fr_deg_bound));
//                     vpr.provides = Some(type2::PolyRepr::ExtendedLagrange(fr_deg_bound))
//                 }
//                 _ => {}
//             },
//             Entry => match local_typs(expr_id) {
//                 Poly(deg) => {
//                     vpr.needs = None;
//                     vpr.provides = Some(type2::PolyRepr::Lagrange(*deg));
//                 }
//                 PolyFr(_) => {
//                     vpr.needs = None;
//                     vpr.provides = Some(type2::PolyRepr::ExtendedLagrange(fr_deg_bound));
//                 }
//                 _ => {}
//             },
//             Interplote { xs, .. } => {
//                 vpr.needs = None;
//                 vpr.provides = Some(type2::PolyRepr::Coef(xs.len() as u64));
//             }
//             AssmblePoly(deg, ..) => {
//                 vpr.needs = None;
//                 vpr.provides = Some(type2::PolyRepr::Coef(*deg));
//             }
//             _ => {}
//         }
//     }

//     pub fn infer_poly_representations<'s>(
//         cg: &Cg<'s>,
//         fr_deg_bound: u64,
//     ) -> Heap<ExprId, VertexPolyReprsentation> {
//         let mut vprs = Heap::repeat(VertexPolyReprsentation::default(), cg.g.order());

//         for (i, v) in cg.g.topology_sort() {
//             decide_need_and_provides(
//                 i,
//                 v,
//                 &|local_id| &cg.g.vertex(local_id).typ(),
//                 &mut vprs[i],
//                 fr_deg_bound,
//             );
//             for pred in v.predecessors() {
//                 if let Some(need) = vprs[i].needs.clone() {
//                     vprs[pred].requested.insert(need);
//                 }
//             }
//         }

//         vprs
//     }

//     pub fn lower_typ(typ: &Typ, poly_repr: &Option<type2::PolyRepr>) -> type2::Typ {
//         use Typ::*;
//         match (typ, poly_repr) {
//             (Poly(..), Some(repr)) => type2::Typ::Poly(repr.clone()),
//             (PolyFr(..), Some(type2::PolyRepr::ExtendedLagrange(deg))) => {
//                 type2::Typ::Poly(type2::PolyRepr::ExtendedLagrange(*deg))
//             }
//             (Scalar, _) => type2::Typ::Scalar,
//             _ => panic!("tree1 type {typ:?} cannot be lowered to representation {poly_repr:?}",),
//         }
//     }

//     pub fn lower_vertex<'s>(v: Vertex<'s>, vpr: &VertexPolyReprsentation) -> type2::Vertex<'s> {
//         use VertexNode::*;
//         let transit::Vertex(v_node, v_typ, v_src) = v;
//         let v_node2 = match v_node {
//             Arith(transit::Arith::Bin(op, lhs, rhs)) => {
//                 type2::VertexNode::Arith(transit::Arith::Bin(
//                     op,
//                     type2::ExprId::from_type1(lhs),
//                     type2::ExprId::from_type1(rhs),
//                 ))
//             }
//             Arith(transit::Arith::Unr(op, x)) => {
//                 type2::VertexNode::Arith(transit::Arith::Unr(op, type2::ExprId::from_type1(x)))
//             }
//             Entry => type2::VertexNode::Entry,
//             Return => type2::VertexNode::Return,
//             Interplote { xs, ys } => type2::VertexNode::Interplote {
//                 xs: xs.into_iter().map(type2::ExprId::from_type1).collect(),
//                 ys: ys.into_iter().map(type2::ExprId::from_type1).collect(),
//             },
//             AssmblePoly(deg, coefs) => type2::VertexNode::AssmblePoly(
//                 deg,
//                 coefs.into_iter().map(type2::ExprId::from_type1).collect(),
//             ),
//         };
//         let v_typ2 = to_type2::lower_typ(&v_typ, &vpr.provides);

//         type2::Vertex::new(v_node2, v_typ2, v_src)
//     }
// }
