use std::collections::BTreeMap;

use zkpoly_common::{define_usize_id, digraph::internal::Digraph};
use zkpoly_runtime::args::{ConstantId, RuntimeType};

use crate::transit::{
    type2::{VertexId, VertexNode},
    Vertex,
};

use super::{template, Cg};

define_usize_id!(EquivalenceClassId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct EquivalenceClass {
    pub id: EquivalenceClassId,
    pub rotate: i32,
}

impl EquivalenceClass {
    pub fn rotate(&mut self, offset: i32) {
        self.rotate += offset;
    }
}

pub type EquivalenceNode = template::VertexNode<
    EquivalenceClass,
    zkpoly_common::arith::ArithGraph<EquivalenceClass, zkpoly_common::arith::ExprId>,
    ConstantId,
    zkpoly_runtime::functions::UserFunctionId,
>;

fn get_equivalence_class(
    node2class: &BTreeMap<EquivalenceNode, EquivalenceClass>,
    node: &EquivalenceNode,
    id_allocator: &mut usize,
) -> EquivalenceClass {
    match node {
        EquivalenceNode::RotateIdx(father, offset) => {
            let mut class = father.clone();
            class.rotate(*offset);
            class
        }
        _ => node2class.get(&node).cloned().unwrap_or_else(|| {
            let id = *id_allocator;
            *id_allocator += 1;
            EquivalenceClass {
                id: id.into(),
                rotate: 0,
            }
        }),
    }
}

pub fn cse<'s, Rt: RuntimeType>(cg: Cg<'s, Rt>) -> Cg<'s, Rt> {
    let mut vid2class: BTreeMap<VertexId, EquivalenceClass> = BTreeMap::new();
    let mut node2class: BTreeMap<EquivalenceNode, EquivalenceClass> = BTreeMap::new();
    let mut class2new_id: BTreeMap<EquivalenceClass, VertexId> = BTreeMap::new();
    let mut new_graph = Digraph::new();
    let mut id_allocator = 0;

    for (vid, v) in cg.g.topology_sort() {
        let Vertex(node, typ, src) = v.clone();

        let eq_node: EquivalenceNode =
            node.relabeled(&mut |vid| vid2class.get(&vid).cloned().unwrap());

        let class = get_equivalence_class(&node2class, &eq_node, &mut id_allocator);
        vid2class.insert(vid, class.clone());

        // insert new node
        if !class2new_id.contains_key(&class) {
            let new_node: VertexNode = node
                .relabeled(&mut |vid| class2new_id.get(vid2class.get(&vid).unwrap()).cloned().unwrap());
            let new_vid = new_graph.add_vertex(Vertex::new(new_node, typ, src));
            class2new_id.insert(class.clone(), new_vid);
        } else {
            // update src info
            let new_vid = class2new_id.get(&class).unwrap().clone();
            new_graph.vertex_mut(new_vid).src_mut().location.extend(src.location);
            // check type
            assert_eq!(
                *new_graph.vertex(new_vid).typ(),
                typ,
                "Type mismatch in CSE: {:?} vs {:?}",
                new_graph.vertex(new_vid).typ(),
                typ
            );
        }

        node2class.insert(eq_node, class.clone());
    }
    Cg {
        output: class2new_id.get(vid2class.get(&cg.output).unwrap()).cloned().unwrap(),
        g: new_graph,
    }
}
