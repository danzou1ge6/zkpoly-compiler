use std::collections::{BTreeMap, BTreeSet};

use crate::transit::{
    type2::{object_analysis::AtModifier, Cg, VertexNode},
    type3::Device,
};
use zkpoly_common::{
    arith::{ArithGraph, FusedType, Mutability, Operation},
    heap::{Heap, UsizeId},
};
use zkpoly_runtime::args::RuntimeType;

use super::{
    object_analysis::{ObjectUse, ObjectsDef, ObjectsDieAfter, VertexValue},
    VertexId,
};

pub fn decide_mutable<'s, Rt: RuntimeType>(
    mut cg: Cg<'s, Rt>,
    obj_def: &ObjectsDef,
    obj_die_after: &ObjectsDieAfter,
) -> Cg<'s, Rt> {
    let connected_component = cg.g.connected_component(cg.output);

    for vid in
        cg.g.vertices()
            .collect::<Vec<_>>()
            .into_iter()
            .filter(|vid| connected_component[*vid])
    {
        if let VertexNode::Arith { arith, chunking } = cg.g.vertex_mut(vid).node_mut() {
            let device = if chunking.is_some() {
                Device::Cpu
            } else {
                Device::Gpu
            };

            let inputs_values = arith
                .inputs
                .iter()
                .map(|&eid| {
                    let vid = *arith.g.vertex(eid).op.unwrap_input_outerid();
                    let v = obj_def.values[&vid].unwrap_single().clone();
                    (eid, v)
                })
                .collect::<BTreeMap<_, _>>();

            let mut mutated_inputs = BTreeSet::new();

            for output_eid in arith.outputs.iter().collect::<Vec<_>>().into_iter() {
                let (_, ft, _, _) = arith.g.vertex(*output_eid).op.unwrap_output();
                if *ft != FusedType::ScalarArray {
                    continue;
                }

                // Find inputs that can be usesd inplace for the output vertex, which satisfies
                // - Input object is not mutable yet
                // - Input has the same FusedType
                // - Input dies after this vertex
                // - Input is the only input that uses its object
                let mut inplace_candidates = arith
                    .inputs
                    .iter()
                    .filter(|&&input_eid| {
                        let input = arith.g.vertex(input_eid);
                        !mutated_inputs.contains(&input_eid)
                            && input.op.unwrap_input_typ() == FusedType::ScalarArray
                            && obj_die_after
                                .get_device(device)
                                .get(&inputs_values[&input_eid].object_id())
                                .is_some_and(|(after_vid, atm)| {
                                    *after_vid == vid && *atm == AtModifier::After
                                })
                    })
                    .filter_map(|&eid| {
                        let same_obj_eids = arith
                            .inputs
                            .iter()
                            .filter(|&&input_eid| {
                                inputs_values[&input_eid].object_id()
                                    == inputs_values[&eid].object_id()
                            })
                            .copied()
                            .collect::<Vec<_>>();

                        if same_obj_eids.len() > 1 {
                            None
                        } else {
                            Some(eid)
                        }
                    });

                if let Some(inplace_eid) = inplace_candidates.next() {
                    // If we found inputs that can be used inplace, we then

                    let (_, _, mutability) = arith.g.vertex_mut(inplace_eid).op.unwrap_input_mut();
                    *mutability = Mutability::Mut;

                    mutated_inputs.insert(inplace_eid);

                    // - Add write-after-read constraints to the output vertex
                    let (_, _, _, in_nodes) =
                        arith.g.vertex_mut(*output_eid).op.unwrap_output_mut();

                    *in_nodes = Some(inplace_eid)
                }
            }
        }
    }
    cg
}
