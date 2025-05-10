use std::collections::{BTreeMap, BTreeSet};

use crate::transit::{
    type2::{object_analysis::AtModifier, Cg, VertexNode},
    type3::Device,
};
use zkpoly_common::{
    arith::{ArithGraph, FusedType, Mutability, Operation},
    heap::UsizeId,
};
use zkpoly_runtime::args::RuntimeType;

use super::object_analysis::{ObjectUse, ObjectsDieAfter, VertexValue};

pub fn decide_mutable<'s, Rt: RuntimeType>(
    mut cg: Cg<'s, Rt>,
    vertex_inputs: &ObjectUse,
    obj_die_after: &ObjectsDieAfter,
) -> Cg<'s, Rt> {
    return cg;
    for vid in cg.g.vertices().collect::<Vec<_>>().into_iter() {
        if let VertexNode::Arith { arith, chunking } = cg.g.vertex_mut(vid).node_mut() {
            let device = if chunking.is_some() {
                Device::Cpu
            } else {
                Device::Gpu
            };

            let inputs_values = arith
                .inputs
                .iter()
                .zip(vertex_inputs.input_of(vid))
                .map(|(input_eid, vv)| (*input_eid, vv.unwrap_single().clone()))
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
                let mut inplace_candidates = arith.inputs.iter().filter(|&&input_eid| {
                    let input = arith.g.vertex(input_eid);
                        !mutated_inputs.contains(&input_eid)
                        && input.op.unwrap_input_typ() == FusedType::ScalarArray
                        && obj_die_after
                            .get_device(device)
                            .get(&inputs_values[&input_eid].object_id())
                            .is_some_and(|(after_vid, atm)| {
                                *after_vid == vid && *atm == AtModifier::After
                            })
                });

                if let Some(inplace_eid) = inplace_candidates.next() {
                    // If we found inputs that can be used inplace, we then
                    // - Mark one of them as mutable
                    // - Add write-after-read constraints to the output vertex
                    let same_obj_eids = arith
                        .inputs
                        .iter()
                        .filter(|&&input_eid| {
                            inputs_values[&input_eid].object_id()
                                == inputs_values[&inplace_eid].object_id()
                        })
                        .copied()
                        .collect::<Vec<_>>();

                    {
                        let (_, _, mutability) =
                            arith.g.vertex_mut(same_obj_eids[0]).op.unwrap_input_mut();
                        *mutability = Mutability::Mut;
                    }

                    for &eid in same_obj_eids.iter() {
                        mutated_inputs.insert(eid);
                    }

                    let (_, _, _, in_nodes) =
                        arith.g.vertex_mut(*output_eid).op.unwrap_output_mut();

                    for eid in same_obj_eids {
                        in_nodes.push(eid);
                    }
                }
            }
        }
    }
    cg
}
