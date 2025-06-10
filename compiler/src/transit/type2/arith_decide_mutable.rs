use std::collections::{BTreeMap, BTreeSet};

use crate::transit::type2::{Cg, VertexNode};
use zkpoly_common::arith::{FusedType, Mutability};
use zkpoly_runtime::args::RuntimeType;

use super::{
    object_analysis::{self, cg_def_use},
    VertexId,
};

pub fn decide_mutable<'s, Rt: RuntimeType>(
    mut cg: Cg<'s, Rt>,
    seq: &[VertexId],
    obj_def_use: &object_analysis::cg_def_use::DefUse,
    execution_devices: impl Fn(VertexId) -> super::Device,
) -> Cg<'s, Rt> {
    for &vid in seq {
        if let VertexNode::Arith { arith, .. } = cg.g.vertex_mut(vid).node_mut() {
            let inputs_values = arith
                .inputs
                .iter()
                .map(|&eid| {
                    let vid = *arith.g.vertex(eid).op.unwrap_input_outerid();
                    let v = obj_def_use.value(vid).unwrap_single().clone();
                    (eid, v)
                })
                .collect::<BTreeMap<_, _>>();

            let device = cg_def_use::decide_device(execution_devices(vid));

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
                            && obj_def_use
                                .dies(inputs_values[&input_eid].object_id(), device)
                                .is_after(vid)
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
