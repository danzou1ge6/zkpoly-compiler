use super::{user_function, Cg, Vertex, VertexId, VertexNode};
use std::collections::BTreeMap;
use zkpoly_runtime::args::RuntimeType;

pub fn correct<'s, Rt: RuntimeType + Clone>(
    mut cg: Cg<'s, Rt>,
    seq: &Vec<VertexId>,
    uf_table: &user_function::Table<Rt>,
) -> Cg<'s, Rt> {
    let successors = cg.g.successors();
    let seq_num = seq
        .iter()
        .enumerate()
        .fold(BTreeMap::new(), |mut acc, (seq, &vid)| {
            acc.insert(vid, seq);
            acc
        });

    for vid in cg.g.vertices() {
        let corrected_mutable_uses =
            cg.g.vertex(vid)
                .mutable_uses(uf_table)
                .collect::<Vec<_>>()
                .into_iter()
                .map(|mutable_succ| {
                    if successors[mutable_succ]
                        .iter()
                        .map(|v| seq_num[v])
                        .max()
                        .unwrap()
                        < seq_num[&vid]
                    {
                        let replication = cg.g.add_vertex(Vertex::new(
                            VertexNode::Replicate(mutable_succ),
                            cg.g.vertex(mutable_succ).typ().clone(),
                            cg.g.vertex(mutable_succ).src().clone(),
                        ));
                        replication
                    } else {
                        mutable_succ
                    }
                })
                .collect::<Vec<_>>();
        cg.g.vertex_mut(vid)
            .mutable_uses_mut(uf_table)
            .zip(corrected_mutable_uses.into_iter())
            .for_each(|(s, v)| *s = v);
    }

    cg
}
