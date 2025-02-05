use std::collections::BTreeMap;
use zkpoly_common::digraph::internal::Predecessors;

use super::super::type3;
use super::*;

fn mut_correction<'s, Rt: RuntimeType>(
    cg: type3::Cg<'s, Rt>,
    uf_table: &user_function::Table<Rt>,
) -> type3::Cg<'s, Rt> {
    let successors = cg.g.successors();

    for vid in cg.g.vertices() {
        let corrected_mutable_uses = cg.g.vertex(vid)
                .mutable_uses(uf_table)
                .collect::<Vec<_>>()
                .into_iter()
                .map(|mutable_succ| {
                    if successors[mutable_succ].len() > 1
                    {
                        let device = cg.g.vertex(mutable_succ).device();
                        let replication = cg.g.add_vertex(type3::Vertex {
                            node: type3::VertexNode::Replicate(mutable_succ),
                            device,
                            typ: cg.g.vertex(mutable_succ).typ().clone(),
                            src: cg.g.vertex(mutable_succ).src().clone(),
                        });
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


pub fn lower<'s, Rt: RuntimeType>(cg: Cg<'s, Rt>) -> type3::Cg<'s, Rt> {
    let mut cg3 = type3::Cg::new();
    let mut gpu2to3 = BTreeMap::<VertexId, type3::VertexId>::new();
    let mut cpu2to3 = BTreeMap::<VertexId, type3::VertexId>::new();

    for (vid, v) in cg.g.topology_sort() {
        // Determine device based on predecessors and successors
        let device = match cg.g.vertex(vid).device() {
            Device::PreferGpu => {
                if cg
                    .g
                    .vertex(vid)
                    .predecessors()
                    .any(|&succ| gpu2to3.contains_key(&succ))
                    || successors[vid]
                        .iter()
                        .any(|succ| cg.g.vertex(*succ).device() == Device::Gpu)
                {
                    type3::Device::Gpu
                } else {
                    type3::Device::Cpu
                }
            }
            Device::Cpu => type3::Device::Cpu,
            Device::Gpu => type3::Device::Gpu,
        };

        // Make sure all predecessors are on desired device
        match device {
            type3::Device::Cpu => {
                for pred in v.predecessors() {
                    if !cpu2to3.contains_key(&pred) {
                        let pred3_gpu = gpu2to3[&pred];
                        let pred3_cpu_v = type3::Vertex {
                            node: type3::VertexNode::D2D {
                                input: pred3_gpu,
                                from: type3::Device::GPU,
                                to: type3::Device::CPU,
                            },
                            device: type3::Device::CPU,
                            typ: cg.g.vertex(pred).typ.clone(),
                            src: cg.g.vertex(pred).src.clone(),
                        };
                        let pred3_cpu = cg3.g.add_vertex(pred3_cpu_v);
                        cpu2to3.insert(pred, pred3_cpu);
                    }
                }
            }
            type3::Device::Gpu => {
                for pred in v.predecessors() {
                    if !gpu2to3.contains_key(&pred) {
                        let pred3_cpu = cpu2to3[&pred];
                        let pred3_gpu_v = type3::Vertex {
                            node: type3::VertexNode::D2D {
                                input: pred3_cpu,
                                from: type3::Device::CPU,
                                to: type3::Device::GPU,
                            },
                            device: type3::Device::GPU,
                            typ: cg.g.vertex(pred).typ.clone(),
                            src: cg.g.vertex(pred).src.clone(),
                        };
                        let pred3_gpu = cg3.g.add_vertex(pred3_gpu_v);
                        gpu2to3.insert(pred, pred3_gpu);
                    }
                }
            }
        }

        // Convert vertex to type3
        let id_mapping = match device {
            type3::Device::Cpu => |i: VertexId| cpu2to3[&i],
            type3::Device::Gpu => |i: VertexId| gpu2to3[&i],
        };

        let v3 = type3::Vertex {
            node: type3::VertexNode::Type2(v.node.relabeled(id_mapping)),
            typ: v.typ().clone(),
            device,
            src: v.src.clone(),
        };

        let v3_id = cg3.g.add_vertex(v3);
        match device {
            type3::Device::Cpu => cpu2to3.insert(vid, v3_id),
            type3::Device::Gpu => gpu2to3.insert(vid, v3_id),
        }
    }

    cg3 = mut_correction(cg3, &cg.uf_table);

    cg3
}
