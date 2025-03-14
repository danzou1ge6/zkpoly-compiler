use std::collections::BTreeMap;

use zkpoly_common::typ::{PolyMeta, PolyType, Slice};
use zkpoly_runtime::args::RuntimeType;

use crate::ast::PolyInit;

use super::{Chunk, Instruction, InstructionNode, VertexNode};

pub fn rewrite<'s, Rt: RuntimeType>(chunk: Chunk<'s, Rt>) -> Chunk<'s, Rt> {
    chunk.with_reg_id_allocator_taken(|mut chunk, mut ra| {
        // Rewrite Type2 NewPoly to Type3 FillPoly
        let mut malloc_mapping = BTreeMap::new();

        for (_i, inst) in chunk.iter_instructions() {
            match &inst.node {
                InstructionNode::Type2 {
                    ids,
                    vertex: VertexNode::NewPoly(deg, _, pty),
                    ..
                } => {
                    assert!(ids.len() == 1);
                    malloc_mapping.insert(ids[0].0, ra.inherit(ids[0].0));
                }
                _ => {}
            }
        }

        chunk.rewrite_instructions(|insts| {
            insts.flat_map(|(_i, inst)| match inst {
                Instruction {
                    node:
                        InstructionNode::Type2 {
                            ids,
                            temp,
                            vertex: VertexNode::NewPoly(deg, init, pty),
                            ..
                        },
                    src,
                } => {
                    assert!(temp.is_empty());
                    std::iter::once(Instruction {
                        node: InstructionNode::FillPoly {
                            id: ids[0].0,
                            operand: malloc_mapping[&ids[0].0],
                            deg: deg as usize,
                            init,
                            pty,
                        },
                        src,
                    })
                }
                mut otherwise => {
                    if otherwise.node.is_allloc() {
                        otherwise.node.ids_mut().for_each(|d| {
                            if let Some(id) = malloc_mapping.get(d) {
                                *d = *id
                            }
                        });
                    }
                    std::iter::once(otherwise)
                }
            })
        });

        // Rewrite Type2 Extend to Type3 FillPoly and Transfer
        let mut malloc_mapping = BTreeMap::new();
        let mut degrees = BTreeMap::new();

        for (_i, inst) in chunk.iter_instructions() {
            match &inst.node {
                InstructionNode::Type2 {
                    ids,
                    vertex: VertexNode::Extend(operand, ..),
                    ..
                } => {
                    assert!(ids.len() == 1);
                    malloc_mapping.insert(ids[0].0, ra.inherit(ids[0].0));
                    let (deg, _) = ra.typ_of(*operand).unwrap_poly();
                    degrees.insert(*operand, deg as usize);
                }
                _ => {}
            }
        }

        chunk.rewrite_instructions(|insts| {
            insts.flat_map(|(_i, inst)| match inst {
                Instruction {
                    node:
                        InstructionNode::Type2 {
                            ids,
                            temp,
                            vertex: VertexNode::Extend(operand, deg),
                            ..
                        },
                    src,
                } => {
                    assert!(temp.is_empty());
                    let r0 = operand;
                    let deg0 = degrees[&r0];
                    let deg = deg as usize;
                    let r1 = ids[0].0;
                    let t0 = malloc_mapping[&r1];
                    let t1 = ra.inherit(t0);
                    let device = ra.device_of(t1);
                    let t2 = ra.alloc(
                        super::typ::Typ::ScalarArray {
                            len: deg0,
                            meta: PolyMeta::Sliced(Slice::new(0, deg0 as u64)),
                        },
                        device,
                    );
                    let t3 = ra.inherit(t0);
                    let inst = |node| Instruction {
                        node,
                        src: src.clone(),
                    };
                    vec![
                        inst(InstructionNode::FillPoly {
                            id: t1,
                            operand: t0,
                            deg,
                            init: PolyInit::Zeros,
                            pty: PolyType::Coef,
                        }),
                        inst(InstructionNode::SetPolyMeta {
                            id: t2,
                            from: t1,
                            offset: 0,
                            len: deg0,
                        }),
                        inst(InstructionNode::TransferToDefed {
                            id: t3,
                            to: t2,
                            from: r0,
                        }),
                        inst(InstructionNode::SetPolyMeta {
                            id: r1,
                            from: t3,
                            offset: 0,
                            len: deg,
                        }),
                    ]
                    .into_iter()
                    .chain(
                        [t1, t3]
                            .into_iter()
                            .map(|t| inst(InstructionNode::StackFree { id: t })),
                    )
                    .collect::<Vec<_>>()
                    .into_iter()
                }
                mut otherwise => {
                    if otherwise.node.is_allloc() {
                        otherwise.node.ids_mut().for_each(|d| {
                            if let Some(id) = malloc_mapping.get(d) {
                                *d = *id
                            }
                        });
                    }
                    vec![otherwise].into_iter()
                }
            })
        });

        (chunk, ra)
    })
}
