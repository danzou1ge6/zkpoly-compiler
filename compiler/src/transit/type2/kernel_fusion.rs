use std::{
    collections::{BTreeMap, BTreeSet},
    panic::Location,
};

use crate::transit::{type2, SourceInfo};

use super::{Cg, Vertex, VertexId, VertexNode};
use zkpoly_common::{
    arith::{
        self, Arith, ArithGraph, ArithUnrOp, ArithVertex, ExprId, FusedType, Mutability, Operation,
        UnrOp,
    },
    digraph::internal::Digraph,
    heap::Heap,
    typ::PolyType,
};
use zkpoly_runtime::args::RuntimeType;

impl<'s, Rt: RuntimeType> Cg<'s, Rt> {
    fn can_fuse(
        &self,
        vid: VertexId,
        fused: &Vec<usize>,
        to: &Vec<bool>,
        from: &Vec<bool>,
        size_limit: Option<usize>,
        cur_size: Option<usize>,
    ) -> bool {
        if size_limit.is_some() && cur_size.is_some() {
            if cur_size.unwrap() >= size_limit.unwrap() {
                return false;
            }
        }
        let v = self.g.vertex(vid);
        let vid_usize: usize = vid.into();
        if fused[vid_usize] != 0 || to[vid_usize] || from[vid_usize] {
            return false;
        }
        match v.node() {
            VertexNode::SingleArith(a) => match a {
                Arith::Bin(arith::BinOp::Pp(op), lhs, rhs) => {
                    if matches!(op, arith::ArithBinOp::Add | arith::ArithBinOp::Sub) {
                        let (pty1, deg1) = self.g.vertex(*lhs).typ().unwrap_poly();
                        let (pty2, deg2) = self.g.vertex(*rhs).typ().unwrap_poly();
                        if *deg1 == *deg2 {
                            true
                        } else {
                            // Only when lhs and rhs both are under coefficient representation
                            // can they have different degrees.
                            // In this case, they cannot be fused.
                            assert!(*pty1 == PolyType::Coef && *pty2 == PolyType::Coef);
                            false
                        }
                    } else {
                        true
                    }
                }
                Arith::Unr(UnrOp::S(ArithUnrOp::Pow(_)), _) => false,
                _ => true,
            },
            _ => false,
        }
    }

    fn get_fuse_type(&self, vid: VertexId) -> FusedType {
        let v = self.g.vertex(vid);
        match v.typ() {
            super::typ::template::Typ::Poly(_) => FusedType::ScalarArray,
            super::typ::template::Typ::Scalar => FusedType::Scalar,
            _ => panic!("src nodes of single arith should have fused type output"),
        }
    }

    fn fuse_bwd(
        &self,
        vid: VertexId,
        fused: &mut Vec<usize>,
        to: &mut Vec<bool>,
        from: &mut Vec<bool>,
        vid2arith: &mut BTreeMap<VertexId, ExprId>,
        succ: &Heap<VertexId, BTreeSet<VertexId>>,
        ag: &mut ArithGraph<VertexId, ExprId>,
        output_v: &mut Vec<Vec<(VertexId, VertexId)>>,
        src_info: &mut Vec<Location<'s>>,
        fuse_id: usize,
        size_limit: usize,
    ) {
        if fused[usize::from(vid)] == fuse_id || vid2arith.contains_key(&vid) {
            return; // is in the current fusion
        }
        if self.can_fuse(
            vid,
            fused,
            to,
            from,
            Some(size_limit),
            Some(vid2arith.len()),
        ) {
            self.fuse_it(
                vid, fused, to, from, vid2arith, succ, ag, output_v, src_info, fuse_id, size_limit,
            );
        } else {
            let vid_arith = ag.g.add_vertex(ArithVertex {
                op: {
                    Operation::Input {
                        outer_id: vid,
                        typ: self.get_fuse_type(vid),
                        mutability: Mutability::Const, // this will be changed after the whole subgraph is fused
                    }
                },
            });
            vid2arith.insert(vid, vid_arith);
            ag.inputs.push(vid_arith);
        }
    }

    // use dfs to mark the "to" nodes
    fn mark_fwd(
        &self,
        vid: VertexId,
        to: &mut Vec<bool>,
        succ: &Heap<VertexId, BTreeSet<VertexId>>,
        start_mark: bool,
        fused: &Vec<usize>,
        from: &Vec<bool>,
        fuse_id: usize,
    ) {
        let vid_usize: usize = vid.into();
        if to[vid_usize] {
            return;
        }
        let can_fuse = self.can_fuse(vid, fused, to, from, None, None);
        if start_mark {
            to[vid_usize] = true
        }
        let mut mark_next = start_mark;
        if !can_fuse && fused[vid_usize] != fuse_id {
            mark_next = true;
        }
        for fwd in succ[vid].iter() {
            self.mark_fwd(*fwd, to, succ, mark_next, fused, from, fuse_id);
        }
    }

    // use dfs to mark the "from" nodes
    fn mark_bwd(
        &self,
        vid: VertexId,
        from: &mut Vec<bool>,
        start_mark: bool,
        fused: &Vec<usize>,
        to: &Vec<bool>,
        fuse_id: usize,
    ) {
        let vid_usize: usize = vid.into();
        if from[vid_usize] {
            return;
        }
        let can_fuse = self.can_fuse(vid, fused, to, from, None, None);
        if start_mark {
            from[vid_usize] = true;
        }
        let mut mark_next = start_mark;
        if !can_fuse && fused[vid_usize] != fuse_id {
            mark_next = true;
        }
        for bwd in self.g.vertex(vid).uses() {
            self.mark_bwd(bwd, from, mark_next, fused, to, fuse_id);
        }
    }

    fn fuse_it(
        &self,
        vid: VertexId,
        fused: &mut Vec<usize>,
        to: &mut Vec<bool>,
        from: &mut Vec<bool>,
        vid2arith: &mut BTreeMap<VertexId, ExprId>,
        succ: &Heap<VertexId, BTreeSet<VertexId>>,
        ag: &mut ArithGraph<VertexId, ExprId>,
        output_v: &mut Vec<Vec<(VertexId, VertexId)>>, // output_v[rank in outputs]: vec[(targetid, old_src_id)]
        src_info: &mut Vec<Location<'s>>,
        fuse_id: usize,
        size_limit: usize,
    ) {
        let v = self.g.vertex(vid);
        match v.node() {
            VertexNode::SingleArith(arith) => {
                self.mark_bwd(vid, from, false, fused, to, fuse_id);
                self.mark_fwd(vid, to, succ, false, fused, from, fuse_id);

                let vid_usize: usize = vid.into();
                fused[vid_usize] = fuse_id;
                let my_arith = ag.g.add_vertex(ArithVertex {
                    op: Operation::Todo, // for id gen
                });

                vid2arith.insert(vid, my_arith);
                src_info.push(v.src().unwrap_location_single().clone());
                match arith {
                    Arith::Bin(_, lhs, rhs) => {
                        self.fuse_bwd(
                            *lhs, fused, to, from, vid2arith, succ, ag, output_v, src_info,
                            fuse_id, size_limit,
                        );
                        self.fuse_bwd(
                            *rhs, fused, to, from, vid2arith, succ, ag, output_v, src_info,
                            fuse_id, size_limit,
                        );
                    }
                    Arith::Unr(_, src) => {
                        self.fuse_bwd(
                            *src, fused, to, from, vid2arith, succ, ag, output_v, src_info,
                            fuse_id, size_limit,
                        );
                    }
                };
                *ag.g.vertex_mut(my_arith) = ArithVertex {
                    op: Operation::Arith(
                        arith.relabeled(&mut |vid| vid2arith.get(&vid).unwrap().clone()),
                    ),
                };
                let mut output_rank = None;
                for node in succ[vid].iter() {
                    if fused[usize::from(*node)] == fuse_id {
                        continue; // is in the current fusion
                    }
                    if self.can_fuse(
                        *node,
                        fused,
                        to,
                        from,
                        Some(size_limit),
                        Some(vid2arith.len()),
                    ) {
                        self.fuse_it(
                            *node, fused, to, from, vid2arith, succ, ag, output_v, src_info,
                            fuse_id, size_limit,
                        );
                    } else {
                        if output_rank.is_none() {
                            output_rank = Some(output_v.len());
                            output_v.push(Vec::new());
                            ag.outputs.push(my_arith); // will be converted to output nodes
                        };
                        output_v[output_rank.unwrap()].push((*node, vid));
                    }
                }
            }
            _ => panic!("can only fuse single arith"),
        }
    }
}

pub fn fuse_arith<'s, Rt: RuntimeType>(mut cg: Cg<'s, Rt>) -> Cg<'s, Rt> {
    let mut succ: Heap<VertexId, BTreeSet<VertexId>> = cg.g.successors();
    let mut fused = vec![0; cg.g.order()];
    let order = cg.g.vertices().collect::<Vec<_>>();
    let mut to = vec![false; cg.g.order()]; // vertexs rely on the current fused kernel
    let mut from = vec![false; cg.g.order()]; // vertexs the current fused kernel relies on
    let mut fuse_id = 1;
    for id in order {
        if cg.can_fuse(id, &mut fused, &mut to, &mut from, None, None) {
            let mut vid2arith = BTreeMap::new();
            let mut ag = ArithGraph {
                inputs: Vec::new(),
                outputs: Vec::new(),
                g: Digraph::new(),
                poly_repr: PolyType::Coef, // revised later
            };
            let mut output_v = Vec::new();
            let mut src_info = Vec::new();
            cg.fuse_it(
                id,
                &mut fused,
                &mut to,
                &mut from,
                &mut vid2arith,
                &succ,
                &mut ag,
                &mut output_v,
                &mut src_info,
                fuse_id,
                512,
            );

            let mut output_polys = 0;
            let (output_types, output_outer_info) = output_v
                .iter()
                .map(|output_i /*(targetid, old_src_id)*/| {
                    let old_src = output_i[0].1;
                    let old = cg.g.vertex(old_src);
                    let fused_type = cg.get_fuse_type(old_src);
                    if fused_type == FusedType::ScalarArray {
                        output_polys += 1
                    }
                    (old.typ().clone(), (fused_type, old_src))
                })
                .collect::<(Vec<_>, Vec<_>)>();

            // correct the polynomial representation
            ag.poly_repr = output_types
                .iter()
                .fold(None, |acc, t| {
                    if let type2::Typ::Poly((t_repr, _)) = t {
                        if let Some(acc_repr) = acc {
                            if &acc_repr != t_repr {
                                panic!("outputs have different polynomial representation");
                            }
                        }
                        Some(t_repr.clone())
                    } else {
                        acc
                    }
                })
                .unwrap_or(PolyType::Coef);

            // change input mutability
            let mut mut_inputs = ag.change_mutability(&succ, output_polys).into_iter();

            // add output nodes
            for (out_arith, (fuse_type, outer_id)) in ag.outputs.iter_mut().zip(output_outer_info) {
                let mut in_node = None;
                if fuse_type == FusedType::ScalarArray {
                    in_node = mut_inputs.next();
                }
                *out_arith = ag.g.add_vertex(ArithVertex {
                    op: Operation::Output {
                        outer_id: outer_id,
                        typ: fuse_type,
                        store_node: *out_arith,
                        in_node: in_node,
                    },
                });
            }

            assert_eq!(output_types.len(), ag.outputs.len());
            let typ = super::typ::template::Typ::Tuple(output_types.clone());

            let node_id = cg.g.add_vertex(Vertex::new(
                VertexNode::Arith {
                    arith: ag,
                    chunking: None,
                },
                typ,
                SourceInfo::new(
                    crate::transit::Locations::Multi(src_info.clone()),
                    Some("fused_arith".to_string()),
                ),
            ));

            // add tuple get to unzip the result
            let mut output_get = Vec::new();
            for (id, typ) in output_types.iter().enumerate() {
                let get = cg.g.add_vertex(Vertex::new(
                    VertexNode::TupleGet(node_id, id),
                    typ.clone(),
                    SourceInfo::new(
                        crate::transit::Locations::Multi(src_info.clone()),
                        Some("fused_arith_tuple_get".to_string()),
                    ),
                ));
                output_get.push(get);
            }

            // modify the consumers to depend on tuple get
            for i in 0..output_v.len() {
                let new_src = output_get[i];
                for (target, old_src) in output_v[i].iter() {
                    for src in cg.g.vertex_mut(*target).uses_mut() {
                        if *src == *old_src {
                            *src = new_src;
                        }
                    }
                }
            }
            // reconstruct succ and clear to and from
            fuse_id += 1;
            to = vec![false; cg.g.order()];
            from = vec![false; cg.g.order()];
            fused.resize(cg.g.order(), 0);
            succ = cg.g.successors().map(&mut |_, v| {
                v.into_iter()
                    .filter(|suc_id| fused[usize::from(*suc_id)] == 0)
                    .collect()
            });

            // // some check for succ and pred
            // for (vid, succs) in succ.iter().enumerate() {
            //     if fused[vid] != 0 {
            //         continue;
            //     }
            //     for suc in succs {
            //         if fused[usize::from(*suc)] != 0 {
            //             panic!("shouldn't have suc to be a fused point");
            //         }
            //     }
            // }

            // for vid in cg.g.vertices() {
            //     let vid_usize = usize::from(vid);
            //     if fused[vid_usize] != 0 {
            //         continue;
            //     }
            //     for pre in cg.g.vertex(vid).uses() {
            //         let pre_usize = usize::from(pre);
            //         if fused[pre_usize] != 0 {
            //             panic!("shouldn't have pre to be a fused point");
            //         }
            //     }
            // }
        }
    }
    cg
}
