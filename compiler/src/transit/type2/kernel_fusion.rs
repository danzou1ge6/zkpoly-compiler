use std::collections::{BTreeMap, BTreeSet};

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
    ) -> bool {
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
        arith2vid: &mut BTreeMap<ExprId, VertexId>,
        succ: &Heap<VertexId, BTreeSet<VertexId>>,
        ag: &mut ArithGraph<VertexId, ExprId>,
        fuse_id: usize,
    ) {
        if fused[usize::from(vid)] == fuse_id || vid2arith.contains_key(&vid) {
            return; // is in the current fusion
        }
        if self.can_fuse(vid, fused, to, from) {
            self.fuse_it(
                vid, fused, to, from, vid2arith, arith2vid, succ, ag, fuse_id,
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
            arith2vid.insert(vid_arith, vid);
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
        let can_fuse = self.can_fuse(vid, fused, to, from); // , None, None);
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
        let can_fuse = self.can_fuse(vid, fused, to, from); // , None, None);
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
        arith2vid: &mut BTreeMap<ExprId, VertexId>,
        succ: &Heap<VertexId, BTreeSet<VertexId>>,
        ag: &mut ArithGraph<VertexId, ExprId>,
        fuse_id: usize,
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
                arith2vid.insert(my_arith, vid);

                match arith {
                    Arith::Bin(_, lhs, rhs) => {
                        self.fuse_bwd(
                            *lhs, fused, to, from, vid2arith, arith2vid, succ, ag, fuse_id,
                        );
                        self.fuse_bwd(
                            *rhs, fused, to, from, vid2arith, arith2vid, succ, ag, fuse_id,
                        );
                    }
                    Arith::Unr(_, src) => {
                        self.fuse_bwd(
                            *src, fused, to, from, vid2arith, arith2vid, succ, ag, fuse_id,
                        );
                    }
                };
                *ag.g.vertex_mut(my_arith) = ArithVertex {
                    op: Operation::Arith(
                        arith.relabeled(&mut |vid| vid2arith.get(&vid).unwrap().clone()),
                    ),
                };

                let mut is_output = false;
                for node in succ[vid].iter() {
                    if fused[usize::from(*node)] == fuse_id {
                        continue; // is in the current fusion
                    }
                    if self.can_fuse(*node, fused, to, from) {
                        self.fuse_it(
                            *node, fused, to, from, vid2arith, arith2vid, succ, ag, fuse_id,
                        );
                    } else {
                        is_output = true;
                    }
                }
                if is_output {
                    let output_arith = ag.g.add_vertex(ArithVertex {
                        op: Operation::Output {
                            outer_id: vid,
                            typ: self.get_fuse_type(vid),
                            store_node: my_arith,
                            in_node: None,
                        },
                    });
                    ag.outputs.push(output_arith);
                    arith2vid.insert(output_arith, vid);
                }
            }
            _ => panic!("can only fuse single arith"),
        }
    }

    pub fn split(
        &self,
        ag: &ArithGraph<VertexId, ExprId>,
        arith2vid: BTreeMap<ExprId, VertexId>,
        size_limit: usize,
    ) -> (
        Vec<ArithGraph<VertexId, ExprId>>,
        Vec<BTreeMap<ExprId, VertexId>>,
    ) {
        if ag.g.order() <= size_limit {
            return (vec![ag.clone()], vec![arith2vid]);
        }
        let (schedule, live_ts, _) = ag.schedule();
        let partition = ag.partition(&schedule, size_limit);
        let mut new_ariths = Vec::new();
        let mut new_arith2vids = Vec::new();

        let need_store = ag
            .outputs
            .iter()
            .map(|vid| {
                let v = ag.g.vertex(*vid);
                if let Operation::Output { store_node, .. } = v.op {
                    store_node
                } else {
                    panic!("output node should be store node");
                }
            })
            .collect::<BTreeSet<_>>(); // mark the nodes that need to be stored
                                       // input nodes can be inferred from the arith graph

        for i in 0..(partition.len() - 1) {
            let start = partition[i];
            let end = partition[i + 1];
            let mut old_aid2new_aid: BTreeMap<ExprId, ExprId> = BTreeMap::new();
            let mut new_arith2vid = BTreeMap::new();
            let mut new_ag = ArithGraph {
                outputs: Vec::new(),
                inputs: Vec::new(),
                g: Digraph::new(),
                poly_repr: ag.poly_repr.clone(), // should be none, so we can use the same
                poly_degree: ag.poly_degree,     // should be none, so we can use the same
            };
            for j in start..end {
                let vid = schedule[j];
                let v = ag.g.vertex(vid);

                // only tackle arith nodes
                if let Operation::Arith(arith) = &v.op {
                    // check inputs
                    for in_id in v.uses() {
                        if !old_aid2new_aid.contains_key(&in_id) {
                            // this is an input
                            let outer_id = arith2vid.get(&in_id).unwrap().clone();
                            let typ = self.get_fuse_type(outer_id);
                            let new_in_id = new_ag.g.add_vertex(ArithVertex {
                                op: Operation::Input {
                                    outer_id,
                                    typ,
                                    mutability: Mutability::Const,
                                },
                            });
                            new_ag.inputs.push(new_in_id);
                            new_arith2vid.insert(new_in_id, outer_id);
                            old_aid2new_aid.insert(in_id, new_in_id);
                        }
                    }

                    // insert the arith node
                    let new_arith = new_ag.g.add_vertex(ArithVertex {
                        op: Operation::Arith(
                            arith.relabeled(&mut |vid| old_aid2new_aid.get(&vid).unwrap().clone()),
                        ),
                    });
                    old_aid2new_aid.insert(vid, new_arith);
                    new_arith2vid.insert(new_arith, arith2vid.get(&vid).unwrap().clone());

                    let live_ts_vid: usize = vid.into();
                    // check outputs
                    if live_ts[live_ts_vid] as usize >= end || need_store.contains(&vid) {
                        // this is an output
                        let outer_id = arith2vid.get(&vid).unwrap().clone();
                        let typ = self.get_fuse_type(outer_id);
                        let new_out_id = new_ag.g.add_vertex(ArithVertex {
                            op: Operation::Output {
                                outer_id,
                                typ,
                                store_node: new_arith,
                                in_node: None,
                            },
                        });
                        new_ag.outputs.push(new_out_id);
                        new_arith2vid.insert(new_out_id, outer_id);
                    }
                }
            }
            if new_ag.g.order() > 0 {
                new_ariths.push(new_ag);
                new_arith2vids.push(new_arith2vid);
            }
        }
        (new_ariths, new_arith2vids)
    }
}

fn get_poly_degree<Rt: RuntimeType>(output_types: &Vec<super::Typ<Rt>>) -> usize {
    output_types.iter().fold(0, |acc, t| match t {
        super::Typ::Poly((_, deg)) => {
            assert!(acc == 0 || acc == *deg as usize);
            *deg as usize
        }
        super::Typ::Scalar => acc,
        _ => panic!("outputs other than polynomials and scalars are not expected"),
    })
}

fn get_poly_repr<Rt: RuntimeType>(output_types: &Vec<super::Typ<Rt>>) -> PolyType {
    // get the polynomial representation
    output_types
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
        .unwrap_or(PolyType::Coef)
}

pub fn fuse_arith<'s, Rt: RuntimeType>(cg: Cg<'s, Rt>, gpu_mem_limit: u64) -> Cg<'s, Rt> {
    let succ: Heap<VertexId, BTreeSet<VertexId>> = cg.g.successors();
    let mut fused = vec![0; cg.g.order()];
    let order = cg.g.topology_sort().map(|(vid, _)| vid).collect::<Vec<_>>();
    let mut to = vec![false; cg.g.order()]; // vertexs rely on the current fused kernel
    let mut from = vec![false; cg.g.order()]; // vertexs the current fused kernel relies on
    let mut fuse_id = 1;

    let mut new_graph = Digraph::new();
    let mut old_id2new_id: BTreeMap<VertexId, VertexId> = BTreeMap::new();
    let mut tuple_gets = BTreeSet::new(); // tuple_gets are new added vertices, we don't need to relabel them

    for id in order {
        if cg.can_fuse(id, &fused, &to, &from) {
            let mut vid2arith = BTreeMap::new();
            let mut arith2vid = BTreeMap::new();
            let mut ag = ArithGraph {
                inputs: Vec::new(),
                outputs: Vec::new(),
                g: Digraph::new(),
                poly_repr: PolyType::Coef, // revised later
                poly_degree: None,         // revised later
            };

            cg.fuse_it(
                id,
                &mut fused,
                &mut to,
                &mut from,
                &mut vid2arith,
                &mut arith2vid,
                &succ,
                &mut ag,
                fuse_id,
            );

            // now we get a fused arith graph, we need to cut it into smaller ones
            // otherwise, the local memory usage will be too large
            // 10GB for 60k vertices with estimated reg usage 4000 on A100
            // 1GB for 5k vertices with estimated reg usage 1000 on A100

            let (ags, arith2vids) = cg.split(&ag, arith2vid, 8192); // 8192 is a magic number for largest arith graph, can be tuned

            // now we have a vector of arith graphs, we need to add them to the new graph
            for (mut ag, arith2vid) in ags.into_iter().zip(arith2vids.iter()) {
                // reorder inputs and outputs
                ag.reorder_input_outputs();

                // collect the output types
                let output_types = ag
                    .outputs
                    .iter()
                    .map(|output_id| {
                        let vid = arith2vid.get(output_id).unwrap();
                        let v = cg.g.vertex(*vid);
                        v.typ().clone()
                    })
                    .collect::<Vec<_>>();

                // fused arith type
                let typ = super::typ::template::Typ::Tuple(output_types.clone());

                // collect all the src infos
                let src_locations =
                    ag.g.vertices()
                        .into_iter()
                        .filter(&mut |arith_id: &ExprId| {
                            let arith = ag.g.vertex(*arith_id);
                            if let Operation::Arith(_) = arith.op {
                                true
                            } else {
                                false
                            }
                        }) // src info only for arith
                        .flat_map(|arith_id| {
                            let vid = arith2vid.get(&arith_id).unwrap();
                            cg.g.vertex(*vid).src().location.clone()
                        })
                        .collect::<Vec<_>>();

                let src_info = SourceInfo::new(src_locations, Some("fused_arith".to_string()));

                // decide the chunking
                ag.poly_degree = Some(get_poly_degree(&output_types));
                // let chunking = if ag.poly_degree.unwrap() > 16 {
                //     Some(16)
                // } else {
                //     None
                // };
                let chunking = ag.decide_chunking::<Rt::Field>(gpu_mem_limit);

                // decide the polynomial representation
                ag.poly_repr = get_poly_repr(&output_types);

                let arith_node = VertexNode::Arith {
                    arith: ag.clone(),
                    chunking,
                };

                // add the arith graph to the new graph
                let node_id = new_graph.add_vertex(Vertex::new(arith_node, typ, src_info));

                // add tuple get to unzip the result
                for ((id, typ), arithid) in output_types.iter().enumerate().zip(ag.outputs.iter()) {
                    let get = new_graph.add_vertex(Vertex::new(
                        VertexNode::TupleGet(node_id, id),
                        typ.clone(),
                        SourceInfo::new(Vec::new(), Some("fused_arith_tuple_get".to_string())),
                    ));
                    let old_id = arith2vid.get(arithid).unwrap().clone();
                    old_id2new_id.insert(old_id, get);
                    tuple_gets.insert(get);
                }
            }

            // reconstruct succ and clear to and from
            fuse_id += 1;
            to = vec![false; cg.g.order()];
            from = vec![false; cg.g.order()];
        } else if fused[usize::from(id)] == 0 {
            // not fused yet and not fusable
            let new_vid = new_graph.add_vertex(cg.g.vertex(id).clone());
            old_id2new_id.insert(id, new_vid);
        }
    }
    // now we need to relabel the graph
    for vid in new_graph.vertices() {
        if tuple_gets.contains(&vid) {
            continue;
        }
        let v = new_graph.vertex(vid).clone();
        let new_node = v
            .node()
            .relabeled(&mut |vid| old_id2new_id.get(&vid).unwrap().clone());
        *new_graph.vertex_mut(vid).node_mut() = new_node;
    }
    Cg {
        g: new_graph,
        output: old_id2new_id.get(&cg.output).unwrap().clone(),
    }
}
