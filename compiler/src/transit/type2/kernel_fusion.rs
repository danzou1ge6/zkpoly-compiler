use std::{
    collections::{BTreeMap, BTreeSet},
    panic::Location,
};

use crate::transit::SourceInfo;

use super::{Cg, Vertex, VertexId, VertexNode};
use zkpoly_common::{
    arith::{Arith, ArithGraph, ArithVertex, ExprId, FusedType, Mutability, Operation},
    digraph::internal::Digraph,
    heap::Heap,
};
use zkpoly_runtime::args::RuntimeType;

impl<'s, Rt: RuntimeType> Cg<'s, Rt> {
    fn can_fuse(&self, vid: VertexId, book: &Vec<bool>) -> bool {
        let v = self.g.vertex(vid);
        let vid_usize: usize = vid.into();
        if book[vid_usize] {
            return false;
        }
        match v.node() {
            VertexNode::SingleArith(_) => true,
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
        book: &mut Vec<bool>,
        vid2arith: &mut BTreeMap<VertexId, ExprId>,
        succ: &Heap<VertexId, BTreeSet<VertexId>>,
        ag: &mut ArithGraph<VertexId, ExprId>,
        output_v: &mut Vec<Vec<(VertexId, VertexId)>>,
        src_info: &mut Vec<Location<'s>>,
    ) {
        if vid2arith.contains_key(&vid) {
            return;
        }
        if self.can_fuse(vid, book) {
            self.fuse_it(vid, book, vid2arith, succ, ag, output_v, src_info);
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

    fn fuse_it(
        &self,
        vid: VertexId,
        book: &mut Vec<bool>,
        vid2arith: &mut BTreeMap<VertexId, ExprId>,
        succ: &Heap<VertexId, BTreeSet<VertexId>>,
        ag: &mut ArithGraph<VertexId, ExprId>,
        output_v: &mut Vec<Vec<(VertexId, VertexId)>>, // output_v[rank in outputs]: vec[(targetid, old_src_id)]
        src_info: &mut Vec<Location<'s>>,
    ) {
        let v = self.g.vertex(vid);
        match v.node() {
            VertexNode::SingleArith(arith) => {
                let vid_usize: usize = vid.into();
                book[vid_usize] = true;
                src_info.push(v.src().unwrap_location_single().clone());
                match arith {
                    Arith::Bin(_, lhs, rhs) => {
                        self.fuse_bwd(*lhs, book, vid2arith, succ, ag, output_v, src_info);
                        self.fuse_bwd(*rhs, book, vid2arith, succ, ag, output_v, src_info);
                    }
                    Arith::Unr(_, src) => {
                        self.fuse_bwd(*src, book, vid2arith, succ, ag, output_v, src_info);
                    }
                };
                let my_arith = ag.g.add_vertex(ArithVertex {
                    op: Operation::Arith(
                        arith.relabeled(&mut |vid| vid2arith.get(&vid).unwrap().clone()),
                    ),
                });
                vid2arith.insert(vid, my_arith);
                let mut write_back = false;
                for node in succ[vid].iter() {
                    if vid2arith.contains_key(node) {
                        continue;
                    }
                    if self.can_fuse(*node, book) {
                        self.fuse_it(*node, book, vid2arith, succ, ag, output_v, src_info);
                    } else {
                        if !write_back {
                            output_v.push(Vec::new())
                        };
                        write_back = true;
                        output_v[ag.outputs.len()].push((*node, vid));
                    }
                }
                if write_back {
                    ag.outputs.push(my_arith);
                }
            }
            _ => panic!("can only fuse single arith"),
        }
    }

    fn check_mutability(
        &self,
        ag: &mut ArithGraph<VertexId, ExprId>,
        succ: &Heap<VertexId, BTreeSet<VertexId>>,
        mut_limit: Option<(usize, usize)>,
    ) -> Option<(usize, usize)> {
        // change mutability
        let succ_arith = ag.g.successors();
        let mut mut_scalars = 0;
        let mut mut_polys = 0;
        for id in ag.inputs.iter() {
            let v = ag.g.vertex_mut(*id);
            if let Operation::Input {
                outer_id,
                typ,
                mutability,
            } = &mut v.op
            {
                if succ_arith[*id].len() == succ[*outer_id].len() {
                    // *mutability = Mutability::Mut; TODO: change this to meet actual need of mutablility
                    match typ {
                        FusedType::Scalar => mut_scalars += 1,
                        FusedType::ScalarArray => mut_polys += 1,
                    }
                    if mut_limit.is_some() {
                        let (scalar_limit, poly_limit) = mut_limit.unwrap();
                        if (*typ == FusedType::Scalar && scalar_limit <= mut_scalars)
                            || (*typ == FusedType::ScalarArray && poly_limit <= mut_polys)
                        {
                            *mutability = Mutability::Mut;
                        }
                    }
                }
            } else {
                panic!("arith vertex in the inputs table should be inputs")
            }
        }
        if mut_limit.is_none() {
            Some((mut_scalars, mut_polys))
        } else {
            None
        }
    }
}

pub fn fuse_arith<'s, Rt: RuntimeType>(mut cg: Cg<'s, Rt>) -> Cg<'s, Rt> {
    let succ: Heap<VertexId, BTreeSet<VertexId>> = cg.g.successors();
    let mut book = vec![false; cg.g.order()];
    let dfs_order = cg.g.dfs().map(|(id, _)| id).collect::<Vec<_>>();
    for id in dfs_order {
        let id_usize: usize = id.clone().into();
        if !book[id_usize] {
            let mut vid2arith = BTreeMap::new();
            let mut ag = ArithGraph {
                inputs: Vec::new(),
                outputs: Vec::new(),
                g: Digraph::new(),
            };
            let mut output_v = Vec::new();
            let mut src_info = Vec::new();
            cg.fuse_it(
                id,
                &mut book,
                &mut vid2arith,
                &succ,
                &mut ag,
                &mut output_v,
                &mut src_info,
            );

            let (mut mut_scalars, mut mut_polys) =
                cg.check_mutability(&mut ag, &succ, None).unwrap(); // first call to get mutable input counts

            // push the node
            let mut output_scalars = 0;
            let mut output_polys = 0;
            let output_types = output_v
                .iter()
                .map(|output_i /*(targetid, old_src_id)*/| {
                    let old_src = output_i[0].1;
                    let old = cg.g.vertex(old_src);
                    match cg.get_fuse_type(old_src) {
                        FusedType::Scalar => output_scalars += 1,
                        FusedType::ScalarArray => output_polys += 1,
                    }
                    old.typ().clone()
                })
                .collect::<Vec<_>>();

            mut_scalars = mut_scalars.min(output_scalars);
            mut_polys = mut_polys.min(output_polys);

            // second call to actually update the inputs need to be mutable
            cg.check_mutability(&mut ag, &succ, Some((mut_scalars, mut_polys)));

            assert_eq!(output_types.len(), ag.outputs.len());
            let typ = super::typ::template::Typ::Tuple(output_types.clone());

            let node_id = cg.g.add_vertex(Vertex::new(
                VertexNode::Arith {
                    arith: ag,
                    mut_scalars,
                    mut_polys,
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

            // modify the succ
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
        }
    }
    cg
}
