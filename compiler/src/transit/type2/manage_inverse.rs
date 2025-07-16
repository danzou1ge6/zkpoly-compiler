use zkpoly_common::arith::{Arith, ArithBinOp, ArithUnrOp, BinOp, SpOp, UnrOp};
use zkpoly_runtime::args::RuntimeType;

use super::template::SliceableNode;
use super::unsliced::{Cg, Vertex, VertexNode};

pub fn manage_inverse<'s, Rt: RuntimeType>(mut cg: Cg<'s, Rt>) -> Cg<'s, Rt> {
    let order = cg.g.vertices().collect::<Vec<_>>();
    for id in order.iter() {
        let vertex = cg.g.vertex(*id).clone();
        if let VertexNode::Sliceable(SliceableNode::SingleArith(arith)) = vertex.node() {
            match arith {
                Arith::Bin(bin_op, lhs, rhs) => {
                    match bin_op {
                        BinOp::Pp(op) => {
                            if *op == ArithBinOp::Div {
                                let inv_rhs = cg.g.add_vertex(Vertex::new(
                                    VertexNode::Sliceable(SliceableNode::BatchedInvert(*rhs)),
                                    vertex.typ().clone(),
                                    vertex.src().clone(),
                                ));
                                *cg.g.vertex_mut(*id) =
                                    Vertex::new(
                                        VertexNode::Sliceable(SliceableNode::SingleArith(
                                            Arith::Bin(BinOp::Pp(ArithBinOp::Mul), *lhs, inv_rhs),
                                        )),
                                        vertex.typ().clone(),
                                        vertex.src().clone(),
                                    );
                            }
                        }
                        BinOp::Sp(op) => {
                            match op {
                                SpOp::Div => {
                                    let inv_rhs = cg.g.add_vertex(Vertex::new(
                                        VertexNode::Sliceable(SliceableNode::BatchedInvert(*rhs)),
                                        vertex.typ().clone(),
                                        vertex.src().clone(),
                                    ));
                                    *cg.g.vertex_mut(*id) = Vertex::new(
                                        VertexNode::Sliceable(SliceableNode::SingleArith(
                                            Arith::Bin(BinOp::Sp(SpOp::Mul), *lhs, inv_rhs),
                                        )),
                                        vertex.typ().clone(),
                                        vertex.src().clone(),
                                    );
                                }
                                SpOp::DivBy => {
                                    let inv_lhs = cg.g.add_vertex(Vertex::new(
                                        VertexNode::Sliceable(SliceableNode::ScalarInvert(*lhs)),
                                        vertex.typ().clone(),
                                        vertex.src().clone(),
                                    ));
                                    *cg.g.vertex_mut(*id) = Vertex::new(
                                        VertexNode::Sliceable(SliceableNode::SingleArith(
                                            Arith::Bin(BinOp::Sp(SpOp::Mul), inv_lhs, *rhs),
                                        )),
                                        vertex.typ().clone(),
                                        vertex.src().clone(),
                                    );
                                }
                                _ => {}
                            }
                        }
                        BinOp::Ss(op) => {
                            if *op == ArithBinOp::Div {
                                let inv_rhs = cg.g.add_vertex(Vertex::new(
                                    VertexNode::Sliceable(SliceableNode::ScalarInvert(*rhs)),
                                    vertex.typ().clone(),
                                    vertex.src().clone(),
                                ));
                                *cg.g.vertex_mut(*id) =
                                    Vertex::new(
                                        VertexNode::Sliceable(SliceableNode::SingleArith(
                                            Arith::Bin(BinOp::Ss(ArithBinOp::Mul), *lhs, inv_rhs),
                                        )),
                                        vertex.typ().clone(),
                                        vertex.src().clone(),
                                    );
                            }
                        }
                    }
                }
                Arith::Unr(unr_op, target) => match unr_op {
                    UnrOp::P(op) => {
                        if *op == ArithUnrOp::Inv {
                            *cg.g.vertex_mut(*id) = Vertex::new(
                                VertexNode::Sliceable(SliceableNode::BatchedInvert(*target)),
                                vertex.typ().clone(),
                                vertex.src().clone(),
                            );
                        }
                    }
                    UnrOp::S(op) => {
                        if *op == ArithUnrOp::Inv {
                            *cg.g.vertex_mut(*id) = Vertex::new(
                                VertexNode::Sliceable(SliceableNode::ScalarInvert(*target)),
                                vertex.typ().clone(),
                                vertex.src().clone(),
                            );
                        }
                    }
                },
            }
        }
    }
    cg
}
