use std::collections::BTreeMap;

use halo2curves::ff::Field;
use zkpoly_common::{arith, typ::PolyType};
use zkpoly_runtime::args::RuntimeType;

use crate::ast::RuntimeCorrespondance;

use super::{Cg, ConstantTable, Typ, Vertex, VertexId, VertexNode};

pub fn mend<'s, Rt: RuntimeType>(
    mut cg: Cg<'s, Rt>,
    constant_table: &mut ConstantTable<Rt>,
) -> Cg<'s, Rt> {
    let order = cg.g.vertices().collect::<Vec<_>>();

    let mut divisors = BTreeMap::new();
    let mut mended = BTreeMap::new();

    for vid in order {
        let replacements: BTreeMap<VertexId, VertexId> =
            cg.g.vertex(vid)
                .uses()
                .collect::<Vec<_>>()
                .into_iter()
                .map(|pred| match cg.g.vertex(pred).node() {
                    VertexNode::Ntt { to, from, .. } if *from == PolyType::Lagrange => {
                        assert!(*to == PolyType::Coef);

                        let replacement = *mended.entry(pred).or_insert_with(|| {
                            let (_, pred_deg) = cg.g.vertex(pred).typ().unwrap_poly();
                            let pred_deg = *pred_deg;

                            let divisor = *divisors.entry(pred_deg).or_insert_with(|| {
                                let f = Rt::Field::from(pred_deg).invert().unwrap();
                                let var = crate::ast::Scalar::to_variable(
                                    zkpoly_runtime::scalar::Scalar::from_ff(&f),
                                );
                                let cid = constant_table.push(super::Constant::new(
                                    var,
                                    format!("intt_divisor_{}", pred_deg),
                                ));
                                let v_constant = cg.g.add_vertex(Vertex::new(
                                    VertexNode::Constant(cid),
                                    Typ::Scalar,
                                    cg.g.vertex(pred).src().clone(),
                                ));
                                v_constant
                            });

                            let v_div = cg.g.add_vertex(Vertex::new(
                                VertexNode::SingleArith(arith::Arith::Bin(
                                    arith::BinOp::Sp(arith::SpOp::Mul),
                                    divisor,
                                    pred,
                                )),
                                Typ::coef(pred_deg),
                                cg.g.vertex(pred).src().clone(),
                            ));
                            v_div
                        });
                        (pred, replacement)
                    }
                    _ => (pred, pred),
                })
                .collect();
        cg.g.vertex_mut(vid)
            .uses_mut()
            .for_each(|pred| *pred = replacements[pred]);
    }

    cg
}
