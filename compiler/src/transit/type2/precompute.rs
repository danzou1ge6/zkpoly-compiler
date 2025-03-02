use std::{collections::BTreeMap, panic::Location};

use halo2curves::group::cofactor;
use zkpoly_common::{load_dynamic::Libs, msm_config::MsmConfig, typ::PolyType};
use zkpoly_core::{
    msm::MSMPrecompute,
    ntt::{GenPqOmegas, SsipPrecompute},
};
use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_runtime::{
    args::{RuntimeType, Variable},
    point::PointArray,
    scalar::ScalarArray,
};

use crate::{
    ast::lowering::{Constant, ConstantTable},
    transit::{
        type2::{typ::template::Typ, NttAlgorithm, VertexId, VertexNode},
        Locations, SourceInfo, Vertex,
    },
    utils::GenOmega,
};

use super::Cg;

pub fn precompute<'s, Rt: RuntimeType>(
    mut cg: Cg<'s, Rt>,
    memory_limit: usize,
    libs: &mut Libs,
    allocator: &mut PinnedMemoryPool,
    constant_tb: &mut ConstantTable<Rt>,
) -> Cg<'s, Rt> {
    let order = cg.g.dfs().map(|(id, _)| id).collect::<Vec<_>>();
    let mut gen_omega = GenOmega::<Rt::Field>::new();
    let mut precompute_ntts: BTreeMap<(u32, bool), NttAlgorithm<VertexId>> = BTreeMap::new();

    let mut msm_precomputes: BTreeMap<(VertexId, MsmConfig), Vec<VertexId>> = BTreeMap::new(); // points -> precomputed points
    for id in order.iter() {
        let vertex = cg.g.vertex(*id).clone();
        match vertex.node() {
            super::template::VertexNode::Ntt { s, to, from, alg } => {
                let inv = match to {
                    PolyType::Coef => {
                        assert_eq!(*from, PolyType::Lagrange);
                        true
                    }
                    PolyType::Lagrange => {
                        assert_eq!(*from, PolyType::Coef);
                        false
                    }
                    _ => unreachable!(),
                };
                let (_, len) = vertex.typ().unwrap_poly();
                assert!(len.is_power_of_two());
                let log_len = len.trailing_zeros();

                let old_alg = precompute_ntts.get(&(log_len, inv));

                if old_alg.is_some() {
                    *cg.g.vertex_mut(*id).node_mut().unwrap_ntt_alg_mut() =
                        old_alg.unwrap().clone()
                } else {
                    let omega = gen_omega.get_omega(log_len, inv);
                    let gen_pq_omegas = GenPqOmegas::<Rt>::new(libs);

                    let recompute_omega_len = 32;
                    let recompute_pq_len = gen_pq_omegas.get_pq_len(log_len);
                    let recompute_len = recompute_omega_len + recompute_pq_len;
                    let mut alg = NttAlgorithm::<VertexId>::decide_alg::<Rt>(
                        len.clone().try_into().unwrap(),
                        memory_limit,
                        recompute_len,
                    );

                    match &mut alg {
                        NttAlgorithm::Precomputed(omegas_id) => {
                            let precompute = SsipPrecompute::<Rt>::new(libs);
                            let f = precompute.get_fn();
                            let mut omega_array =
                                ScalarArray::alloc_cpu((len / 2) as usize, allocator);
                            f(&mut omega_array, &omega).unwrap();
                            let c_id = constant_tb.push(crate::ast::lowering::Constant {
                                name: Some(format!(
                                    "precompute twiddle factor for ntt k = {log_len}, inv: {inv}"
                                )),
                                value: Variable::ScalarArray(omega_array),
                            });
                            let load_c = cg.g.add_vertex(Vertex::new(
                                VertexNode::Constant(c_id),
                                Typ::Poly((PolyType::Coef, len / 2)),
                                SourceInfo::new(
                                    Locations::Single(*Location::caller()),
                                    Some("load precomputed twiddle factors".to_string()),
                                ),
                            ));
                            *omegas_id = load_c;
                        }
                        NttAlgorithm::Standard {
                            pq: pq_id,
                            omega: omegas_id,
                        } => {
                            let f = gen_pq_omegas.get_fn();
                            let mut pq = ScalarArray::alloc_cpu(recompute_pq_len, allocator);
                            let mut omegas = ScalarArray::alloc_cpu(recompute_omega_len, allocator);
                            f(&mut pq, &mut omegas, *len as usize, &omega).unwrap();
                            let pq_cid = constant_tb.push(crate::ast::lowering::Constant {
                                name: Some(format!("pq for ntt k = {log_len}, inv: {inv}")),
                                value: Variable::ScalarArray(pq),
                            });
                            let omegas_cid = constant_tb.push(crate::ast::lowering::Constant {
                                name: Some(format!(
                                    "omega bases for ntt k = {log_len}, inv: {inv}"
                                )),
                                value: Variable::ScalarArray(omegas),
                            });
                            let load_pq = cg.g.add_vertex(Vertex::new(
                                VertexNode::Constant(pq_cid),
                                Typ::Poly((PolyType::Coef, recompute_pq_len as u64)),
                                SourceInfo::new(
                                    Locations::Single(*Location::caller()),
                                    Some("load precomputed twiddle factors".to_string()),
                                ),
                            ));
                            let load_omegas = cg.g.add_vertex(Vertex::new(
                                VertexNode::Constant(omegas_cid),
                                Typ::Poly((PolyType::Coef, recompute_omega_len as u64)),
                                SourceInfo::new(
                                    Locations::Single(*Location::caller()),
                                    Some("load precomputed twiddle factors".to_string()),
                                ),
                            ));
                            *pq_id = load_pq;
                            *omegas_id = load_omegas;
                        }
                        NttAlgorithm::Undecieded => {
                            unreachable!("returned algorithm can't be undecided")
                        }
                    }
                    *cg.g.vertex_mut(*id).node_mut().unwrap_ntt_alg_mut() = alg.clone();
                    precompute_ntts.insert((log_len, inv), alg);
                }
            }
            super::template::VertexNode::Msm { points, .. } => {
                assert_eq!(points.len(), 1);
                let input_id = points[0];
                let config = MsmConfig::default();

                let precompute = msm_precomputes.get(&(input_id, config.clone()));

                if precompute.is_some() {
                    let (_, points, old_config) = cg.g.vertex_mut(*id).node_mut().unwrap_msm();
                    *points = precompute.unwrap().clone();
                    *old_config = config;
                } else {
                    let msm_precompute_func = MSMPrecompute::<Rt>::new(libs, config.clone());
                    let f = msm_precompute_func.get_fn();
                    let input_constant_id = cg.g.vertex(input_id).node().unwrap_constant();
                    let input_points =
                        (*constant_tb[*input_constant_id].value.unwrap_point_array()).clone();
                    let len = input_points.len;
                    let mut point_arrays = vec![input_points];
                    for _ in 1..config.get_precompute() {
                        let point_array = PointArray::<Rt::PointAffine>::new(
                            len,
                            allocator.allocate(len),
                            zkpoly_runtime::devices::DeviceType::CPU,
                        );
                        point_arrays.push(point_array);
                    }
                    f(
                        point_arrays.iter_mut().map(|array| array).collect(),
                        len,
                        4, // allow at most 4 cards to do precompute together
                    )
                    .unwrap();

                    let c_ids = point_arrays
                        .into_iter()
                        .skip(1)
                        .map(|array| {
                            constant_tb.push(Constant {
                                name: Some("precompute points for msm".to_string()),
                                value: Variable::PointArray(array),
                            })
                        })
                        .collect::<Vec<_>>();

                    let mut load_ids = vec![input_id];
                    for c_id in c_ids {
                        let load_id = cg.g.add_vertex(Vertex::new(
                            VertexNode::Constant(c_id),
                            cg.g.vertex(input_id).typ().clone(),
                            cg.g.vertex(input_id).src().clone(),
                        ));
                        load_ids.push(load_id);
                    }
                    msm_precomputes.insert((input_id, config.clone()), load_ids.clone());

                    let (_, points, old_config) = cg.g.vertex_mut(*id).node_mut().unwrap_msm();
                    *points = load_ids;
                    *old_config = config;
                }
            }
            _ => {}
        }
    }

    cg
}
