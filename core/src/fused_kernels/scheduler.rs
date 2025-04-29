use std::collections::{BinaryHeap, VecDeque};

use zkpoly_common::{arith::ArithGraph, heap::UsizeId};


fn check_seq_is_topology_sort<OuterId: UsizeId, InnerId: UsizeId + 'static>(
    seq: impl Iterator<Item = InnerId>,
    ag: &ArithGraph<OuterId, InnerId>,
) -> bool {
    let mut deg_in = ag.g.degrees_in_no_multiedge();
    let successors = ag.g.successors();

    for vid in seq {
        if deg_in[vid]!= 0 {
            return false;
        }
        for &succ_id in successors[vid].iter() {
            deg_in[succ_id] -= 1;
        }
    }

    true
}

pub fn schedule<OuterId: UsizeId, InnerId: UsizeId + 'static>(
    ag: &ArithGraph<OuterId, InnerId>,
) -> Vec<InnerId> {
    // reference: Optimal and Heuristic Min-Reg Scheduling Algorithms for GPU Programs, Algorithm 2
    // https://arxiv.org/abs/2303.06855
    let succ = ag.g.successors(); // follow the name in the paper
    let mut out_degs = ag.g.degrees_out();
    let mut max_rp = vec![0; ag.g.order()]; // max_rp[i] is the max register pressure at the subtree rooted at i, not accurate in dag (Sethi-Ullman algorithm)
    let def_sz = vec![1; ag.g.order()]; // all arith op have one output
    let mut live_ts: Vec<i32> = vec![-1; ag.g.order()]; //  the time-step at which the last use of node i is scheduled.
    let mut w = VecDeque::new();
    let mut q = BinaryHeap::new();
    let mut pushed = vec![false; ag.g.order()]; // to check if the node is already scheduled

    // build the max_rp
    for (vid, v) in ag.g.topology_sort() {
        let mut max_rp_sub_def_sz: Vec<(i32, InnerId)> = Vec::new();
        for child_id in v.uses() {
            max_rp_sub_def_sz.push((max_rp[child_id.into()] - def_sz[child_id.into()], child_id));
        }
        max_rp_sub_def_sz.sort(); // from the smallest to the largest max_rp - def_sz
        let mut cur_max_rp = 0;
        for (_, child_id) in max_rp_sub_def_sz.iter() {
            let child_id: usize = child_id.clone().into();
            cur_max_rp = max_rp[child_id].max(cur_max_rp + def_sz[child_id]);
        }
        max_rp[vid.into()] = cur_max_rp;
    }

    let mut cur_step = ag.g.order(); // start from the output i.e. the last step

    // push the output nodes into the queue
    for (vid, out_deg) in out_degs.iter().enumerate() {
        if *out_deg == 0 {
            q.push((-max_rp[vid] + def_sz[vid], vid)); // as the heap is a max heap, we need to negate the value
            live_ts[vid] = cur_step as i32; // output are always live at the last step
        }
    }

    // schedule the nodes
    let mut schedule = Vec::new();

    while !q.is_empty() {
        let (_, cur_id) = q.pop().unwrap();
        w.push_back(cur_id);
        while !w.is_empty() {
            let cur_id = w.pop_front().unwrap();
            schedule.push(cur_id.clone());
            for source_id in ag.g.vertex(InnerId::from(cur_id)).uses() {
                live_ts[source_id.into()] = (cur_step as i32).max(live_ts[source_id.into()]);
                out_degs[source_id] -= 1;
            }
            for pred_id in ag.g.vertex(InnerId::from(cur_id)).uses() {
                if out_degs[pred_id] == 0 && !pushed[pred_id.clone().into()] {
                    let cur_id: usize = cur_id.into();
                    let mut d = if live_ts[cur_id] > 0 {
                        def_sz[cur_id]
                    } else {
                        0
                    };
                    for succ_of_pred_id in succ[pred_id].iter() {
                        d = if live_ts[succ_of_pred_id.clone().into()] < 0 {
                            d - def_sz[succ_of_pred_id.clone().into()]
                        } else {
                            d
                        }
                    }
                    if d >= 0 {
                        w.push_back(pred_id.clone().into());
                    } else {
                        q.push((
                            -max_rp[pred_id.clone().into()] + def_sz[pred_id.clone().into()],
                            pred_id.clone().into(),
                        ));
                    }
                    pushed[pred_id.clone().into()] = true;
                }
            }
            cur_step = cur_step - 1;
        }
    }

    assert_eq!(
        schedule.len(),
        ag.g.order(),
        "The schedule size is not equal to the graph order"
    );

    schedule.reverse();
    let seq = schedule
        .iter()
        .map(|id| InnerId::from(*id))
        .collect::<Vec<_>>();

    if !check_seq_is_topology_sort(seq.iter().copied(), ag) {
        panic!("The schedule is not a topological sort");
    }

    seq
    // ag.g.topology_sort().map(|(a, _)| a).collect()
}
