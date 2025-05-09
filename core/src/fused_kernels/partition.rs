// implementation of the partitioning algorithm
// Optimal Sequential Partitions of Graphs
// BRIAN W. KERNIGHAN
// Bell Telephone Laboratories, Incorporated, Murray Hill, New Jersey
// https://dl.acm.org/doi/pdf/10.1145/321623.321627

use std::collections::HashMap;

use zkpoly_common::{arith::ArithGraph, heap::UsizeId};

pub fn partition<OuterId: UsizeId, InnerId: UsizeId + 'static>(
    ag: &ArithGraph<OuterId, InnerId>,
    order: &Vec<InnerId>,
    chunk_upper_bound: usize,
) -> Vec<usize> {
    let len = ag.g.order();
    let out_degs = ag.g.degrees_out_no_multiedge();
    let succ = ag.g.successors();
    let mut t = vec![0; len + 1];
    let mut l = vec![0; len + 1];
    let mut c = HashMap::new();
    for i in 0..=len as i64 {
        c.insert((i, i), 0); // c(i, i) = 0
    }
    for x in 1..=(len as i64) {
        let mut minimized_y = -1;
        let mut relevent_in_degree = 0; // c_(y,x-1) + c_(y + 1, x-1) + ... + c_(x-1, x-1)
        let changing_point = order[(x - 1) as usize];

        for y in ((x - chunk_upper_bound as i64).max(0)..x).rev(){
            let y_point = order[y as usize];
            relevent_in_degree += succ[y_point].contains(&changing_point) as usize;
            let c_xy = c[&(x - 1, y)] + out_degs[changing_point] - relevent_in_degree;
            c.insert((x, y), c_xy);
            let new_val = t[y as usize] + c_xy;
            if minimized_y == -1 || t[x as usize] >= new_val {
                minimized_y = y;
                t[x as usize] = new_val;
            }
        }
        l[x as usize] = minimized_y as usize;
    }
    let mut breaking_points = vec![len];
    let mut cur_rank = len;
    while cur_rank > 0 {
        breaking_points.push(l[cur_rank]);
        cur_rank = l[cur_rank];
    }
    breaking_points.reverse();
    breaking_points
}
