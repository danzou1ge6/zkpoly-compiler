use super::{Device, VertexId};
use zkpoly_common::digraph::internal::{Predecessors, SubDigraph};
use zkpoly_runtime::args::RuntimeType;

use std::collections::BTreeMap;

/// Decides on which device a vertex is executed.
pub fn decide<'s, Rt: RuntimeType>(
    g: &SubDigraph<'_, VertexId, super::Vertex<'s, Rt>>,
) -> BTreeMap<VertexId, Device> {
    let mut devices = BTreeMap::new();

    for (vid, v) in g.topology_sort() {
        use super::DevicePreference::*;
        let device = match v.device() {
            PreferGpu => {
                if g.successors_of(vid)
                    .any(|vid| g.vertex(vid).device() == Gpu)
                    || g.vertex(vid)
                        .predecessors()
                        .any(|vid| devices[&vid] == Device::Cpu)
                {
                    Device::Gpu(0)
                } else {
                    Device::Cpu
                }
            }
            Gpu => Device::Gpu(0),
            Cpu => Device::Cpu,
        };

        devices.insert(vid, device);
    }

    devices
}
