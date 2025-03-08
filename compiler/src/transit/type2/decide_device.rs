use zkpoly_common::digraph::internal::{SubDigraph, Predecessors};
use zkpoly_runtime::args::RuntimeType;
use super::{VertexId, Device};
use crate::transit::type3::Device as DeterminedDevice;

use std::collections::BTreeMap;

pub fn decide<'s, Rt: RuntimeType>(
    g: &SubDigraph<'_, VertexId, super::Vertex<'s, Rt>>,
) -> BTreeMap<VertexId, DeterminedDevice> {
    let mut devices = BTreeMap::new();

    for (vid, v) in g.topology_sort() {
        let device = match v.device() {
            Device::PreferGpu => {
                if g.successors_of(vid)
                    .any(|vid| g.vertex(vid).device() == Device::Gpu)
                    || g.vertex(vid)
                        .predecessors()
                        .any(|vid| devices[&vid] == DeterminedDevice::Cpu)
                {
                    DeterminedDevice::Gpu
                } else {
                    DeterminedDevice::Cpu
                }
            }
            Device::Gpu => DeterminedDevice::Gpu,
            Device::Cpu => DeterminedDevice::Cpu,
        };

        let device = match device {
            DeterminedDevice::Cpu => {
                if v.typ().stack_allocable() {
                    DeterminedDevice::Stack
                } else {
                    DeterminedDevice::Cpu
                }
            }
            DeterminedDevice::Gpu => DeterminedDevice::Gpu,
            DeterminedDevice::Stack => unreachable!(),
        };

        devices.insert(vid, device);
    }

    devices
}

