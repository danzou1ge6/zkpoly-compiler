use std::collections::BTreeMap;

use zkpoly_common::{define_usize_id, heap::IdAllocator};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::type3::{Device, DeviceSpecific, Size, SmithereenSize};

use super::super::{Cg, Typ, VertexId};

define_usize_id!(ObjectId);

#[derive(Debug, Clone)]
pub enum Value {
    Tuple(Vec<ObjectId>),
    Single(ObjectId),
}

impl Value {
    pub fn object_ids<'s>(&'s self) -> Box<dyn Iterator<Item = ObjectId> + 's> {
        use Value::*;
        match self {
            Tuple(ss) => Box::new(ss.iter().cloned()),
            Single(s) => Box::new([s.clone()].into_iter()),
        }
    }
}

/// After which type2 vertex executes the object dies.
/// If an object is not used by any vertex on that device, the correspounding map will not contain the object.
pub type ObjectsDieAfter = DeviceSpecific<BTreeMap<ObjectId, VertexId>>;

pub type DeviceCollection = DeviceSpecific<bool>;

impl DeviceCollection {
    pub fn empty() -> Self {
        Self {
            gpu: false,
            cpu: false,
            stack: false,
        }
    }

    pub fn add(&mut self, device: Device) {
        match device {
            Device::Gpu => self.gpu = true,
            Device::Cpu => self.cpu = true,
            Device::Stack => self.stack = true,
        }
    }
    
    pub fn gpu(&self) -> bool {
        self.gpu
    }
    
    pub fn cpu(&self) -> bool {
        self.cpu
    }
    
    pub fn stack(&self) -> bool {
        self.stack
    }
}

pub struct ObjectsDieAfterReversed {
    pub after: BTreeMap<VertexId, BTreeMap<ObjectId, DeviceCollection>>,
}

impl ObjectsDieAfter {
    pub fn iter(&self) -> impl Iterator<Item = (Device, &BTreeMap<ObjectId, VertexId>)> {
        [
            (Device::Gpu, &self.gpu),
            (Device::Cpu, &self.cpu),
            (Device::Stack, &self.stack),
        ]
        .into_iter()
    }

    pub fn reversed(&self) -> ObjectsDieAfterReversed {
        let mut after = BTreeMap::new();

        self.iter().for_each(|(device, mapping)| {
            mapping.iter().for_each(|(oid, vid)| {
                after
                    .entry(*vid)
                    .or_insert_with(BTreeMap::new)
                    .entry(*oid)
                    .or_insert_with(DeviceCollection::empty)
                    .add(device);
            });
        });

        ObjectsDieAfterReversed { after }
    }
}

impl ObjectsDieAfter {
    pub fn empty() -> Self {
        Self {
            gpu: BTreeMap::new(),
            cpu: BTreeMap::new(),
            stack: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ObjectsDefUse {
    pub values: BTreeMap<VertexId, Value>,
    pub defs: BTreeMap<ObjectId, VertexId>,
    pub sizes: BTreeMap<ObjectId, Size>,
}

pub fn analyze_def_use<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
) -> (ObjectsDefUse, IdAllocator<ObjectId>) {
    let mut object_id_allocator = IdAllocator::new();
    let mut values: BTreeMap<VertexId, Value> = BTreeMap::new();
    let mut defs = BTreeMap::new();
    let mut sizes = BTreeMap::new();

    for (vid, v) in cg.g.topology_sort() {
        use super::super::template::VertexNode::*;
        match v.node() {
            TupleGet(pred, i) | ArrayGet(pred, i) => {
                let pred_value = values[pred].clone();

                match pred_value {
                    Value::Tuple(ss) => {
                        values.insert(vid, Value::Single(ss[*i]));
                    }
                    Value::Single(..) => panic!("expected array or tuple here"),
                }
            }
            Array(elements) => {
                let value = Value::Tuple(
                    elements
                        .iter()
                        .map(|e| match values[e].clone() {
                            Value::Tuple(..) => {
                                panic!("nested array or tuple not supported")
                            }
                            Value::Single(s) => s,
                        })
                        .collect(),
                );

                values.insert(vid, value);
            }
            _otherwise => {
                let value = match v.typ() {
                    Typ::Array(typ, len) => {
                        let elements = vec![object_id_allocator.alloc(); *len];

                        elements.iter().for_each(|elem| {
                            sizes.insert(*elem, typ.size().unwrap_single());
                        });

                        Value::Tuple(elements)
                    }
                    Typ::Tuple(elements) => Value::Tuple(elements.iter().map(|e| {
                        let id = object_id_allocator.alloc();
                        sizes.insert(id, e.size().unwrap_single());
                        id
                    }).collect()),
                    otherwise => {
                        let id = object_id_allocator.alloc();
                        sizes.insert(id, otherwise.size().unwrap_single());
                        Value::Single(id)
                    }
                };

                for oid in value.object_ids() {
                    defs.insert(oid, vid);
                }

                values.insert(vid, value);
            }
        }
    }

    (
        ObjectsDefUse {
            values,
            defs,
            sizes: sizes
                .into_iter()
                .map(|(id, s)| (id, Size::Smithereen(SmithereenSize(s))))
                .collect(),
        },
        object_id_allocator,
    )
}

pub fn analyze_die_after<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
    seq: &[VertexId],
    devices: &BTreeMap<VertexId, Device>,
    def_use: &ObjectsDefUse,
) -> ObjectsDieAfter {
    let mut die_after = ObjectsDieAfter::empty();
    for &vid in seq.iter() {
        cg.g.vertex(vid)
            .uses()
            .map(|input_vid| def_use.values[&input_vid].object_ids())
            .flatten()
            .for_each(|obj_id| {
                let device = devices[&vid];
                die_after.get_device_mut(device).insert(obj_id, vid);
            });
    }
    die_after
}
