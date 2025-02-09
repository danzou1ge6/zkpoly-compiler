use std::collections::BTreeMap;

use zkpoly_common::{define_usize_id, heap::IdAllocator};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::type3::{Device, DeviceSpecific, Size, SmithereenSize};

use super::super::{Cg, Typ, VertexId};

define_usize_id!(ObjectId);

#[derive(Debug, Clone)]
pub enum Value {
    Poly {
        rotation: i32,
        slice: (u64, u64),
        object_id: ObjectId,
    },
    Other(ObjectId)
}

impl Value {
    pub fn object_id(&self) -> ObjectId {
        match self {
            Value::Poly { object_id,.. } => *object_id,
            Value::Other(object_id) => *object_id,
        }
    }

    pub fn new<Rt: RuntimeType>(obj_id_allocator: &mut IdAllocator<ObjectId>, typ: &Typ<Rt>) -> Self {
        match typ {
            Typ::Poly {deg, ..}  => {
                let object_id = obj_id_allocator.alloc();
                Value::Poly {
                    rotation: 0,
                    slice: (0, *deg),
                    object_id,
                }
            }
            _otherwise => {
                let object_id = obj_id_allocator
                   .alloc();
                Value::Other(object_id)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum VertexValue {
    Tuple(Vec<Value>),
    Single(Value),
}

impl VertexValue {
    pub fn object_ids<'s>(&'s self) -> Box<dyn Iterator<Item = ObjectId> + 's> {
        use VertexValue::*;
        match self {
            Tuple(ss) => Box::new(ss.iter().map(Value::object_id)),
            Single(s) => Box::new([s.object_id()].into_iter()),
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
    pub values: BTreeMap<VertexId, VertexValue>,
    pub defs: BTreeMap<ObjectId, VertexId>,
    pub sizes: BTreeMap<ObjectId, Size>,
}

pub fn analyze_def_use<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
) -> (ObjectsDefUse, IdAllocator<ObjectId>) {
    let mut object_id_allocator = IdAllocator::new();
    let mut values: BTreeMap<VertexId, VertexValue> = BTreeMap::new();
    let mut defs = BTreeMap::new();
    let mut sizes = BTreeMap::new();

    for (vid, v) in cg.g.topology_sort() {
        use super::super::template::VertexNode::*;
        match v.node() {
            TupleGet(pred, i) | ArrayGet(pred, i) => {
                let pred_value = values[pred].clone();

                match pred_value {
                    VertexValue::Tuple(ss) => {
                        values.insert(vid, VertexValue::Single(ss[*i].clone()));
                    }
                    VertexValue::Single(..) => panic!("expected array or tuple here"),
                }
            }
            Array(elements) => {
                let value = VertexValue::Tuple(
                    elements
                        .iter()
                        .map(|e| match values[e].clone() {
                            VertexValue::Tuple(..) => {
                                panic!("nested array or tuple not supported")
                            }
                            VertexValue::Single(s) => s,
                        })
                        .collect(),
                );

                values.insert(vid, value);
            }
            RotateIdx(pred, delta) => {
                let pred_value = values[pred].clone();

                if let VertexValue::Single(Value::Poly { rotation, slice, object_id }) = pred_value {
                    let value = Value::Poly {
                        rotation: rotation + delta,
                        slice,
                        object_id,
                    };

                    values.insert(vid, VertexValue::Single(value));
                } else {
                    panic!("expected poly here");
                }
            }
            _otherwise => {
                let value = match v.typ() {
                    Typ::Array(typ, len) => {
                        let elements = vec![Value::new(&mut object_id_allocator, typ.as_ref()); *len];

                        elements.iter().for_each(|elem| {
                            sizes.insert(elem.object_id(), typ.size().unwrap_single());
                        });

                        VertexValue::Tuple(elements)
                    }
                    Typ::Tuple(elements) => VertexValue::Tuple(elements.iter().map(|e| {
                        let value = Value::new(&mut object_id_allocator, e);
                        sizes.insert(value.object_id(), e.size().unwrap_single());

                        value
                    }).collect()),
                    otherwise => {
                        let value = Value::new(&mut object_id_allocator, otherwise);
                        sizes.insert(value.object_id(), otherwise.size().unwrap_single());
                        VertexValue::Single(value)
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
