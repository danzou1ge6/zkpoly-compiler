use std::collections::BTreeMap;

use zkpoly_common::{define_usize_id, heap::IdAllocator};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::type3::{Device, DeviceSpecific, Size, SmithereenSize, typ::Slice};

use super::super::{Cg, Typ, VertexId};

define_usize_id!(ObjectId);

#[derive(Debug, Clone)]
pub enum ValueNode {
    Poly { rotation: i32, slice: Slice },
    Other,
}

#[derive(Debug, Clone)]
pub struct Value {
    object_id: ObjectId,
    device: Device,
    node: ValueNode,
}

impl Value {
    pub fn object_id(&self) -> ObjectId {
        self.object_id
    }

    pub fn object_id_mut(&mut self) -> &mut ObjectId {
        &mut self.object_id
    }

    pub fn new<Rt: RuntimeType>(
        obj_id_allocator: &mut IdAllocator<ObjectId>,
        typ: &Typ<Rt>,
        device: Device,
    ) -> Self {
        match typ {
            Typ::Poly { deg, .. } => {
                let object_id = obj_id_allocator.alloc();
                Value {
                    node: ValueNode::Poly {
                        rotation: 0,
                        slice: Slice::new(0, *deg),
                    },
                    object_id,
                    device,
                }
            }
            _otherwise => {
                let object_id = obj_id_allocator.alloc();
                Value {
                    node: ValueNode::Other,
                    object_id,
                    device,
                }
            }
        }
    }

    pub fn with_device(&self, device: Device) -> Self {
        Value {
            node: self.node.clone(),
            object_id: self.object_id,
            device,
        }
    }

    pub fn node(&self) -> &ValueNode {
        &self.node
    }

    pub fn node_mut(&mut self) -> &mut ValueNode {
        &mut self.node
    }

    pub fn unwrap_poly(&self) -> (i32, Slice) {
        use ValueNode::*;
        match &self.node {
            Poly { rotation, slice } => {
                (*rotation, slice.clone())
            }
            _otherwise => panic!("not a poly"),
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

    pub fn with_device(&self, device: Device) -> Self {
        use VertexValue::*;
        match self {
            Tuple(ss) => VertexValue::Tuple(ss.iter().map(|s| s.with_device(device)).collect()),
            Single(s) => VertexValue::Single(s.with_device(device)),
        }
    }

    pub fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &Value> + 'a> {
        use VertexValue::*;
        match self {
            Tuple(ss) => Box::new(ss.iter()),
            Single(s) => Box::new([s].into_iter()),
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &mut Value> + 'a> {
        use VertexValue::*;
        match self {
            Tuple(ss) => Box::new(ss.iter_mut()),
            Single(s) => Box::new([s].into_iter()),
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

#[derive(Debug, Clone)]
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
    devices: impl Fn(VertexId) -> Device,
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

                if let VertexValue::Single(Value {
                    node: ValueNode::Poly { rotation, slice },
                    object_id,
                    device,
                }) = pred_value
                {
                    let value = Value {
                        node: ValueNode::Poly {
                            rotation: rotation + delta,
                            slice,
                        },
                        object_id,
                        device,
                    };

                    values.insert(vid, VertexValue::Single(value));
                } else {
                    panic!("expected poly here");
                }
            }
            _otherwise => {
                let value = match v.typ() {
                    Typ::Array(typ, len) => {
                        let elements =
                            vec![
                                Value::new(&mut object_id_allocator, typ.as_ref(), devices(vid));
                                *len
                            ];

                        elements.iter().for_each(|elem| {
                            sizes.insert(elem.object_id(), typ.size().unwrap_single());
                        });

                        VertexValue::Tuple(elements)
                    }
                    Typ::Tuple(elements) => VertexValue::Tuple(
                        elements
                            .iter()
                            .map(|e| {
                                let value = Value::new(&mut object_id_allocator, e, devices(vid));
                                sizes.insert(value.object_id(), e.size().unwrap_single());

                                value
                            })
                            .collect(),
                    ),
                    otherwise => {
                        let value = Value::new(&mut object_id_allocator, otherwise, devices(vid));
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

#[derive(Debug, Clone)]
pub struct VertexInputs {
    pub(super) inputs: BTreeMap<VertexId, Vec<VertexValue>>,
    pub(super) cloned_slices: BTreeMap<ObjectId, (ObjectId, Slice)>,
}

fn collect_slice_range_on_device<'s, Rt: RuntimeType>(
    device: Device,
    vi: &BTreeMap<VertexId, Vec<VertexValue>>,
    cg: &Cg<'s, Rt>,
    devices: &impl Fn(VertexId) -> Device,
    obj_id_allocator: &mut IdAllocator<ObjectId>,
) -> BTreeMap<ObjectId, (ObjectId, Slice)> {
    let mut slice_range = BTreeMap::new();

    for vid in cg.g.vertices() {
        if devices(vid) != device {
            continue;
        }

        vi[&vid].iter().for_each(|vv| {
            vv.iter().for_each(|value| {
                if let ValueNode::Poly { slice, .. } = value.node() {
                    let (_, old_slice) = slice_range
                        .entry(value.object_id())
                        .or_insert_with(|| (obj_id_allocator.alloc(), slice.clone()));
                    old_slice.union_with(slice);
                }
            });
        })
    }

    slice_range
}

pub fn plan_vertex_inputs<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
    def_use: &ObjectsDefUse,
    devices: impl Fn(VertexId) -> Device,
    obj_id_allocator: &mut IdAllocator<ObjectId>,
) -> VertexInputs {
    let mut inputs = BTreeMap::new();

    for vid in cg.g.vertices() {
        let v = cg.g.vertex(vid);

        inputs.insert(
            vid,
            v.uses()
                .map(|input_vid| def_use.values[&input_vid].with_device(devices(vid)))
                .collect(),
        );
    }

    let gpu_slice_range =
        collect_slice_range_on_device(Device::Gpu, &inputs, cg, &devices, obj_id_allocator);

    let cloned_slices: BTreeMap<ObjectId, (ObjectId, Slice)> = gpu_slice_range
        .iter()
        .map(|(original_obj, (sliced_obj, slice))| (*sliced_obj, (*original_obj, slice.clone())))
        .collect();

    for vid in cg.g.vertices() {
        if devices(vid) != Device::Gpu {
            continue;
        }

        inputs.get_mut(&vid).unwrap().iter_mut().for_each(|vv| {
            vv.iter_mut().for_each(|value| {
                let original_obj = value.object_id();

                let sliced_obj_id = if let ValueNode::Poly { slice, .. } = value.node_mut() {
                    let (sliced_obj, complete_slice) = gpu_slice_range[&original_obj].clone();

                    *slice = slice.relative_of(&complete_slice);
                    Some(sliced_obj)
                } else {
                    None
                };

                if let Some(sliced_obj_id) = sliced_obj_id {
                    *value.object_id_mut() = sliced_obj_id;
                }
            });
        });
    }

    VertexInputs {
        inputs,
        cloned_slices,
    }
}

pub fn analyze_die_after(
    seq: &[VertexId],
    devices: &BTreeMap<VertexId, Device>,
    vertex_inputs: &VertexInputs,
) -> ObjectsDieAfter {
    let mut die_after = ObjectsDieAfter::empty();
    for &vid in seq.iter() {
        vertex_inputs.inputs[&vid].iter()
            .map(|input_vv| input_vv.iter())
            .flatten()
            .for_each(|input_value| {
                let device = devices[&vid];
                die_after.get_device_mut(device).insert(input_value.object_id(), vid);
            });
    }
    die_after
}
