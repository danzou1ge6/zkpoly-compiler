use std::collections::BTreeMap;

use zkpoly_common::{define_usize_id, heap::IdAllocator, typ::PolyMeta};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::type3::{
    typ::Slice as SliceRange, Device, DeviceSpecific, Size, SmithereenSize,
};

use super::super::{Cg, Typ, VertexId};

define_usize_id!(ObjectId);

#[derive(Debug, Clone)]
pub enum ValueNode {
    SlicedPoly { slice: SliceRange, deg: u64 },
    Poly { rotation: i32, deg: u64 },
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
            Typ::Poly((_, deg)) => {
                let object_id = obj_id_allocator.alloc();
                Value {
                    node: ValueNode::Poly {
                        rotation: 0,
                        deg: *deg,
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

    pub fn unwrap_single(&self) -> &Value {
        use VertexValue::*;
        match self {
            Single(s) => s,
            _ => panic!("called unwrap_single on VertexValue::Tuple"),
        }
    }

    pub fn unwrap_single_mut(&mut self) -> &mut Value {
        use VertexValue::*;
        match self {
            Single(s) => s,
            _ => panic!("called unwrap_single on VertexValue::Tuple"),
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
    pub cloned_slices: BTreeMap<(ObjectId, PolyMeta), ObjectId>,
}

fn rotated_offset(begin: u64, offset: i64, cycle: u64) -> u64 {
    (begin as i64 + offset) as u64 % cycle
}

pub fn analyze_def_use<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
    devices: impl Fn(VertexId) -> Device,
) -> (ObjectsDefUse, IdAllocator<ObjectId>) {
    let mut object_id_allocator = IdAllocator::new();
    let mut values: BTreeMap<VertexId, VertexValue> = BTreeMap::new();
    let mut defs = BTreeMap::new();
    let mut sizes = BTreeMap::new();
    let mut cloned_slices = BTreeMap::new();

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
                let pred_value = values[pred].unwrap_single();

                let value = match pred_value.node() {
                    ValueNode::Poly { rotation, deg } => Value {
                        node: ValueNode::Poly {
                            rotation: *rotation + *delta,
                            deg: *deg,
                        },
                        device: pred_value.device,
                        object_id: pred_value.object_id,
                    },
                    ValueNode::SlicedPoly { slice, deg } => {
                        // Rotation after slice is not implemented, so we clone the slice to a new polynomial
                        let cloned_obj_id = object_id_allocator.alloc();
                        cloned_slices.insert(
                            (pred_value.object_id(), PolyMeta::Sliced(*slice)),
                            cloned_obj_id,
                        );
                        Value {
                            node: ValueNode::Poly {
                                rotation: *delta,
                                deg: slice.len(),
                            },
                            // Let it be pred_value.device for now, we will update it later
                            device: pred_value.device,
                            object_id: cloned_obj_id,
                        }
                    }
                    _ => panic!("only polynomials can be rotated"),
                };

                values.insert(vid, VertexValue::Single(value));
            }
            Slice(pred, begin, end) => {
                let pred_value = values[pred].unwrap_single();

                let value = match pred_value.node() {
                    ValueNode::Poly { rotation, deg } => Value {
                        node: ValueNode::SlicedPoly {
                            slice: SliceRange::new(
                                rotated_offset(*begin, *rotation as i64, *deg),
                                *end - *begin,
                            ),
                            deg: *deg,
                        },
                        device: pred_value.device,
                        object_id: pred_value.object_id,
                    },
                    ValueNode::SlicedPoly { slice, deg } => Value {
                        node: ValueNode::SlicedPoly {
                            slice: SliceRange::new(
                                rotated_offset(slice.begin(), *begin as i64, *deg),
                                *end - slice.begin(),
                            ),
                            deg: *deg,
                        },
                        device: pred_value.device,
                        object_id: pred_value.object_id,
                    },
                    _ => panic!("only polynomials can be sliced"),
                };

                values.insert(vid, VertexValue::Single(value));
            }
            otherwise => {
                assert!(!otherwise.is_virtual());

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
            cloned_slices,
        },
        object_id_allocator,
    )
}

#[derive(Debug, Clone)]
pub struct VertexInputs {
    pub(super) inputs: BTreeMap<VertexId, Vec<VertexValue>>,
    pub(super) cloned_slices_reversed: BTreeMap<ObjectId, (ObjectId, PolyMeta)>,
}

pub fn plan_vertex_inputs<'s, Rt: RuntimeType>(
    cg: &Cg<'s, Rt>,
    def_use: &mut ObjectsDefUse,
    devices: impl Fn(VertexId) -> Device,
    obj_id_allocator: &mut IdAllocator<ObjectId>,
) -> VertexInputs {
    let mut inputs: BTreeMap<VertexId, Vec<VertexValue>> = BTreeMap::new();

    for vid in cg.g.vertices() {
        let v = cg.g.vertex(vid);

        inputs.insert(
            vid,
            v.uses()
                .map(|input_vid| def_use.values[&input_vid].with_device(devices(vid)))
                .collect(),
        );
    }

    // - SlicedPoly not allowed on GPU
    for vid in cg.g.vertices() {
        if devices(vid) != Device::Gpu {
            continue;
        }

        inputs.get_mut(&vid).unwrap().iter_mut().for_each(|vv| {
            vv.iter_mut().for_each(|value| {
                let obj_id = value.object_id();

                let new_value = match value.node() {
                    ValueNode::SlicedPoly { slice, .. } => {
                        let cloned_obj_id = *def_use
                            .cloned_slices
                            .entry((obj_id, PolyMeta::Sliced(*slice)))
                            .or_insert_with(|| obj_id_allocator.alloc());

                        Some(Value {
                            node: ValueNode::Poly {
                                rotation: 0,
                                deg: slice.len(),
                            },
                            device: Device::Gpu,
                            object_id: cloned_obj_id,
                        })
                    }
                    _ => None,
                };
                if let Some(new_value) = new_value {
                    *value = new_value;
                }
            });
        });
    }

    // - Input polynomials of MSM cannot be rotated or sliced
    for vid in cg.g.vertices() {
        use super::super::template::VertexNode::*;
        match cg.g.vertex(vid).node() {
            Msm { polys, .. } => inputs.get_mut(&vid).unwrap().iter_mut().for_each(|vv| {
                let value = vv.unwrap_single_mut();
                let obj_id = value.object_id();

                let new_value = match value.node() {
                    ValueNode::SlicedPoly { slice, .. } => {
                        let cloned_oj_id = *def_use
                            .cloned_slices
                            .entry((obj_id, PolyMeta::Sliced(*slice)))
                            .or_insert_with(|| obj_id_allocator.alloc());

                        Some(Value {
                            node: ValueNode::Poly {
                                rotation: 0,
                                deg: slice.len(),
                            },
                            device: Device::Gpu,
                            object_id: cloned_oj_id,
                        })
                    }
                    ValueNode::Poly { rotation, deg } if *rotation != 0 => {
                        let cloned_oj_id = *def_use
                            .cloned_slices
                            .entry((obj_id, PolyMeta::Rotated(*rotation)))
                            .or_insert_with(|| obj_id_allocator.alloc());

                        Some(Value {
                            node: ValueNode::Poly {
                                rotation: 0,
                                deg: *deg,
                            },
                            device: Device::Gpu,
                            object_id: cloned_oj_id,
                        })
                    }
                    _ => panic!("inputs of MSM must be polynomials"),
                };

                if let Some(new_value) = new_value {
                    *value = new_value;
                }
            }),
            _ => {}
        }
    }

    let cloned_slices_reversed = def_use
        .cloned_slices
        .iter()
        .map(|((sliced_obj, slice), obj)| (*obj, (*sliced_obj, slice.clone())))
        .collect();

    VertexInputs {
        inputs,
        cloned_slices_reversed,
    }
}

pub fn analyze_die_after(
    seq: &[VertexId],
    _devices: &BTreeMap<VertexId, Device>,
    vertex_inputs: &VertexInputs,
) -> ObjectsDieAfter {
    let mut die_after = ObjectsDieAfter::empty();
    for &vid in seq.iter() {
        vertex_inputs.inputs[&vid]
            .iter()
            .map(|input_vv| input_vv.iter())
            .flatten()
            .for_each(|input_value| {
                let device = input_value.device;
                die_after
                    .get_device_mut(device)
                    .insert(input_value.object_id(), vid);

                if device == Device::Gpu {
                    if let Some((sliced_obj, _)) = vertex_inputs
                        .cloned_slices_reversed
                        .get(&input_value.object_id())
                    {
                        die_after
                            .get_device_mut(Device::Cpu)
                            .insert(*sliced_obj, vid);
                    }
                }
            });
    }
    die_after
}
