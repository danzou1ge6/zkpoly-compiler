use std::collections::{BTreeMap, BTreeSet};

use zkpoly_common::{
    define_usize_id, digraph::internal::SubDigraph, heap::IdAllocator, typ::PolyMeta,
};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::{
    type2,
    type3::{typ::Slice as SliceRange, Device, DeviceSpecific, Size, SmithereenSize},
};

use super::{Typ, VertexId};

define_usize_id!(ObjectId);

#[derive(Debug, Clone)]
pub enum ValueNode {
    // deg here is degree of the object, not the slice
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
    pub fn device(&self) -> Device {
        self.device
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtModifier {
    Before,
    After,
}

/// After which type2 vertex executes the object dies.
/// If an object is not used by any vertex on that device, the correspounding map will not contain the object.
pub type ObjectsDieAfter = DeviceSpecific<BTreeMap<ObjectId, (VertexId, AtModifier)>>;

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
    after: BTreeMap<VertexId, BTreeMap<ObjectId, DeviceCollection>>,
    before: BTreeMap<VertexId, BTreeMap<ObjectId, DeviceCollection>>,
}

impl ObjectsDieAfterReversed {
    pub fn after<'a>(
        &'a self,
        vid: VertexId,
    ) -> Box<dyn Iterator<Item = (ObjectId, &'a DeviceCollection)> + 'a> {
        self.after.get(&vid).map_or_else(
            || Box::new(std::iter::empty()) as Box<dyn Iterator<Item = _>>,
            |x| Box::new(x.iter().map(|(vid, ds)| (*vid, ds))),
        )
    }

    pub fn before<'a>(
        &'a self,
        vid: VertexId,
    ) -> Box<dyn Iterator<Item = (ObjectId, &'a DeviceCollection)> + 'a> {
        self.before.get(&vid).map_or_else(
            || Box::new(std::iter::empty()) as Box<dyn Iterator<Item = _>>,
            |x| Box::new(x.iter().map(|(vid, ds)| (*vid, ds))),
        )
    }
}

impl ObjectsDieAfter {
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (Device, &BTreeMap<ObjectId, (VertexId, AtModifier)>)> {
        [
            (Device::Gpu, &self.gpu),
            (Device::Cpu, &self.cpu),
            (Device::Stack, &self.stack),
        ]
        .into_iter()
    }

    pub fn reversed(&self) -> ObjectsDieAfterReversed {
        let mut after = BTreeMap::new();
        let mut before = BTreeMap::new();

        self.iter().for_each(|(device, mapping)| {
            mapping.iter().for_each(|(oid, (vid, modifier))| {
                let map = match modifier {
                    AtModifier::Before => &mut before,
                    AtModifier::After => &mut after,
                };
                map.entry(*vid)
                    .or_insert_with(BTreeMap::new)
                    .entry(*oid)
                    .or_insert_with(DeviceCollection::empty)
                    .add(device);
            });
        });

        ObjectsDieAfterReversed { after, before }
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
pub struct ObjectsDef {
    pub values: BTreeMap<VertexId, VertexValue>,
    pub defs: BTreeMap<ObjectId, VertexId>,
    pub sizes: BTreeMap<ObjectId, Size>,
    pub cloned_slices: BTreeMap<(ObjectId, PolyMeta), ObjectId>,
    pub immortal_on_cpu: BTreeSet<ObjectId>,
}

pub fn def_at_device(def: &ObjectsDef, uses: &ObjectUse, oid: ObjectId) -> Device {
    if let Some(vid) = def.defs.get(&oid) {
        def.values[&vid]
            .iter()
            .find(|v| v.object_id() == oid)
            .unwrap()
            .device()
    } else {
        let (cloned_obj, _) = uses.cloned_slice_from(oid).unwrap();
        def_at_device(def, uses, cloned_obj)
    }
}

fn rotated_offset(begin: u64, offset: i64, cycle: u64) -> u64 {
    (begin as i64 + offset) as u64 % cycle
}

fn decide_device<Rt: RuntimeType>(executed_on: type2::Device, typ: &Typ<Rt>) -> Device {
    match executed_on {
        type2::Device::PreferGpu => panic!("PreferGpu should have been resolved"),
        type2::Device::Cpu => {
            if typ.stack_allocable() {
                Device::Stack
            } else {
                Device::Cpu
            }
        }
        type2::Device::Gpu => Device::Gpu,
    }
}

pub fn analyze_def<'s, Rt: RuntimeType>(
    g: &SubDigraph<'_, VertexId, type2::Vertex<'s, Rt>>,
    seq: &[VertexId],
    devices: impl Fn(VertexId) -> type2::Device,
) -> (ObjectsDef, IdAllocator<ObjectId>) {
    let mut object_id_allocator = IdAllocator::new();
    let mut values: BTreeMap<VertexId, VertexValue> = BTreeMap::new();
    let mut defs = BTreeMap::new();
    let mut sizes = BTreeMap::new();
    let mut cloned_slices = BTreeMap::new();
    let mut immortal_on_cpu = BTreeSet::new();

    for &vid in seq.iter() {
        let v = g.vertex(vid);

        use super::template::VertexNode::*;
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
                    ValueNode::SlicedPoly { slice, .. } => {
                        // Rotation after slice is not implemented, so we clone the slice to a new polynomial
                        let cloned_obj_id = object_id_allocator.alloc();
                        cloned_slices.insert(
                            (pred_value.object_id(), PolyMeta::Sliced(*slice)),
                            cloned_obj_id,
                        );
                        sizes.insert(
                            cloned_obj_id,
                            Typ::<Rt>::lagrange(slice.len()).size().unwrap_single(),
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
                                *end - *begin,
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
            AssertEq(a, _, _) | Print(a, _) => {
                let pred_value = values[a].clone();
                values.insert(vid, pred_value.with_device(Device::Cpu));
            }
            otherwise => {
                assert!(!otherwise.is_virtual());

                let value = match v.typ() {
                    Typ::Array(typ, len) => {
                        let elements: Vec<_> = (0..*len)
                            .map(|_| {
                                Value::new(
                                    &mut object_id_allocator,
                                    typ.as_ref(),
                                    decide_device(devices(vid), typ.as_ref()),
                                )
                            })
                            .collect();

                        elements.iter().for_each(|elem| {
                            sizes.insert(elem.object_id(), typ.size().unwrap_single());
                        });

                        VertexValue::Tuple(elements)
                    }
                    Typ::Tuple(elements) => VertexValue::Tuple(
                        elements
                            .iter()
                            .map(|e| {
                                let value = Value::new(
                                    &mut object_id_allocator,
                                    e,
                                    decide_device(devices(vid), e),
                                );
                                sizes.insert(value.object_id(), e.size().unwrap_single());

                                value
                            })
                            .collect(),
                    ),
                    otherwise => {
                        let value = Value::new(
                            &mut object_id_allocator,
                            otherwise,
                            decide_device(devices(vid), otherwise),
                        );
                        sizes.insert(value.object_id(), otherwise.size().unwrap_single());
                        VertexValue::Single(value)
                    }
                };

                for oid in value.object_ids() {
                    defs.insert(oid, vid);
                    if otherwise.immortal_on_cpu() {
                        immortal_on_cpu.insert(oid);
                    }
                }

                values.insert(vid, value);
            }
        }
    }

    (
        ObjectsDef {
            values,
            defs,
            sizes: sizes
                .into_iter()
                .map(|(id, s)| (id, Size::Smithereen(SmithereenSize(s))))
                .collect(),
            cloned_slices,
            immortal_on_cpu,
        },
        object_id_allocator,
    )
}

#[derive(Debug, Clone)]
pub struct ObjectUse {
    pub(super) inputs: BTreeMap<VertexId, Vec<VertexValue>>,
    /// a |-> b, s where a is cloned from s slicing into b
    pub(super) cloned_slices_reversed: BTreeMap<ObjectId, (ObjectId, PolyMeta)>,
}

impl ObjectUse {
    pub fn input_of(&self, vid: VertexId) -> impl Iterator<Item = &VertexValue> {
        self.inputs.get(&vid).unwrap().iter()
    }

    pub fn input_values_of(&self, vid: VertexId) -> impl Iterator<Item = &Value> {
        self.input_of(vid).flat_map(|vv| vv.iter())
    }

    pub fn cloned_slice_from(&self, oid: ObjectId) -> Option<(ObjectId, PolyMeta)> {
        self.cloned_slices_reversed.get(&oid).cloned()
    }
}

pub fn plan_vertex_inputs<'s, Rt: RuntimeType>(
    g: &SubDigraph<'_, VertexId, type2::Vertex<'s, Rt>>,
    defs: &mut ObjectsDef,
    devices: impl Fn(VertexId) -> type2::Device,
    obj_id_allocator: &mut IdAllocator<ObjectId>,
) -> ObjectUse {
    let mut inputs: BTreeMap<VertexId, Vec<VertexValue>> = BTreeMap::new();

    let add_cloned_slice =
        |def_use: &mut ObjectsDef,
         obj_id,
         pmeta: PolyMeta,
         before_slice_deg,
         obj_id_allocator: &mut IdAllocator<ObjectId>| {
            let cloned_obj_id = *def_use
                .cloned_slices
                .entry((obj_id, pmeta.clone()))
                .or_insert_with(|| obj_id_allocator.alloc());

            def_use.sizes.insert(
                cloned_obj_id,
                Size::new(
                    Typ::<Rt>::lagrange(pmeta.len(before_slice_deg))
                        .size()
                        .unwrap_single(),
                ),
            );
            cloned_obj_id
        };

    for vid in g.vertices() {
        let v = g.vertex(vid);

        inputs.insert(
            vid,
            v.uses()
                .map(|input_vid| {
                    defs.values[&input_vid]
                        .with_device(decide_device(devices(vid), g.vertex(input_vid).typ()))
                })
                .collect(),
        );
    }

    // - SlicedPoly not allowed on GPU
    for vid in g.vertices() {
        if devices(vid) != type2::Device::Gpu {
            continue;
        }

        inputs.get_mut(&vid).unwrap().iter_mut().for_each(|vv| {
            vv.iter_mut().for_each(|value| {
                let obj_id = value.object_id();

                let new_value = match value.node() {
                    ValueNode::SlicedPoly { slice, deg } => {
                        let cloned_obj_id = add_cloned_slice(
                            defs,
                            obj_id,
                            PolyMeta::Sliced(*slice),
                            *deg,
                            obj_id_allocator,
                        );

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
    for vid in g.vertices() {
        use super::template::VertexNode::*;
        match g.vertex(vid).node() {
            Msm { .. } => inputs.get_mut(&vid).unwrap().iter_mut().for_each(|vv| {
                let value = vv.unwrap_single_mut();
                let obj_id = value.object_id();

                let new_value = match value.node() {
                    ValueNode::SlicedPoly { slice, deg } => {
                        let cloned_oj_id = add_cloned_slice(
                            defs,
                            obj_id,
                            PolyMeta::Sliced(*slice),
                            *deg,
                            obj_id_allocator,
                        );

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
                        let cloned_oj_id = add_cloned_slice(
                            defs,
                            obj_id,
                            PolyMeta::Rotated(*rotation),
                            *deg,
                            obj_id_allocator,
                        );

                        Some(Value {
                            node: ValueNode::Poly {
                                rotation: 0,
                                deg: *deg,
                            },
                            device: Device::Gpu,
                            object_id: cloned_oj_id,
                        })
                    }
                    _ => None,
                };

                if let Some(new_value) = new_value {
                    *value = new_value;
                }
            }),
            _ => {}
        }
    }

    let cloned_slices_reversed = defs
        .cloned_slices
        .iter()
        .map(|((sliced_obj, slice), obj)| (*obj, (*sliced_obj, slice.clone())))
        .collect();

    ObjectUse {
        inputs,
        cloned_slices_reversed,
    }
}

pub fn analyze_die_after<'s, Rt: RuntimeType>(
    g: &SubDigraph<'_, VertexId, type2::Vertex<'s, Rt>>,
    seq: &[VertexId],
    def: &ObjectsDef,
    uses: &ObjectUse,
    vertex_inputs: &ObjectUse,
) -> ObjectsDieAfter {
    let mut die_after = ObjectsDieAfter::empty();
    let mut use_at = |obj_id: ObjectId, vid: VertexId, dev: Device| {
        let def_at_device = def_at_device(def, uses, obj_id);

        use Device::*;
        let devices = match (def_at_device, dev) {
            // The object is defined on GPU, and it's used by some GPU task now.
            // We force that the object lives after current task.
            (Gpu, Gpu) => vec![(Gpu, AtModifier::After)],
            // The object is defined on GPU, and it's used by some CPU task now.
            // - If the object has been used on CPU, it should has already been on CPU,
            //   so we only enforce liveness on CPU,
            //   allowing the object die earliear on GPU.
            // - Otherwise, the object needs to be transferred to CPU now,
            //   so we force that the object lives before current task,
            //   that is, after its transfer to CPU.
            (Gpu, other) => {
                if die_after.get_device(other).contains_key(&obj_id) {
                    vec![(other, AtModifier::After)]
                } else {
                    vec![(other, AtModifier::After), (Gpu, AtModifier::Before)]
                }
            }
            // The object is defined on CPU, and it's used by some GPU task now.
            // The object must live as long as it's needed.
            (other, Gpu) => {
                vec![(Gpu, AtModifier::After), (other, AtModifier::Before)]
            }
            // The object is defined on CPU, and it's used by some CPU task now.
            (_, other) => vec![(other, AtModifier::After)],
        };

        for (dev, am) in devices {
            die_after.get_device_mut(dev).insert(obj_id, (vid, am));
        }
    };

    for &vid in seq.iter() {
        if g.vertex(vid).node().is_virtual() {
            continue;
        }

        vertex_inputs.inputs[&vid]
            .iter()
            .map(|input_vv| input_vv.iter())
            .flatten()
            .for_each(|input_value| {
                let device = input_value.device;
                use_at(input_value.object_id(), vid, device);

                if let Some((sliced_obj, _)) = vertex_inputs
                    .cloned_slices_reversed
                    .get(&input_value.object_id())
                {
                    use_at(*sliced_obj, vid, def_at_device(def, uses, *sliced_obj));
                }
            });
    }

    for obj_id in def.immortal_on_cpu.iter().cloned() {
        let _ = die_after.get_device_mut(Device::Cpu).remove(&obj_id);
    }

    die_after
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsedByEntry {
    vid: VertexId,
    dev: Device,
}

impl PartialOrd for UsedByEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.vid.cmp(&other.vid) {
            std::cmp::Ordering::Equal => {
                if self.dev == other.dev {
                    Some(std::cmp::Ordering::Equal)
                } else {
                    None
                }
            }
            ord => Some(ord),
        }
    }
}

impl Ord for UsedByEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.vid.cmp(&other.vid) {
            std::cmp::Ordering::Equal => {
                if self.dev == other.dev {
                    std::cmp::Ordering::Equal
                } else {
                    panic!("comparing two UsedByEntry at same vertex with different devices")
                }
            }
            ord => ord,
        }
    }
}

#[derive(Debug)]
pub struct ObjectsUsedBy {
    pub(super) used_by: BTreeMap<ObjectId, BTreeSet<UsedByEntry>>,
}

impl ObjectsUsedBy {
    pub fn used_by<'a>(&'a self, oid: ObjectId) -> impl Iterator<Item = VertexId> + 'a {
        self.used_by
            .get(&oid)
            .unwrap()
            .iter()
            .cloned()
            .map(|UsedByEntry { vid, .. }| vid)
    }
}

pub fn analyze_used_by(seq: &[VertexId], vertex_inputs: &ObjectUse) -> ObjectsUsedBy {
    let mut used_by: BTreeMap<ObjectId, BTreeSet<UsedByEntry>> = BTreeMap::new();
    for &vid in seq {
        for v in vertex_inputs.input_values_of(vid) {
            used_by
                .entry(v.object_id())
                .or_default()
                .insert(UsedByEntry {
                    dev: v.device(),
                    vid,
                });
        }
    }

    ObjectsUsedBy { used_by }
}

#[derive(Debug)]
pub struct ObjectsGpuNextUse {
    pub(super) next_use: Vec<BTreeMap<ObjectId, usize>>,
    pub(super) first_use_at: BTreeMap<ObjectId, usize>,
}

use super::memory_planning::Instant;
impl ObjectsGpuNextUse {
    pub fn iter_updates<'a>(
        &'a self,
    ) -> impl Iterator<Item = impl Iterator<Item = (ObjectId, Instant)> + 'a> + 'a {
        self.next_use.iter().map(|next_use| {
            next_use
                .iter()
                .map(|(oid, next_use)| (*oid, Instant(*next_use)))
        })
    }

    pub fn first_use_of(&self, oid: ObjectId) -> Option<Instant> {
        Some(Instant(self.first_use_at.get(&oid).cloned()?))
    }
}

pub fn analyze_gpu_next_use<'s, Rt: RuntimeType>(
    g: &SubDigraph<'_, VertexId, type2::Vertex<'s, Rt>>,
    seq: &[VertexId],
    vertex_inputs: &ObjectUse,
    used_by: &ObjectsUsedBy,
) -> ObjectsGpuNextUse {
    let seq_num = seq
        .iter()
        .cloned()
        .enumerate()
        .fold(BTreeMap::new(), |mut seq_num, (i, vid)| {
            seq_num.insert(vid, i);
            seq_num
        });

    // Collect instant of direct uses of each object
    let mut cpu_use_mark = BTreeSet::new();
    let mut chains: BTreeMap<ObjectId, BTreeSet<usize>> = used_by
        .used_by
        .iter()
        .map(|(oid, vids)| {
            let seq = vids
                .iter()
                .filter_map(|UsedByEntry { vid, dev }| {
                    if g.vertex(*vid).node().is_virtual() {
                        None
                    } else if *dev == Device::Gpu {
                        Some(seq_num[vid])
                    } else {
                        if cpu_use_mark.insert(*oid) {
                            Some(seq_num[vid])
                        } else {
                            None
                        }
                    }
                })
                .collect();
            (*oid, seq)
        })
        .collect();

    // Add instant of slicing-uses of each object
    let mut chains_updates: BTreeMap<ObjectId, Vec<usize>> = BTreeMap::new();
    for (oid, (sliced_obj, _)) in vertex_inputs.cloned_slices_reversed.iter() {
        if let Some(oid_first_use) = chains[oid].first().cloned() {
            chains_updates
                .entry(*sliced_obj)
                .or_default()
                .push(oid_first_use);
        };
    }

    for (oid, updates) in chains_updates {
        chains.entry(oid).or_default().extend(updates);
    }

    let first_use_at = chains
        .iter()
        .filter_map(|(oid, uses)| Some((*oid, *uses.first()?)))
        .collect();

    // Produce the next_use sequence for memory planning
    let mut next_use = vec![BTreeMap::new(); seq.len()];
    chains.into_iter().for_each(|(oid, uses)| {
        uses.iter()
            .zip(uses.iter().skip(1))
            .for_each(|(isntant1, instant2)| {
                next_use[*isntant1].insert(oid, *instant2);
            });
    });

    ObjectsGpuNextUse {
        next_use,
        first_use_at,
    }
}

