use crate::driver::HardwareInfo;
use crate::transit::type2::object_analysis::value::{OutputValue, ValueNode};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;

use super::{value, ObjectId};
use crate::transit::type2::{self, VertexId};
use crate::transit::type3::{Device, DeviceSpecific};
use value::{Value, VertexOutput};
use zkpoly_common::{
    arith::Mutability, bijection::Bijection, digraph::internal::SubDigraph, heap::IdAllocator,
    typ::Slice,
};
use zkpoly_runtime::args::RuntimeType;

type VertexInput = value::VertexInput<Value>;

/// Results of analyzing value definitions of each vertex
pub struct DefUse {
    /// Value output of each vertex.
    /// All vertices should appear in this mapping.
    pub(super) value: BTreeMap<VertexId, VertexOutput>,

    /// At which vertex an object is defined.
    /// All allocated objects should appear in this mapping.
    pub(super) def_at: BTreeMap<ObjectId, VertexId>,

    /// On which device an object is defined
    /// All allocated objects should appear in this mapping.
    ///
    /// For cloned slice objects, this is the device where they are first used.
    pub(super) def_on: BTreeMap<ObjectId, Device>,

    /// Value input of each vertex.
    /// All vertices should appear in this mapping.
    pub(super) input: BTreeMap<VertexId, Vec<(VertexId, VertexInput)>>,

    /// Sometimes, we need to clone a slice out of a polynomial.
    /// When a slice is rotated, it is no longer a continuous (and allowing wrapping around) piece of memory.
    /// Some `((cloned, slice), new)` in this map means that `new` is cloned from `cloned` with `slice`.
    pub(super) cloned_slices: Bijection<(ObjectId, Slice), ObjectId>,

    /// The set of objects that should not be deallocated on CPU.
    /// For example, constants and computation graph inputs.
    pub(super) cpu_immortal_objects: BTreeSet<ObjectId>,

    /// The last vertex that uses an object on some device.
    /// All allocated objects except unused ones should appear in this mapping.
    ///
    /// Caution that there can be cases when some outputs of a vertex are used,
    /// but others are not. This vertex is still in the connected component of the graph.
    pub(super) object_dies_after: BTreeMap<ObjectId, DeviceSpecific<Option<VertexId>>>,
}

/// Determine where outputs and inputs of a vertex are, based on where it is executed.
fn decide_device<Rt: RuntimeType>(executed_on: type2::Device, typ: &type2::Typ<Rt>) -> Device {
    match executed_on {
        type2::Device::Cpu => {
            if typ.stack_allocable() {
                Device::Stack
            } else {
                Device::Cpu
            }
        }
        type2::Device::Gpu(i) => Device::Gpu(i),
    }
}

/// Mark that `object_id` is used at vertex `vid` on `device`.
/// Used for obtaining last use of each object on each device,
/// which is then used to decide whether an operation can be carried out inplace without making
/// object clones.
///
/// This analysis does not have to be tight,
/// that is, we can extend liveness longer then actually needed,
/// as it does not affect correctness.
fn mark_use(
    vid: VertexId,
    object_id: ObjectId,
    device: Device,
    hd_info: &HardwareInfo,
    def_on: &mut BTreeMap<ObjectId, Device>,
    used_on: &mut BTreeMap<ObjectId, DeviceSpecific<bool>>,
    cloned_slices: &ClonedSliceRegistry,
    obj_last_use: &mut BTreeMap<ObjectId, DeviceSpecific<Option<VertexId>>>,
) {
    fn extend_lifetime(
        obj_last_use: &mut BTreeMap<ObjectId, DeviceSpecific<Option<VertexId>>>,
        object_id: ObjectId,
        device: Device,
        vid: VertexId,
    ) {
        *obj_last_use
            .entry(object_id)
            .or_default()
            .get_device_mut(device) = Some(vid);

        // Considering that parent memory device has bigger space,
        // it's better that we keep objects alive on parent devices until the object dies
        // on child device,
        // to trade parent memory for transfer costs.
        if let Some(parent_device) = device.parent() {
            extend_lifetime(obj_last_use, object_id, parent_device, vid);
        }
    }

    let is_used_on = |object_id, device| {
        used_on
            .get(&object_id)
            .is_some_and(|places| *places.get_device(device))
    };

    // If
    // - The object is sliced from another object,
    // - And is used for the first time for all devices
    // We follow the proceeding simple rules
    // - If the sliced object is defined on CPU, and it is sliced to GPU,
    //   - If the object has been used on GPU, then we extend its lifetime on GPU
    //   - Otherwise, extend its lifetime on CPU
    // - If the sliced object is defined and used on the same device, we extend its lifetime on it
    // - If the sliced object is defined on GPU, and it is sliced to CPU,
    //   similar to the first case
    if !Device::iter(hd_info.n_gpus()).any(|d| is_used_on(object_id, d)) {
        // We must use `device` for `def_on` of `object_id`, is `object_id` itself can later be clone-sliced,
        // and we don't want its `def_on` to be memory device of the [`VertexNode::RotateIdx`] vertex.
        def_on.insert(object_id, device);

        if let Some((cloned_obj, _)) = cloned_slices.0.get_backward(&object_id) {
            let def_on_device = def_on[&cloned_obj];
            if def_on_device == device {
                extend_lifetime(obj_last_use, *cloned_obj, device, vid);
            } else {
                if is_used_on(*cloned_obj, device) {
                    extend_lifetime(obj_last_use, *cloned_obj, device, vid);
                } else {
                    extend_lifetime(obj_last_use, *cloned_obj, def_on_device, vid);
                }
            }
        }
    }

    // Extend the object's lifetime on the device where it is used
    extend_lifetime(obj_last_use, object_id, device, vid);

    *used_on.entry(object_id).or_default().get_device_mut(device) = true;
}

/// Collects input values of a vertex.
/// This function basically just take output values of used vertices, and alters their device to `memory_device`.
/// It is also responsible for cloning slices out of polynomials for kernels that do not support sliced polynomials.
fn vertex_input_of<'s, Rt: RuntimeType>(
    vid: VertexId,
    v: &type2::Vertex<'s, Rt>,
    memory_device: Device,
    value: &BTreeMap<VertexId, VertexOutput>,
    hd_info: &HardwareInfo,
    cloned_slices: &mut ClonedSliceRegistry,
    object_id_allocator: &mut IdAllocator<ObjectId>,
    def_at: &mut BTreeMap<ObjectId, VertexId>,
    def_on: &mut BTreeMap<ObjectId, Device>,
    used_on: &mut BTreeMap<ObjectId, DeviceSpecific<bool>>,
    obj_last_use: &mut BTreeMap<ObjectId, DeviceSpecific<Option<VertexId>>>,
) -> Vec<(VertexId, VertexInput)> {
    let mutable_used_vids = v.mutable_uses().collect::<Vec<_>>();
    let uses = v.uses().map(|u| {
        (
            u,
            if mutable_used_vids.contains(&u) {
                Mutability::Mut
            } else {
                Mutability::Const
            },
        )
    });

    let inputs = uses
        .map(|(used_vid, mutability)| {
            let iv = match &value[&used_vid] {
                VertexOutput::Single(ov) => {
                    if v.node().no_supports_sliced_inputs() && ov.is_sliced() {
                        // Some vertices' currespounding kernel does not support sliced polynomials,
                        // so we make clones for them
                        let (_, slice) = ov.node().unwrap_poly();
                        let new_object = cloned_slices.register(
                            vid,
                            memory_device,
                            ov.object_id(),
                            slice.clone(),
                            object_id_allocator,
                            def_at,
                        );
                        let new_value = ov
                            .with_object_id(new_object)
                            .with_node(ValueNode::plain_scalar_array(slice.len() as usize))
                            .with_device(memory_device);
                        VertexInput::Single(new_value, mutability)
                    } else {
                        VertexInput::Single(ov.with_device(memory_device), mutability)
                    }
                }
                VertexOutput::Tuple(tuple) => VertexInput::Tuple({
                    tuple
                        .iter()
                        .map(|ov| ov.with_device(memory_device))
                        .collect::<Vec<_>>()
                }),
            };
            (used_vid, iv)
        })
        .collect::<Vec<_>>();

    // Stuff like TupleGet or Rotate does not count as a use
    if !v.is_virtual() {
        for object in inputs
            .iter()
            .map(|(_, vi)| vi.iter().map(|v| v.object_id()))
            .flatten()
        {
            mark_use(
                vid,
                object,
                memory_device,
                hd_info,
                def_on,
                used_on,
                cloned_slices,
                obj_last_use,
            );
        }
    }

    inputs
}

/// Determine [`ValueNode`] for a freshly defined Type2 type.
/// That is, no slice, no rotation.
fn lower_typ<Rt: RuntimeType>(t2typ: &type2::Typ<Rt>) -> ValueNode {
    use type2::typ::template::Typ::*;
    match t2typ {
        Poly((_, deg0)) => ValueNode::plain_scalar_array(*deg0 as usize),
        Scalar => ValueNode::Scalar,
        Point => ValueNode::Point,
        PointBase { log_n } => ValueNode::PointBase {
            len: 2usize.pow(*log_n),
        },
        Transcript => ValueNode::Transcript,
        Tuple(..) => panic!("tuple unexpected"),
        Array(..) => panic!("array unexpected"),
        Any(tid, size) => ValueNode::Any(tid.clone().into(), *size as usize),
        _Phantom(_) => unreachable!(),
    }
}

/// Collect output values of a vertex.
///
/// Most vertices in the graph define an object, while some vertices only modifies the register.
/// For exmaple, by rotating or slicing a polynomial, we only care about the offset and length information
/// stored in the register, but not their data (i.e., the object the register value points to).
///
/// For the former case, only slice information of the value is not inherited from currespounding input value;
///
/// For the latter case, a new value is defined based on `memory_device` and the vertex's Type2 type,
/// plus a newly allocated [`ObjectId`].
/// Also, if some output took memory of a inplace input, the currespounding output value inherits
/// the input's slice information.
/// This behaviour is determined by how kernels are written.
fn vertex_output_of<'s, Rt: RuntimeType>(
    vid: VertexId,
    v: &type2::Vertex<'s, Rt>,
    execution_device: type2::Device,
    memory_device: Device,
    uf_table: &type2::user_function::Table<Rt>,
    input: &[(VertexId, VertexInput)],
    value: &BTreeMap<VertexId, VertexOutput>,
    cloned_slices: &mut ClonedSliceRegistry,
    object_id_allocator: &mut IdAllocator<ObjectId>,
    cpu_immortal_objects: &mut BTreeSet<ObjectId>,
    def_at: &mut BTreeMap<ObjectId, VertexId>,
    def_on: &mut BTreeMap<ObjectId, Device>,
) -> VertexOutput {
    use type2::template::VertexNode::*;
    match v.node() {
        // These vertices only modify registers
        TupleGet(pred, i) | ArrayGet(pred, i) => {
            let pred_value = value[pred].clone();

            match pred_value {
                VertexOutput::Tuple(tuple) => VertexOutput::Single(tuple[*i].clone()),
                VertexOutput::Single(..) => panic!("expected tuple or array"),
            }
        }
        Array(elements) => VertexOutput::Tuple(
            elements
                .iter()
                .map(|e| match value[e].clone() {
                    VertexOutput::Tuple(..) => {
                        panic!("nested array or tuple not supported")
                    }
                    VertexOutput::Single(s) => s,
                })
                .collect(),
        ),
        RotateIdx(pred, delta) => {
            let pred_value: &Value = value[pred].unwrap_single().deref();
            if let Some(rotated_value) = pred_value.rotate(*delta) {
                VertexOutput::Single(OutputValue::non_inplace(rotated_value))
            } else {
                // Rotating a incomplete slice produces non-continuous data, so we need to clone the slice
                let (deg, slice) = pred_value.node().unwrap_poly();
                let new_obj_id = cloned_slices.register(
                    vid,
                    memory_device,
                    pred_value.object_id(),
                    *slice,
                    object_id_allocator,
                    def_at,
                );
                // We don't record `def_on` here, as a [`RotateIdx`] vertex must be later used for it to appear in the graph
                let new_value = pred_value
                    .with_object_id(new_obj_id)
                    .with_node(ValueNode::plain_scalar_array(deg))
                    .rotate(*delta)
                    .unwrap();
                VertexOutput::Single(OutputValue::non_inplace(new_value))
            }
        }
        Slice(pred, begin, end) => {
            let pred_value: &Value = value[pred].unwrap_single().deref();
            let new_value = pred_value.slice(*begin, *end);
            VertexOutput::Single(OutputValue::non_inplace(new_value))
        }
        // Output value of [`Return`] vertex is the desired output value of the computation graph
        AssertEq(a, _, _) | Print(a, _) | Return(a) => {
            let pred_value = value[a].clone();
            pred_value.with_device(Device::Cpu)
        }
        // Following vertices define new objects
        otherwise => {
            assert!(!otherwise.is_virtual());
            let is_tuple = matches!(v.typ(), type2::Typ::Tuple(..) | type2::Typ::Array(..));

            let values = v
                .typ()
                .iter()
                .zip(v.outputs_inplace(uf_table, execution_device))
                .map(|(typ, output_inplace)| {
                    if let Some(inplace_input) = output_inplace {
                        // For inplace input, if it is a slice, then output is also a slice of the same shape
                        let i = input
                            .iter()
                            .position(|(vid, _)| vid == &inplace_input)
                            .unwrap();
                        let (_, input_value) = &input[i];
                        let (input_value, mutability) = input_value.unwrap_single();

                        if *mutability != Mutability::Mut {
                            panic!("inplace input must be mut at {:?}", vid);
                        }

                        let output_value_node_from_typ = lower_typ(typ);
                        if !output_value_node_from_typ.compatible(input_value.node()) {
                            panic!(
                                "incompatible value at {:?}: from typ: {:?}, input value: {:?}",
                                vid,
                                output_value_node_from_typ,
                                input_value.node()
                            );
                        }

                        OutputValue::new(
                            input_value.with_object_id(object_id_allocator.alloc()),
                            Some(i),
                        )
                    } else {
                        let value_node = lower_typ(typ);
                        OutputValue::non_inplace(Value::new(
                            object_id_allocator.alloc(),
                            memory_device,
                            value_node,
                        ))
                    }
                })
                .collect::<Vec<_>>();

            for object_id in values.iter().map(|v| v.object_id()) {
                def_at.insert(object_id, vid);
                def_on.insert(object_id, memory_device);

                if v.node().immortal_on_cpu() {
                    cpu_immortal_objects.insert(object_id);
                }
            }

            if is_tuple {
                VertexOutput::Tuple(values)
            } else {
                VertexOutput::Single(values[0].clone())
            }
        }
    }
}

struct ClonedSliceRegistry(Bijection<(ObjectId, Slice), ObjectId>);

impl ClonedSliceRegistry {
    /// Register that a new object is cloned fro m `cloned_object` with `slice`at vertex `vid`.
    /// In this case, we consider the object to be defined at the vertex where it is first used.
    pub fn register(
        &mut self,
        vid: VertexId,
        device: Device,
        cloned_object: ObjectId,
        slice: Slice,
        object_id_allocator: &mut IdAllocator<ObjectId>,
        def_at: &mut BTreeMap<ObjectId, VertexId>,
    ) -> ObjectId {
        if let Some(new_obj) = self.0.get_forward(&(cloned_object, slice)) {
            *new_obj
        } else {
            let new_obj = object_id_allocator.alloc();
            self.0.insert((cloned_object, slice), new_obj);
            def_at.insert(new_obj, vid);
            new_obj
        }
    }
}

impl DefUse {
    /// Analyze a computation graph for a [`DefUse`].
    pub fn analyze<'s, Rt: RuntimeType>(
        g: &SubDigraph<'_, VertexId, type2::Vertex<'s, Rt>>,
        uf_table: &type2::user_function::Table<Rt>,
        seq: &[VertexId],
        return_vid: VertexId,
        execution_device: impl Fn(VertexId) -> type2::Device,
        hd_info: &HardwareInfo
    ) -> (Self, IdAllocator<ObjectId>) {
        let mut object_id_allocator = IdAllocator::new();

        // These are the results to be obtained
        let mut value: BTreeMap<VertexId, VertexOutput> = BTreeMap::new();
        let mut def_at = BTreeMap::new();
        let mut input = BTreeMap::new();
        let mut cloned_slices = ClonedSliceRegistry(Bijection::new());
        let mut cpu_immortal_objects = BTreeSet::new();
        let mut obj_last_use = BTreeMap::new();
        let mut def_on = BTreeMap::new();

        // Intermediate keeping track of whether some object has been used on some device when
        // visiting current vertex
        let mut used_on = BTreeMap::new();

        // Contract: After this loop, all vertices appear in `value` and `input`
        for &vid in seq.iter() {
            // Invariant: All allocated objects should appear in `def_at`, `def_on`

            let v = g.vertex(vid);
            let execution_device = execution_device(vid);
            let memory_device = decide_device(execution_device, v.typ());

            // Invariant: `used_on` records all uses before this vertex

            // Inputs of the vertex
            let v_inputs = vertex_input_of(
                vid,
                v,
                memory_device,
                &value,
                hd_info,
                &mut cloned_slices,
                &mut object_id_allocator,
                &mut def_at,
                &mut def_on,
                &mut used_on,
                &mut obj_last_use,
            );

            // Output of the vertex
            let vo = vertex_output_of(
                vid,
                v,
                execution_device,
                memory_device,
                uf_table,
                &v_inputs,
                &value,
                &mut cloned_slices,
                &mut object_id_allocator,
                &mut cpu_immortal_objects,
                &mut def_at,
                &mut def_on,
            );

            value.insert(vid, vo);
            input.insert(vid, v_inputs);
        }

        let def_use = Self {
            value,
            def_at,
            input,
            cloned_slices: cloned_slices.0,
            def_on,
            cpu_immortal_objects,
            object_dies_after: obj_last_use,
        };

        def_use.check_contracts(g, seq, return_vid, &object_id_allocator);

        (def_use, object_id_allocator)
    }

    fn check_contracts<'s, Rt: RuntimeType>(
        &self,
        g: &SubDigraph<'_, VertexId, type2::Vertex<'s, Rt>>,
        seq: &[VertexId],
        return_vid: VertexId,
        object_id_allocator: &IdAllocator<ObjectId>,
    ) {
        macro_rules! check_key {
            ($mapping:expr, $key:expr, $name:expr) => {
                if !$mapping.contains_key($key) {
                    panic!("entry not found for key {:?} in {}", $key, $name)
                }
            };
        }

        let returned_objects = self.value[&return_vid]
            .iter()
            .map(|v| v.object_id())
            .collect::<BTreeSet<_>>();

        let vertex_defined_objects = seq
            .iter()
            .filter(|&&vid| !g.vertex(vid).is_virtual())
            .map(|&vid| self.value[&vid].iter().map(|v| v.object_id()))
            .flatten()
            .collect::<BTreeSet<_>>();

        for &vid in seq.iter() {
            check_key!(&self.input, &vid, "input");
            check_key!(&self.value, &vid, "value");
        }

        for object in object_id_allocator.allocated_ids() {
            check_key!(&self.def_at, &object, "def_at");
            check_key!(&self.def_on, &object, "def_on");

            if !(self.object_dies_after.contains_key(&object) || returned_objects.contains(&object))
            {
                panic!(
                    "{:?} is neither found in obj_last_use nor returned_objects",
                    object
                );
            }

            if !(vertex_defined_objects.contains(&object)
                || self.cloned_slices.get_backward(&object).is_some())
            {
                panic!(
                    "{:?} is neither found in defined_objects nor cloned_slices",
                    object
                )
            }
        }
    }

    /// Get the vertex after which an object dies on `device`.
    /// Returns [`None`] if the object is never used on `device`.
    pub fn dies_after(&self, object_id: ObjectId, device: Device) -> Option<VertexId> {
        self.object_dies_after
            .get(&object_id)
            .map(|places| places.get_device(device).clone())
            .flatten()
    }

    pub fn object_is_cloned_slice(&self, object_id: ObjectId) -> Option<(ObjectId, Slice)> {
        self.cloned_slices.get_backward(&object_id).cloned()
    }
}
