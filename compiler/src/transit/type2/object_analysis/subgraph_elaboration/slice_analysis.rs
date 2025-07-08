use super::prelude::*;
use super::Chunker;

#[derive(Debug, Clone)]
pub struct Object {
    id: ObjectId,
    ranges: Vec<Slice>,
    typ: Typ,
}

impl Object {
    fn new(id: ObjectId, typ: Typ) -> Self {
        Self {
            id,
            ranges: Vec::new(),
            typ,
        }
    }

    pub fn id(&self) -> ObjectId {
        self.id
    }

    pub fn node(&self) -> &Typ {
        &self.typ
    }
}

impl Object {
    fn add_range(&mut self, range: Slice) {
        if matches!(&self.typ, Typ::PointBase { .. } | Typ::ScalarArray { .. }) {
            self.ranges.push(range);
        }
    }

    fn set_ranges(&mut self, ranges: Vec<Slice>) {
        if matches!(&self.typ, Typ::PointBase { .. } | Typ::ScalarArray { .. }) {
            self.ranges = ranges;
        }
    }

    pub(super) fn ranges<'a>(&'a self) -> impl Iterator<Item = &'a Slice> + 'a {
        self.ranges.iter()
    }
}

pub type Value = value::Value<Object, ()>;

/// The desired inputed and outputed slices for each vertex in the sliceable subgraph.
/// That is, foreach vertex, the inputs are ensured to produce all slices needed for all outputs,
/// but the actual outputs may contain more slices then outputs really needs.
pub struct DefUse {
    /// The desired output of each vertex
    value: BTreeMap<VertexId, VertexOutput<OutputValue<Value, VertexId>>>,
    /// The desired input of each vertex.
    /// For vertices that do computations (excluding Return, RotateIdx and TupleGet),
    /// We take the smallest slice aligned to chunk boundary that contains the desired outputed slices.
    /// e.g.
    ///          |------|     |------|       desired outputed slices
    ///       |     |      |      |      |   chunk boundaries
    ///       |--------------------------|   input slice
    input: BTreeMap<VertexId, Vec<(VertexId, VertexInput<Value>)>>,
}

impl DefUse {
    fn empty() -> Self {
        Self {
            value: BTreeMap::new(),
            input: BTreeMap::new(),
        }
    }

    pub fn value(&self, v: VertexId) -> &VertexOutput<OutputValue<Value, VertexId>> {
        self.value.get(&v).unwrap()
    }

    pub fn input(&self, v: VertexId) -> &Vec<(VertexId, VertexInput<Value>)> {
        self.input.get(&v).unwrap()
    }

    fn value_mut(&mut self, v: VertexId) -> &mut VertexOutput<OutputValue<Value, VertexId>> {
        self.value.get_mut(&v).unwrap()
    }

    fn input_mut(&mut self, v: VertexId) -> &mut Vec<(VertexId, VertexInput<Value>)> {
        self.input.get_mut(&v).unwrap()
    }

    fn output_of<'s, Rt: RuntimeType>(
        &self,
        v: &Vertex<'s, Rt>,
        memory_device: type3::Device,
        obj_id_allocator: &mut IdAllocator<ObjectId>,
    ) -> VertexOutput<OutputValue<Value, VertexId>> {
        use sliceable_subgraph::template::VertexNode::*;
        match v.node() {
            TupleGet(t, i) => {
                let pred_value = &self.value[t];
                match pred_value {
                    VertexOutput::Tuple(tuple) => VertexOutput::Single(tuple[*i].clone()),
                    VertexOutput::Single(..) => panic!("expected tuple or array"),
                }
            }
            Return(vids) => {
                let pred_values = vids
                    .iter()
                    .map(|vid| {
                        OutputValue::non_inplace(self.value[vid].unwrap_single().deref().clone())
                    })
                    .collect::<Vec<_>>();
                VertexOutput::Tuple(pred_values)
            }
            _ => {
                let is_tuple = matches!(v.typ(), type2::Typ::Tuple(..) | type2::Typ::Array(..));
                let values = v
                    .typ()
                    .iter()
                    .zip(v.node().outputs_inplace())
                    .map(|(typ, output_inplace)| {
                        OutputValue::new(
                            Value::new(
                                Object::new(obj_id_allocator.alloc(), lower_typ(typ).erase_p()),
                                memory_device,
                                (),
                            ),
                            output_inplace,
                        )
                    })
                    .collect::<Vec<_>>();

                if is_tuple {
                    VertexOutput::Tuple(values)
                } else {
                    let mut values = values;
                    VertexOutput::Single(values.pop().unwrap())
                }
            }
        }
    }

    fn input_of<'s, Rt: RuntimeType>(
        &self,
        v: &Vertex<'s, Rt>,
        memory_device: type3::Device,
    ) -> Vec<(VertexId, VertexInput<Value>)> {
        let mutable_used_vids = v.node().mutable_uses().collect::<Vec<_>>();
        let uses = v.node().uses().map(|u| {
            (
                u,
                if mutable_used_vids.contains(&u) {
                    Mutability::Mut
                } else {
                    Mutability::Const
                },
            )
        });

        let with_device = |val: Value| {
            if matches!(memory_device, type3::Device::Stack) {
                panic!("stack is not a memory device at this stage")
            } else if matches!(v.node(), VertexNode::TupleGet(..)) {
                val
            } else {
                val.with_device(memory_device)
            }
        };

        let inputs = uses
            .map(|(used_vid, mutability)| {
                let iv = match &self.value[&used_vid] {
                    VertexOutput::Single(ov) => VertexInput::Single(ov.deref().clone(), mutability),
                    VertexOutput::Tuple(tuple) => VertexInput::Tuple(
                        tuple
                            .iter()
                            .map(|ov| with_device(ov.deref().clone()))
                            .collect::<Vec<_>>(),
                    ),
                };
                (used_vid, iv)
            })
            .collect::<Vec<_>>();

        inputs
    }

    // Forward propagation from inputs to return vertex to acquire object information for each vertex
    pub fn infer_objects<'s, Rt: RuntimeType>(
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        execution_device: impl Fn(VertexId) -> Device,
        constant_devices: &Heap<ConstantId, type3::Device>,
        obj_id_allocator: &mut IdAllocator<ObjectId>,
    ) -> Self {
        let mut def_use = Self::empty();

        for (vid, v) in cg.g.topology_sort() {
            let memory_device = match v.node() {
                VertexNode::Inner(SliceableNode::Constant(cid)) => constant_devices[*cid],
                _ => match execution_device(vid) {
                    Device::Gpu(i) => type3::Device::Gpu(i),
                    Device::Cpu => type3::Device::Cpu,
                },
            };

            let input = def_use.input_of(v, memory_device);
            let output = def_use.output_of(v, memory_device, obj_id_allocator);

            def_use.input.insert(vid, input);
            def_use.value.insert(vid, output);
        }

        def_use
    }

    // Back propagation from return vertex to acquire needed slice for each vertex
    pub fn infer_slice<'s, Rt: RuntimeType>(
        &mut self,
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        chunker: &Chunker,
    ) {
        for (vid, v) in cg.g.topology_sort_inv() {
            if let VertexNode::Return(..) = v.node() {
                self.value_mut(vid)
                    .iter_mut()
                    .for_each(|vo| vo.object_mut().add_range(chunker.output_range()))
            }

            // Propagate needed slices from output to inputs
            match v.node() {
                // Rotations should only have one input and one output
                VertexNode::Inner(SliceableNode::RotateIdx(_, delta)) => {
                    let output_slices = &self.value[&vid].unwrap_single().deref().object().ranges;
                    let (_, deg) = v.typ().unwrap_poly();
                    let rotated_slices = output_slices
                        .iter()
                        .map(|s| s.rotated(-*delta as i64, *deg))
                        .collect::<Vec<_>>();
                    self.input_mut(vid).iter_mut().for_each(|(_, iv)| {
                        iv.iter_mut()
                            .for_each(|v| v.object_mut().set_ranges(rotated_slices.clone()))
                    });
                }
                // For inner vertices that take slices as inputs and outputs,
                // we need each input to include all slices for every output
                VertexNode::Inner(..) => {
                    let output_slices = self.value[&vid]
                        .iter()
                        .map(|ov| ov.deref().object().ranges.iter())
                        .flatten()
                        .cloned();

                    let out_slice = chunker.including_slice(output_slices);

                    self.input_mut(vid).iter_mut().for_each(|(_, iv)| {
                        iv.iter_mut()
                            .for_each(|v| v.object_mut().set_ranges(vec![out_slice]))
                    });
                }
                // Return is actually tuple packing
                VertexNode::Return(..) => {
                    let output_values = self.value[&vid]
                        .unwrap_tuple()
                        .iter()
                        .map(|ov| ov.deref().clone())
                        .collect::<Vec<_>>();
                    self.input_mut(vid)
                        .iter_mut()
                        .zip(output_values.into_iter())
                        .for_each(|((_, iv), ov)| {
                            *iv.unwrap_single_mut().0.deref_mut() = ov;
                        })
                }
                // For vertices that outputs partial scalar/point results, we enforce their inputs to include the output range
                // [0, chunk_size)
                VertexNode::Last(..) => self.input_mut(vid).iter_mut().for_each(|(_, iv)| {
                    iv.iter_mut()
                        .for_each(|v| v.object_mut().add_range(chunker.output_range()))
                }),
                // For tuple gets, we propagate slice requirement to the currespounding element in the input tuple
                VertexNode::TupleGet(_, i) => {
                    let output_value = self.value[&vid].unwrap_single().deref().clone();
                    let iv = self.input_mut(vid)[0].1.iter_mut().nth(*i).unwrap();
                    *iv = output_value;
                }
                VertexNode::Input(..) => todo!(),
            }

            self.input[&vid]
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|(uid, vi)| {
                    self.value_mut(uid)
                        .iter_mut()
                        .zip(vi.iter())
                        .for_each(|(u_out, v_in)| {
                            v_in.object()
                                .ranges
                                .iter()
                                .for_each(|s| u_out.object_mut().add_range(s.clone()))
                        });
                });
        }
    }
}
