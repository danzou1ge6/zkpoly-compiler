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
    pub fn can_be_sliced(&self) -> bool {
        matches!(&self.typ, Typ::PointBase { .. } | Typ::ScalarArray { .. })
    }
    fn add_range(&mut self, range: Slice) {
        if self.can_be_sliced() {
            self.ranges.push(range);
        }
    }

    fn set_ranges(&mut self, ranges: Vec<Slice>) {
        if self.can_be_sliced() {
            self.ranges = ranges;
        }
    }

    pub(super) fn ranges<'a>(&'a self) -> impl Iterator<Item = &'a Slice> + 'a {
        self.ranges.iter()
    }

    fn with_range(&self, range: Slice) -> Self {
        Self {
            id: self.id,
            ranges: vec![range],
            typ: self.typ.clone(),
        }
    }
}

pub type Atom = value::Atom<Object, ()>;
pub type Value = value::Tree<Atom>;
pub type OutputValue = value::OutputValue<Atom, VertexId>;
pub type InputValue = value::InputValue<Atom>;

pub struct Concat {
    pub former: Object,
    pub latter: Object,
}

/// Keep tracks of the intermediates between prologue and body,
/// and between body and body.
pub struct Intermediates {
    /// After execution of some vertex in prologue, we need to slice some objects out of its outputs,
    /// and output the produced object as an intermediate
    prologue_slice_out: BTreeMap<VertexId, Vec<Object>>,
    /// After execution of some vertex in body, we need to concat two slices to get output that can be used by successive vertices.
    body_concat_in: BTreeMap<VertexId, Vec<Concat>>,
    /// Same as `prologue_slice_out`
    body_slice_out: BTreeMap<VertexId, Vec<Object>>,
}

impl Intermediates {
    fn add_prologue_slice_out(&mut self, vid: VertexId, obj: Object) {
        self.prologue_slice_out.entry(vid).or_default().push(obj);
    }

    fn add_body_concat_in(&mut self, vid: VertexId, concat: Concat) {
        self.body_concat_in.entry(vid).or_default().push(concat)
    }

    fn add_body_slice_out(&mut self, vid: VertexId, obj: Object) {
        self.body_slice_out.entry(vid).or_default().push(obj)
    }
}

trait Planner {
    fn calculated_range<'a>(
        &mut self,
        vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        objects: impl Iterator<Item = &'a Object>,
    ) -> Slice;
    fn inputed_range<'a>(
        &mut self,
        vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        objects: impl Iterator<Item = &'a Object>,
    ) -> Slice;
    fn output_range(&self) -> Slice;
}

pub struct ProloguePlanner<'a> {
    chunker: &'a Chunker,
}

impl<'b> Planner for ProloguePlanner<'b> {
    fn calculated_range<'a>(
        &mut self,
        _vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        _objects: impl Iterator<Item = &'a Object>,
    ) -> Slice {
        self.chunker.including_slice(needed_range)
    }

    fn inputed_range<'a>(
        &mut self,
        _vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        _objects: impl Iterator<Item = &'a Object>,
    ) -> Slice {
        self.chunker.including_chunks(needed_range)
    }

    fn output_range(&self) -> Slice {
        self.chunker.output_range()
    }
}

/// For a vertex with successors need range x,
/// we determine the range to calculate and intermediates in the following manner:
///   |--------------------|    x
///                |-------|    the calculated range is exactly the last chunk_size
///   |------------|            we need this range taken from last iteration
///           |------------|    and this range is fed to next iteration
pub struct BodyPlanner<'a> {
    chunker: &'a Chunker,
    itm: Intermediates,
}

impl<'b> BodyPlanner<'b> {
    fn plan_itm_for_desired_range<'a>(
        &mut self,
        vid: VertexId,
        including_range: Slice,
        objects: impl Iterator<Item = &'a Object>,
    ) -> Slice {
        let itm_in_range = self.chunker.new_slice(
            including_range.begin(),
            including_range.len() - self.chunker.chunk_size(),
        );
        let itm_out_range = self
            .chunker
            .rotate(&itm_in_range, self.chunker.chunk_size() as i64);
        let calculated_range = self
            .chunker
            .new_slice(itm_in_range.end(), self.chunker.chunk_size());

        objects.for_each(|obj| {
            self.itm
                .add_prologue_slice_out(vid, obj.with_range(itm_in_range));
            self.itm
                .add_body_slice_out(vid, obj.with_range(itm_out_range));
            self.itm.add_body_concat_in(
                vid,
                Concat {
                    former: obj.with_range(itm_in_range),
                    latter: obj.with_range(calculated_range),
                },
            );
        });

        calculated_range
    }
}

impl<'b> Planner for BodyPlanner<'b> {
    fn calculated_range<'a>(
        &mut self,
        vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        objects: impl Iterator<Item = &'a Object>,
    ) -> Slice {
        let including_range = self.chunker.including_slice(needed_range);
        self.plan_itm_for_desired_range(vid, including_range, objects)
    }

    fn inputed_range<'a>(
        &mut self,
        vid: VertexId,
        needed_range: impl Iterator<Item = Slice>,
        objects: impl Iterator<Item = &'a Object>,
    ) -> Slice {
        let including_range = self.chunker.including_chunks(needed_range);
        self.plan_itm_for_desired_range(vid, including_range, objects)
    }

    fn output_range(&self) -> Slice {
        self.chunker.rotate(
            &self.chunker.output_range(),
            self.chunker.chunk_size() as i64,
        )
    }
}

/// The desired inputed and outputed slices for each vertex in the sliceable subgraph.
/// That is, foreach vertex, the inputs are ensured to produce all slices needed for all outputs,
/// but the actual outputs may contain more slices then outputs really needs.
pub struct DefUse {
    /// The desired output of each vertex
    value: BTreeMap<VertexId, OutputValue>,
    /// The desired input of each vertex.
    /// For vertices that do computations (excluding Return, RotateIdx and TupleGet),
    /// We take the smallest slice aligned to chunk boundary that contains the desired outputed slices.
    /// e.g.
    ///          |------|     |------|       desired outputed slices
    ///       |     |      |      |      |   chunk boundaries
    ///       |--------------------------|   input slice
    input: BTreeMap<VertexId, Vec<(VertexId, InputValue)>>,
}

impl DefUse {
    fn empty() -> Self {
        Self {
            value: BTreeMap::new(),
            input: BTreeMap::new(),
        }
    }

    pub fn value(&self, v: VertexId) -> &OutputValue {
        self.value.get(&v).unwrap()
    }

    pub fn input(&self, v: VertexId) -> &Vec<(VertexId, InputValue)> {
        self.input.get(&v).unwrap()
    }

    fn value_mut(&mut self, v: VertexId) -> &mut OutputValue {
        self.value.get_mut(&v).unwrap()
    }

    fn input_mut(&mut self, v: VertexId) -> &mut Vec<(VertexId, InputValue)> {
        self.input.get_mut(&v).unwrap()
    }

    fn output_of<'s, Rt: RuntimeType>(
        &self,
        v: &Vertex<'s, Rt>,
        memory_device: type3::Device,
        obj_id_allocator: &mut IdAllocator<ObjectId>,
    ) -> OutputValue {
        use sliceable_subgraph::template::VertexNode::*;
        match v.node() {
            TupleGet(t, i) => {
                let pred_value = &self.value[t];
                match pred_value {
                    OutputValue::Tuple(tuple) => tuple[*i].clone(),
                    OutputValue::Single(..) => panic!("expected tuple or array"),
                }
            }
            Return(vids) => {
                let pred_values = vids
                    .iter()
                    .map(|vid| {
                        self.value[vid]
                            .clone()
                            .map_ref(|osv| osv.with_non_inplace())
                    })
                    .collect::<Vec<_>>();
                OutputValue::Tuple(pred_values)
            }
            _ => lower_typ(v.typ(), &mut |vnode| {
                value::OutputT::non_inplace(Atom::new(
                    Object::new(obj_id_allocator.alloc(), vnode.erase_p()),
                    memory_device,
                    (),
                ))
            }),
        }
    }

    fn input_of<'s, Rt: RuntimeType>(
        &self,
        v: &Vertex<'s, Rt>,
        memory_device: type3::Device,
    ) -> Vec<(VertexId, InputValue)> {
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

        let with_device = |val: Atom| {
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
                let iv = self.value[&used_vid].map_ref(|osv| {
                    value::InputT::new(with_device(osv.deref().clone()), mutability)
                });
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
            let mut output = def_use.output_of(v, memory_device, obj_id_allocator);

            output
                .iter1_mut()
                .zip(v.node().outputs_inplace())
                .for_each(|(ov, inplace_of)| {
                    if let Some(inplace_of) = inplace_of {
                        *ov.unwrap_leaf_mut().inplace_of_mut() = Some(inplace_of);
                    }
                });

            def_use.input.insert(vid, input);
            def_use.value.insert(vid, output);
        }

        def_use
    }

    // Back propagation from return vertex to acquire needed slice for each vertex
    pub fn infer_slice<'s, Rt: RuntimeType>(
        &mut self,
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        planner: &mut impl Planner,
    ) {
        todo!("consider not to propagate slice information for scalars");
        for (vid, v) in cg.g.topology_sort_inv() {
            // Return vertex's output range is set to first chunk
            if let VertexNode::Return(..) = v.node() {
                self.value_mut(vid)
                    .iter_mut()
                    .for_each(|vo| vo.object_mut().add_range(planner.output_range()))
            }

            // Propagate needed slices from output to inputs.
            match v.node() {
                // Rotations should only have one input and one output
                VertexNode::Inner(SliceableNode::RotateIdx(_, delta)) => {
                    let output_slices = &self.value[&vid].unwrap_leaf().deref().object().ranges;
                    let (_, deg) = v.typ().unwrap_poly();
                    let rotated_slices = output_slices
                        .iter()
                        .map(|s| s.rotated(-*delta as i64, *deg))
                        .collect::<Vec<_>>();
                    assert!(self.input(vid).len() == 1);
                    self.input_mut(vid).iter_mut().for_each(|(_, iv)| {
                        iv.unwrap_leaf_mut()
                            .object_mut()
                            .set_ranges(rotated_slices.clone());
                    });
                }
                // For inner vertices that take slices as inputs and outputs,
                // we need each input to include all slices for every output
                VertexNode::Inner(..) | VertexNode::Input(..) => {
                    let output_slices = self.value[&vid]
                        .iter()
                        .map(|ov| ov.deref().object().ranges.iter())
                        .flatten()
                        .cloned();

                    // Some part of the desired output can be taken from last iteration.
                    // `planner` is responsible for deciding and keeping track of that.
                    let out_slice = match v.node() {
                        VertexNode::Input(..) => planner.inputed_range(
                            vid,
                            output_slices,
                            self.value[&vid].iter().map(|ov| ov.object()),
                        ),
                        VertexNode::Inner(..) => planner.calculated_range(
                            vid,
                            output_slices,
                            self.value[&vid].iter().map(|ov| ov.object()),
                        ),
                        _ => unreachable!(),
                    };

                    self.value_mut(vid).iter_mut().for_each(|ov| {
                        ov.object_mut().set_ranges(vec![out_slice]);
                    });

                    if let VertexNode::Inner(..) = v.node() {
                        self.input_mut(vid)
                            .iter_mut()
                            .map(|(_, iv)| iv.iter_mut().map(|iv| iv.deref_mut()))
                            .flatten()
                            .for_each(|v| v.object_mut().set_ranges(vec![out_slice]));
                    }
                }
                // Return is actually tuple packing
                VertexNode::Return(..) => {
                    let output_values = self.value[&vid]
                        .iter1()
                        .map(|ov| ov.map_ref(|osv| value::InputT::immutable(osv.deref().clone())))
                        .collect::<Vec<_>>();
                    self.input_mut(vid)
                        .iter_mut()
                        .zip(output_values.into_iter())
                        .for_each(|((_, iv), ov)| *iv = ov)
                }
                // For vertices that outputs partial scalar/point results, we enforce their inputs to include the output range
                // [0, chunk_size)
                VertexNode::Last(..) => self.input_mut(vid).iter_mut().for_each(|(_, iv)| {
                    iv.iter_mut()
                        .for_each(|v| v.object_mut().set_ranges(vec![planner.output_range()]))
                }),
                // For tuple gets, we propagate slice requirement to the currespounding element in the input tuple
                VertexNode::TupleGet(_, i) => {
                    let output_value = self.value[&vid]
                        .map_ref(|osv| value::InputT::immutable(osv.deref().clone()));
                    let iv = self.input_mut(vid)[0].1.iter1_mut().nth(*i).unwrap();
                    *iv = output_value;
                }
            }

            // Propagate neede slices from inputs to outputs
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
