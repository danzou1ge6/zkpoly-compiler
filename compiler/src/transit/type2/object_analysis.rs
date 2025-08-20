pub mod value;

mod prelude {
    pub use std::{
        collections::{BTreeMap, BTreeSet},
        ops::Deref,
    };

    pub use super::value::ValueNode;

    pub use super::cg_def_use::{DefUse, Die};
    pub use zkpoly_common::{
        define_usize_id,
        digraph::internal::SubDigraph,
        heap::{Heap, IdAllocator, UsizeId},
        load_dynamic::Libs,
        typ::Slice,
    };
    pub use zkpoly_runtime::args::RuntimeType;

    pub use crate::transit::{
        type2::{self},
        type3::Device,
        SourceInfo,
    };

    pub use super::super::{Cg, VertexId};

    pub use super::template::{ChunkedOp, MemoryOp, Type2Op};
}

use prelude::*;
mod subgraph_elaboration;

pub fn decide_device_for_non_constant(execution_device: type2::Device) -> Device {
    match execution_device {
        type2::Device::Cpu => Device::Cpu,
        type2::Device::Gpu(i) => Device::Gpu(i),
    }
}

/// Determine where outputs and inputs of a vertex are, based on where it is executed.
fn decide_device<'s, Rt: RuntimeType>(
    v: &type2::Vertex<'s, Rt>,
    execution_device: type2::Device,
    constants_device: &Heap<type2::ConstantId, Device>,
) -> Device {
    use type2::template::SliceableNode::*;
    use type2::template::VertexNode::*;
    match v.node() {
        UnsliceableConstant(cid) | Sliceable(Constant(cid)) => constants_device[*cid],
        _ => decide_device_for_non_constant(execution_device),
    }
}

/// Determine [`ValueNode`] for a freshly defined Type2 type.
/// That is, no slice, no rotation.
pub fn lower_typ<S, Rt: RuntimeType>(
    t2typ: &type2::Typ<Rt>,
    leaf_constructor: &mut impl FnMut(ValueNode) -> S,
) -> value::Tree<S>
where
    S: Clone,
{
    use type2::typ::template::Typ::*;
    let f = leaf_constructor;
    match t2typ {
        Poly((_, deg0)) => value::Tree::Single(f(ValueNode::plain_scalar_array(*deg0 as usize))),
        Scalar => value::Tree::Single(f(ValueNode::Scalar)),
        Point => value::Tree::Single(f(ValueNode::Point)),
        PointBase { log_n } => value::Tree::Single(f(ValueNode::PointBase {
            len: 2usize.pow(*log_n),
        })),
        Transcript => value::Tree::Single(f(ValueNode::Transcript)),
        Tuple(vs) => {
            let vs = vs.iter().map(|x| lower_typ(x, f)).collect();
            value::Tree::Tuple(vs)
        }
        Array(t, len) => {
            let t = lower_typ(t, f);
            value::Tree::Tuple((0..*len).map(|_| t.clone()).collect())
        }
        Any(tid, size) => {
            value::Tree::Single(f(ValueNode::Any(tid.clone().into(), *size as usize)))
        }
        _Phantom(_) => unreachable!(),
    }
}

pub mod template {

    use super::{define_usize_id, value, Device, Heap, Slice, SourceInfo, VertexId};
    use crate::transit::type2;

    pub type Atom<O> = value::Atom<O, value::ValueNode>;
    pub type Value<O> = value::Tree<Atom<O>>;
    pub type OutputAtom<O> = value::OutputT<Atom<O>, Atom<O>>;
    pub type InputAtom<O> = value::InputT<Atom<O>>;

    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct ResidentalT<V, P>(V, P);

    impl<V, P> ResidentalT<V, P> {
        pub fn new(value: V, pointer: P) -> Self {
            ResidentalT(value, pointer)
        }

        pub fn pointer(&self) -> &P {
            &self.1
        }

        pub fn with_pointer(self, pointer: P) -> Self {
            ResidentalT(self.0, pointer)
        }

        pub fn value(&self) -> &V {
            &self.0
        }
    }

    impl<V, P> std::ops::Deref for ResidentalT<V, P> {
        type Target = V;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<V, P> From<V> for ResidentalT<V, P>
    where
        P: Default,
    {
        fn from(value: V) -> Self {
            ResidentalT(value, P::default())
        }
    }

    impl<V, P> ResidentalT<V, Option<P>> {
        pub fn assume_pointed(self) -> ResidentalT<V, P> {
            ResidentalT(self.0, self.1.unwrap())
        }
    }

    pub type ResidentalAtom<O, P> = ResidentalT<Atom<O>, P>;
    pub type ResidentalValue<O, P> = value::Tree<ResidentalAtom<O, P>>;

    pub type InputRv<O, P> = value::Tree<value::InputT<ResidentalAtom<O, P>>>;
    pub type OutputRv<O, P> =
        value::Tree<value::OutputT<ResidentalAtom<O, P>, ResidentalAtom<O, P>>>;

    pub type VertexNode<O, P> = type2::no_subgraph::alt_label::VertexNode<InputRv<O, P>>;
    pub type ChunkedNode<O, P> =
        type2::sliceable_subgraph::alt_label::VertexNode<InputRv<O, P>, InputRv<O, P>>;

    #[derive(Debug, Clone)]
    pub struct Type2Op<'s, O, P> {
        vid: VertexId,
        output: OutputRv<O, P>,
        node: VertexNode<O, P>,
        temporaries: Vec<ResidentalAtom<O, P>>,
        src: SourceInfo<'s>,
    }

    #[derive(Debug, Clone)]
    pub struct ChunkedOp<'s, O, P> {
        vid: type2::sliceable_subgraph::VertexId,
        output: OutputRv<O, P>,
        node: ChunkedNode<O, P>,
        src: SourceInfo<'s>,
    }

    impl<'s, O, P> ChunkedOp<'s, O, P> {
        pub fn new(
            vid: type2::sliceable_subgraph::VertexId,
            output: OutputRv<O, P>,
            node: ChunkedNode<O, P>,
            src: SourceInfo<'s>,
        ) -> Self {
            Self {
                vid,
                output,
                node,
                src,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub enum MemoryOp<O, P> {
        /// [`Clone(o1, d, o2, s)`]  means o1, d <- Clone o2, s,
        /// i.e., Clone a (potential) slice `s` of object `o2` and store it to a new object `o1` on device `d`.
        Clone(O, Device, O, Option<Slice>),

        /// [`Clone2(o1, d, {s_i})`] means o1, d <- Clone {o2_i}, s_i,
        /// i.e., Clone slices `s_i` and concatenate them to obtain `o1` on device `d`
        Clone2(O, Device, Vec<(O, Slice)>),

        /// Like [`Eject`], [`Reclaim`] and [`Tranfer`], but here transfers are carried out
        /// by units of objects, and slicing is supported.
        EjectObject(Atom<O>, ResidentalAtom<O, P>),
        ReclaimObject(ResidentalAtom<O, P>, Atom<O>),
        TransferObject(ResidentalAtom<O, P>, ResidentalAtom<O, P>),

        /// [`Eject(d1, t, d2, p)`] means d1 <- Eject t, d2, p,
        /// i.e., eject object `t` from device `d2` at memory location `p` to device `d1`.
        ///
        /// This operation is emitted when we are planninng memory of a device but not its parent.
        /// At this point, we only know where the object resides on `d2` but not on `d1`.
        /// Later after we has planned memory of `d1`, this operation should be rewritten to a [`Transfer`]
        Eject(Device, O, Device, P),

        /// [`Reclaim(d1, p, t, d2`] means d1, p <- Reclaim t, d2,
        /// i.e., reclaim object `t` from device `d2` to device `t1` at memory location `p`.
        /// This operation serves a similar purpose as [`Eject`], but in the opposite direction.
        ///
        /// However, when really planning memory of `d2`, if the token is not found on `d2`,
        /// it must have been poped to `d2`'s parent device.
        /// In this case, we postpone reclaim to planning of `d2`'s parent device.
        Reclaim(Device, P, O, Device),

        /// [`Transfer(d1, p1, t, d2, p2)`] means d1, p <- Transfer t, d2, p,
        /// i.e. transfer object `t` from device `d2` at memory location `p2` to device `d1` at memory location `p1`.
        Transfer(Device, P, O, Device, P),

        /// Allocate some object on device
        Allocate(O, Device, P),

        /// Deallocate some object on device
        Deallocate(O, Device, P),
    }

    /// A memory relevant operation involved in executing a Type2 computation graph.
    ///
    /// Type parameter [`T`] is the smallest unit of memory transferring, Token.
    /// [`P`] means Pointer, by which the object is accessed from memory.
    ///
    /// Only operations with a specific pointer for each operand can be lowered to Type3.
    #[derive(Debug, Clone)]
    pub enum Operation<'s, O, P> {
        /// A Type2 vertex translated to value inputs/outputs.
        ///
        /// Depending on current device of memory planning, some of the values' pointers
        /// may be unknown.
        ///
        /// All temporary spaces belong to the same object.
        Type2(Type2Op<'s, O, Option<P>>),

        /// A subgraph of Type2 CG that is executed for each chunk of output polynomials
        Subgraph {
            uses: Vec<ResidentalValue<O, Option<P>>>,
            // TODO some SgOperation's here and ?
            src: SourceInfo<'s>,
        },

        M(MemoryOp<O, P>),
    }

    impl<'s, O, P> From<MemoryOp<O, P>> for Operation<'s, O, P> {
        fn from(value: MemoryOp<O, P>) -> Self {
            Self::M(value)
        }
    }

    impl<'s, O, P> From<Type2Op<'s, O, Option<P>>> for Operation<'s, O, P> {
        fn from(value: Type2Op<'s, O, Option<P>>) -> Self {
            Self::Type2(value)
        }
    }

    #[derive(Debug, Clone)]
    pub enum SubgraphOperation<'s, O, P> {
        /// A Type2 vertex translated to value inputs/outputs.
        ///
        /// Depending on current device of memory planning, some of the values' pointers
        /// may be unknown.
        ///
        /// All temporary spaces belong to the same object.
        Type2(ChunkedOp<'s, O, Option<P>>),

        M(MemoryOp<O, P>),
    }

    impl<'s, O, P> From<MemoryOp<O, P>> for SubgraphOperation<'s, O, P> {
        fn from(value: MemoryOp<O, P>) -> Self {
            Self::M(value)
        }
    }

    impl<'s, O, P> From<ChunkedOp<'s, O, Option<P>>> for SubgraphOperation<'s, O, P> {
        fn from(value: ChunkedOp<'s, O, Option<P>>) -> Self {
            Self::Type2(value)
        }
    }

    impl<'s, T, P> Operation<'s, T, P>
    where
        P: Clone + 'static,
        Object: for<'a> From<&'a T>,
    {
        /// Get all objects used by this operation, and on which device they are used.
        pub fn object_uses<'a>(&'a self) -> Box<dyn Iterator<Item = (Object, Device)> + 'a> {
            use Operation::*;
            match self {
                Type2(_, _, node, temps, _) => Box::new(
                    node.uses_ref()
                        .map(|vi| vi.iter().map(|v| (v.object().clone(), v.device())))
                        .flatten()
                        .chain(temps.iter().map(|v| (v.object().clone(), v.device()))),
                ),
                Subgraph { uses, .. } => {
                    Box::new(uses.iter().map(|rv| (rv.object().clone(), rv.device())))
                }
                Chunked(_, node, _) => Box::new(
                    node.uses_ref()
                        .map(|vi| vi.iter().map(|v| (v.object().clone(), v.device())))
                        .flatten(),
                ),
                // Default the use to the device where the new object is required to.
                // This does not have to be accurate, as memory planner will choose any device that has the object.
                Clone(_, d, u, _) => Box::new([(u.clone(), *d)].into_iter()),
                Clone2(_, d, srcs) => Box::new(srcs.iter().map(|(o, _)| (o.clone(), *d))),
                EjectObject(_, rv) | TransferObject(_, rv) => {
                    Box::new([(rv.object().clone(), rv.device())].into_iter())
                }
                ReclaimObject(_, v) => Box::new([(v.object().clone(), v.device())].into_iter()),
                Eject(_, t, d, _) | Reclaim(_, _, t, d) | Transfer(_, _, t, d, _) => {
                    Box::new([(t.into(), *d)].into_iter())
                }
                Allocate(_, _, _) | Deallocate(_, _, _) => Box::new(std::iter::empty()),
            }
        }

        /// Get all objects defined by this operation, and on which device they are used
        pub fn object_defs<'a>(&'a self) -> Box<dyn Iterator<Item = (Object, Device)> + 'a> {
            use Operation::*;
            match self {
                Type2(_, outputs, ..) => Box::new(
                    outputs
                        .iter()
                        .map(|(v, _)| (v.object().clone(), v.device())),
                ),
                Subgraph { .. } => todo!(),
                Chunked(outputs, _, _) => Box::new(
                    outputs
                        .iter()
                        .map(|(v, _)| (v.object().clone(), v.device())),
                ),
                Clone(o, d, _, _) => Box::new([(o.clone(), *d)].into_iter()),
                Clone2(o, d, _) => Box::new([(o.clone(), *d)].into_iter()),
                ReclaimObject(rv, _) | TransferObject(rv, _) => {
                    Box::new([(rv.object().clone(), rv.device())].into_iter())
                }
                EjectObject(v, _) => Box::new([(v.object().clone(), v.device())].into_iter()),
                Eject(d, t, ..) | Reclaim(d, _, t, ..) | Transfer(d, _, t, ..) => {
                    Box::new([(t.into(), *d)].into_iter())
                }
                Allocate(_, _, _) | Deallocate(_, _, _) => Box::new(std::iter::empty()),
            }
        }

        pub fn ready_for_type3(&self) -> bool {
            use Operation::*;
            match self {
                Type2(_, outputs, node, temps, _src) => {
                    outputs.iter().all(|(rv, _)| rv.pointer().is_some())
                        && node
                            .uses_ref()
                            .all(|vi| vi.iter().all(|v| v.pointer().is_some()))
                        && temps.iter().all(|rv| rv.pointer().is_some())
                }
                TransferObject(..) => true,
                Transfer(..) => true,
                Allocate(..) | Deallocate(..) => true,
                _ => false,
            }
        }
    }

    define_usize_id!(Index);

    impl Index {
        pub fn inf() -> Self {
            Index::from(usize::MAX)
        }
    }

    /// A sequence of [`Operation`]
    #[derive(Debug, Clone)]
    pub struct OperationSeq<R>(Heap<Index, R>);

    impl<R> OperationSeq<R> {
        pub fn empty() -> Self {
            OperationSeq(Heap::new())
        }

        pub fn emit(&mut self, op: impl Into<R>) {
            self.0.push(op.into());
        }

        pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Index, &'a R)> + 'a {
            self.0.iter_with_id()
        }

        pub fn into_iter(self) -> impl Iterator<Item = (Index, R)> {
            self.0.into_iter_with_id()
        }
    }
}

pub use template::{Index, Operation, OperationSeq};

pub mod cg_def_use;
pub mod size;

/// Given an iterator of sizes of buffers, and align the buffer must obey,
/// returns the total size of the merged buffer and the (offset, length)'s of each buffer slice,
/// where the length may be longer than original size.
fn merge_temporary_buffer(sizes: impl Iterator<Item = u64>, align: u64) -> (u64, Vec<Slice>) {
    let mut offset = 0;
    let mut slices = Vec::new();

    let align = |offset: u64| -> u64 { (offset + align - 1) / align * align };

    for size in sizes {
        let next_offset = align(offset + size);
        slices.push(Slice::new(offset, next_offset - offset));

        offset = next_offset;
    }

    (offset, slices)
}

impl<'s, T, P> OperationSeq<'s, T, P>
where
    P: UsizeId,
{
    pub fn construct<Rt: RuntimeType>(
        cg: &Cg<'s, Rt>,
        g: &SubDigraph<'_, VertexId, type2::Vertex<'s, Rt>>,
        seq: &[VertexId],
        execution_device: impl Fn(VertexId) -> type2::Device,
        def_use: &DefUse,
        obj_id_allocator: &mut IdAllocator<ObjectId>,
        libs: &mut Libs,
    ) -> Self {
        let mut ops = OperationSeq::empty();

        // This closure emits an operation that makes a slice clone if it no clone been made
        let mut sliced_is_cloned = BTreeSet::new();
        let mut make_slice_if_needed = |v: &Value, ops: &mut OperationSeq<T, P>| {
            if let Some((sliced_object, slice)) = def_use.object_is_cloned_slice(v.object_id()) {
                if !sliced_is_cloned.contains(&v.object_id()) {
                    sliced_is_cloned.insert(v.object_id());

                    ops.emit(Operation::Clone(
                        v.object_id(),
                        v.device(),
                        sliced_object,
                        Some(slice),
                    ))
                }
            }
        };

        for &vid in seq.iter().filter(|&vid| !g.vertex(*vid).is_virtual()) {
            let v = g.vertex(vid);

            match v.node() {
                type2::template::VertexNode::Subgraph(g) => todo!(),
                _ => {
                    let mut vid2mutated_obj = BTreeMap::new();

                    let node = v.node().relabeled(|u| {
                        def_use.input[&vid]
                            .iter()
                            .find(|(vid, _)| u == *vid)
                            .map(|(input_vid, vi)| {
                                // Clone slices, if needed
                                vi.iter().for_each(|v| {
                                    make_slice_if_needed(v, &mut ops);
                                });

                                // If input is mutated
                                // - If the inputed object dies after this vertex on this its device,
                                //   it's space will be simply reused.
                                // - Otherwise, we make a clone for it
                                let vi = if let Some(mutated_v) = vi.mutable() {
                                    match def_use.dies(mutated_v.object_id(), mutated_v.device()) {
                                        Die::After(x) if x == vid => {
                                            vid2mutated_obj.insert(input_vid, mutated_v.object_id());
                                            VertexInput::single_mutable(mutated_v.clone())
                                        }
                                        Die::After(..) | Die::Never => {
                                            let temp_object = obj_id_allocator.alloc();
                                            ops.emit(Operation::Clone(
                                                temp_object,
                                                mutated_v.device(),
                                                mutated_v.object_id(),
                                                None,
                                            ));

                                            vid2mutated_obj.insert(input_vid, temp_object);
                                            VertexInput::single_mutable(
                                                mutated_v.with_object_id(temp_object),
                                            )
                                        }
                                        Die::NeverUsed => panic!("{:?} cannot be unused on device {:?} because we are using it now", mutated_v.object_id(), mutated_v.device())
                                    }
                                } else {
                                    vi.clone()
                                };

                                vi.v_into()
                            })
                            .unwrap()
                    });

                    let output = def_use.value[&vid]
                        .iter()
                        .map(|v| {
                            (
                                v.deref().clone().into(),
                                v.iplace_of().map(|inplaced_vid| {
                                    vid2mutated_obj
                                        .get(&inplaced_vid)
                                        .unwrap_or_else(|| {
                                            panic!(
                                                "inplace object not found for input {:?} at {:?}",
                                                inplaced_vid, vid
                                            )
                                        })
                                        .clone()
                                }),
                            )
                        })
                        .collect();

                    let temp = cg
                        .temporary_space_needed(vid, execution_device(vid), libs)
                        .map_or_else(
                            || Vec::new(),
                            |(sizes, md)| {
                                let temp_buffer_obj_id = obj_id_allocator.alloc();
                                let (size, slices) = merge_temporary_buffer(sizes.into_iter(), 32);

                                slices
                                    .into_iter()
                                    .map(|s| {
                                        Value::new(
                                            Object::not_sliced(temp_buffer_obj_id),
                                            md,
                                            ValueNode::GpuBuffer(size as usize, s),
                                        )
                                        .into()
                                    })
                                    .collect()
                            },
                        );

                    let t2op = Operation::Type2(vid, output, node, temp, v.src().clone());

                    ops.emit(t2op);
                }
            }
        }

        ops
    }
}

pub mod liveness;
pub mod object_info;
mod subgraph_def_use;
