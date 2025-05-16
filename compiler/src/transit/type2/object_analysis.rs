//! An object is an invariant piece of data. It can be in any memory device.
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
};

use cg_def_use::DefUse;
use template::ResidentalValue;
use zkpoly_common::{
    arith::Mutability,
    define_usize_id,
    digraph::internal::SubDigraph,
    heap::{Heap, IdAllocator},
    load_dynamic::Libs,
    typ::{PolyMeta, Slice},
};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::{
    type2,
    type3::{typ::Slice as SliceRange, Device, DeviceSpecific, Size, SmithereenSize},
    SourceInfo,
};

use super::{Cg, Typ, VertexId};

pub mod value;
pub use value::{ObjectId, Value, ValueNode, VertexInput, VertexOutput};

pub type VertexNode = type2::alt_label::VertexNode<VertexInput<Value>>;

pub mod template {
    use super::{define_usize_id, Device, Heap, ObjectId, Slice, SourceInfo, Value, VertexInput, VertexId};
    use crate::transit::type2;

    /// A value that we know where its pointer points to.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct ResidentalValue<P>(Value, P);

    impl<P> ResidentalValue<P> {
        pub fn pointer(&self) -> &P {
            &self.1
        }

        pub fn with_pointer(self, pointer: P) -> Self {
            ResidentalValue(self.0, pointer)
        }

        pub fn value(&self) -> &Value {
            &self.0
        }
    }

    impl<P> std::ops::Deref for ResidentalValue<P> {
        type Target = Value;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<P> From<Value> for ResidentalValue<P>
    where
        P: Default,
    {
        fn from(value: Value) -> Self {
            ResidentalValue(value, P::default())
        }
    }

    /// A memory relevant operation involved in executing a Type2 computation graph.
    ///
    /// Type parameter [`T`] is the smallest unit of memory transferring, Token.
    /// [`P`] means Pointer, by which the object is accessed from memory.
    #[derive(Debug, Clone)]
    pub enum Operation<'s, T, P> {
        /// A Type2 vertex translated to value inputs/outputs.
        /// For `Type2(o, i, t, s)`, `o` are the outputs, `i` are the inputs and `t` are temporary spaces.
        ///
        /// Depending on current device of memory planning, some of the values' pointers
        /// may be unknown.
        Type2(
            VertexId,
            Vec<ResidentalValue<Option<P>>>,
            type2::alt_label::VertexNode<VertexInput<ResidentalValue<Option<P>>>>,
            Vec<ResidentalValue<Option<P>>>,
            SourceInfo<'s>,
        ),

        /// [`CloneSlice(o1, d, o2, s)`]  means o1 <- CloneSlice d, o2, s,
        /// i.e., Clone a slice `s` of object `o2` and store it to a new object `o1` on device `d`
        CloneSlice(ObjectId, Device, ObjectId, Slice),

        /// Similar to [`CloneSlice`], but does not slice the source object.
        Clone(ObjectId, Device, ObjectId),

        /// Reassign memory space of `.2` on `.1` to a new object ID
        Move(ObjectId, Device, ObjectId),

        /// [`Eject(d1, t, d2, p)`] means d1 <- Eject t, d2, p,
        /// i.e., eject token `t` from device `d2` at memory location `p` to device `d1`.
        ///
        /// This operation is emitted when we are planninng memory of a device but not its parent.
        /// At this point, we only know where the object resides on `d2` but not on `d1`.
        /// Later after we has planned memory of `d1`, this operation should be rewritten to a [`Transfer`]
        Eject(Device, T, Device, P),

        /// [`Reclaim(d1, p, t, d2`] means d1, p <- Reclaim t, d2,
        /// i.e., reclaim token `t` from device `d2` to device `t1` at memory location `p`.
        ///
        /// This operation serves a similar purpose as [`Eject`], but in the opposite direction.
        Reclaim(Device, P, T, Device),

        /// [`Transfer(d1, p1, t, d2, p2)`] means d1, p <- Transfer t, d2, p,
        /// i.e. transfer token `t` from device `d2` at memory location `p2` to device `d1` at memory location `p1`.
        Transfer(Device, P, T, Device, P),

        /// Allocate some token on device
        Allocate(T, Device, P),

        /// Deallocate some token on device
        Deallocate(T, Device, P),
    }

    impl<'s, T, P> Operation<'s, T, P>
    where
        P: Clone,
        ObjectId: for<'a> From<&'a T>,
    {
        /// Get all objects used by this operation, and on which device they are used.
        pub fn object_uses<'a>(&'a self) -> Box<dyn Iterator<Item = (ObjectId, Device)> + 'a> {
            use Operation::*;
            match self {
                Type2(_, _, node, temps, _) => Box::new(
                    node.uses_ref()
                        .map(|vi| vi.iter().map(|v| (v.object_id(), v.device())))
                        .flatten()
                        .chain(temps.iter().map(|v| (v.object_id(), v.device()))),
                ),
                CloneSlice(_, d, u, _) | Clone(_, d, u) | Move(_, d, u) => {
                    Box::new([(*u, *d)].into_iter())
                }
                Eject(_, t, d, _) | Reclaim(_, _, t, d) | Transfer(_, _, t, d, _) => {
                    Box::new([(t.into(), *d)].into_iter())
                }
                Allocate(_, _, _) | Deallocate(_, _, _) => Box::new(std::iter::empty()),
            }
        }
    }

    define_usize_id!(Index);

    /// A sequence of [`Operation`]
    #[derive(Debug, Clone)]
    pub struct OperationSeq<'s, T, P>(Heap<Index, Operation<'s, T, P>>);

    impl<'s, T, P> OperationSeq<'s, T, P> {
        pub fn empty() -> Self {
            OperationSeq(Heap::new())
        }

        pub fn emit(&mut self, op: Operation<'s, T, P>) {
            self.0.push(op);
        }

        pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Index, &'a Operation<'s, T, P>)> + 'a {
            self.0.iter_with_id()
        }

        pub fn into_iter(self) -> impl Iterator<Item = (Index, Operation<'s, T, P>)> {
            self.0.into_iter_with_id()
        }
    }
}

pub type Operation<'s> = template::Operation<'s, ObjectId, ()>;
pub type OperationSeq<'s> = template::OperationSeq<'s, ObjectId, ()>;
pub use template::Index;

pub mod cg_def_use;

impl<'s> OperationSeq<'s> {
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
        let mut make_slice_if_needed = |v: &Value, ops: &mut OperationSeq| {
            if let Some((sliced_object, slice)) = def_use.object_is_cloned_slice(v.object_id()) {
                if !sliced_is_cloned.contains(&v.object_id()) {
                    sliced_is_cloned.insert(v.object_id());

                    ops.emit(Operation::CloneSlice(
                        v.object_id(),
                        v.device(),
                        sliced_object,
                        slice,
                    ))
                }
            }
        };

        for &vid in seq.iter().filter(|&vid| !g.vertex(*vid).is_virtual()) {
            let v = g.vertex(vid);
            let node = v.node().relabeled(|u| {
                def_use.input[&vid]
                    .iter()
                    .find(|(vid, _)| *vid == u)
                    .map(|(_, vi)| {
                        // Clone slices, if needed
                        vi.iter().for_each(|v| {
                            make_slice_if_needed(v, &mut ops);
                        });

                        // If input is mutated
                        // - If the inputed object dies after this vertex on this its device, it's space is simply reused
                        // - Otherwise, we make a clone for it
                        let vi = if let Some(mutated_v) = vi.mutable() {
                            let temp_object = obj_id_allocator.alloc();

                            let constructor = if def_use
                                .dies_after(mutated_v.object_id(), mutated_v.device())
                                .unwrap()
                                == vid
                            {
                                Operation::Move
                            } else {
                                Operation::Clone
                            };

                            ops.emit(constructor(
                                temp_object,
                                mutated_v.device(),
                                mutated_v.object_id(),
                            ));
                            VertexInput::single_mutable(mutated_v.with_object_id(temp_object))
                        } else {
                            vi.clone()
                        };

                        vi.v_into()
                    })
                    .unwrap()
            });

            let temp = cg
                .temporary_space_needed(vid, execution_device(vid), libs)
                .map_or_else(
                    || Vec::new(),
                    |(sizes, md)| {
                        sizes
                            .into_iter()
                            .map(|s| {
                                Value::new(
                                    obj_id_allocator.alloc(),
                                    md,
                                    ValueNode::GpuBuffer(s as usize),
                                ).into()
                            })
                            .collect()
                    },
                );

            let t2op = Operation::Type2(
                vid,
                def_use.value[&vid]
                    .iter()
                    .map(|v| v.deref().clone().into())
                    .collect(),
                node,
                temp,
                v.src().clone(),
            );

            ops.emit(t2op);
        }

        ops
    }
}

pub mod liveness;
pub mod object_info;
