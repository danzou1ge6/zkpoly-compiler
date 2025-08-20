use zkpoly_common::heap::UsizeId;

use super::super::template::ResidentalT;
use super::prelude::*;
use super::slice_analysis;
use super::value::Object;

pub struct PrologueBuilder {
    /// The exact slice that the object ID is available in
    available_slices: BTreeMap<ObjectId, Slice>,
}

pub struct Prologue<'s, P> {
    // Each element `(inside, outside)` means slicing input object `outside` to obtain object `inside` used by internal ops.
    inputs: Vec<(Object, Object)>,
    ops: OperationSeq<SubgraphOperation<'s, Object, P>>,
    intermediates: Vec<Object>,
    outputs: Vec<Object>,
}

struct AllEqualReducer<T>(Option<T>);

impl<T> AllEqualReducer<T> {
    pub fn take(self) -> Option<T> {
        self.0
    }
}

impl<T> FromIterator<T> for AllEqualReducer<T>
where
    T: Eq,
{
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self {
        AllEqualReducer(iter.into_iter().fold(None, |acc, item| match acc {
            Some(acc) => {
                if acc == item {
                    Some(acc)
                } else {
                    panic!("not all equal")
                }
            }
            None => Some(item),
        }))
    }
}

/// Produce an object for object analysis. Refer to `make_value` for detail.
fn make_object(sav: &slice_analysis::Atom, object_slice: Option<Slice>) -> Object {
    Object::new(
        sav.object().id(),
        sav.object().node().clone().with_slice(object_slice),
    )
}

/// Produce an atom for object analysis.
/// The produce atom has normalized node, the provided object ID and the desired device from slice analysis.
///
/// When `saa` 's object is part of a polynomial, the position of the part in the polynomial is specified by `object_is_slice`.
/// The produced atom's node is normalized to the object.
fn make_value<P>(
    saa: &slice_analysis::Atom,
    object_is_slice: Option<Slice>,
) -> ResidentalAtom<Object, Option<P>> {
    ResidentalT::new(
        value::Atom::new(
            make_object(saa, object_is_slice.clone()),
            saa.device(),
            saa.object()
                .node()
                .clone()
                .with_slice(object_is_slice.map(|s| Slice::new(0, s.len()))),
        ),
        None,
    )
}

impl PrologueBuilder {
    pub fn construct<'s, P: UsizeId, Rt: RuntimeType>(
        &mut self,
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        inputs_mapping: impl Fn(type2::VertexId) -> super::super::template::InputRv<Object, Option<P>>,
        def_use: &slice_analysis::DefUse,
        obj_id_allocator: &mut IdAllocator<ObjectId>,
    ) -> Prologue<'s, P>
    where
        P: Ord,
    {
        let mut ops = OperationSeq::empty();

        // Note that Subgraph has no dead vertices for now
        for vid in cg.g.vertices() {
            let v = cg.g.vertex(vid);

            use sliceable_subgraph::template::VertexNode::*;
            use type2::template::SliceableNode::*;
            match v.node() {
                Inner(RotateIdx(..))  => {
                    // Their information are already embeded in [`Object`]'s
                }
                TupleGet(..) => todo!("how to unpack nested tuples"),
                Input(..) | Return(..) => todo!("consider how to convey prologue input and output operations to memory planning; also, we may have intermediate return in the future"),
                _ => {
                    // All inputs should have the same slice
                    let the_slice = def_use
                        .input(vid)
                        .iter()
                        .map(|(_, vi)| vi.iter().map(|v| v.object().ranges().cloned()).flatten())
                        .flatten()
                        .collect::<AllEqualReducer<_>>()
                        .take();

                    // If input is sliced, then each output should be defined on the same slice
                    if let Some(slice) = the_slice {
                        def_use.value(vid).iter().for_each(|ov| {
                            if ov.object().can_be_sliced() {
                                self.available_slices.insert(ov.object().id(), slice);
                            }
                        })
                    }

                    let mut mutable_inputs = BTreeMap::new();

                    // TODO handle slice-outs

                    // Rewrite inputs to [`InputRv`]'s for object analysis
                    let node: ChunkedNode<Object, Option<P>> = v.node().relabeled(
                        |u| {
                            def_use
                                .input(vid)
                                .iter()
                                .find(|(vid, _)| u == *vid)
                                .map(|(_, vi)| {
                                    // Make cloned slice if needed
                                    // - If the input is sliced
                                    //   - If the needed slice is same as slice object we already have, use it
                                    //   - Otherwise, emit a clone operation to obtain the desired input slice
                                    // - Otherwise, trivially produce value (scalar, etc.)
                                    vi.map_ref(|v| {
                                        let (rv, cloned) = if let Some(input_slice) = v.object().ranges().next() {
                                            let availabele_slice =
                                                &self.available_slices[&v.object().id()];
                                            if availabele_slice == input_slice {
                                                (
                                                    make_value::<P>(
                                                        v,
                                                        Some(input_slice.clone())
                                                    ),
                                                    false,
                                                )
                                            } else {
                                                let cloned_value =  make_value(
                                                    v,
                                                    Some(input_slice.clone())
                                                );
                                                ops.emit(MemoryOp::Clone(
                                                    cloned_value.object().clone(),
                                                    v.device(),
                                                    make_object(v, Some(availabele_slice.clone())),
                                                    Some(input_slice.relative_to(availabele_slice)),
                                                ));

                                                (cloned_value, true)
                                            }
                                        } else {
                                            (
                                                make_value(v, None),
                                                false,
                                            )
                                        };

                                        match v.mutability() {
                                            Mutability::Mut => {
                                                if cloned {
                                                    value::InputT::new(rv, Mutability::Mut)
                                                } else {
                                                    let new_obj_id = obj_id_allocator.alloc();
                                                    let cloned_obj = Object::new(new_obj_id, rv.node().clone());

                                                    ops.emit(MemoryOp::Clone(
                                                        cloned_obj.clone(),
                                                        v.device(),
                                                        rv.object().clone(),
                                                        None,
                                                    ));

                                                    let cloned_rv = 
                                                        ResidentalT::new(
                                                            value::Atom::new(cloned_obj, rv.device(), rv.node().clone()),
                                                            None
                                                        );

                                                    mutable_inputs.insert(u, cloned_rv.clone());

                                                    value::InputT::new(
                                                        cloned_rv,
                                                        Mutability::Mut
                                                    )
                                                }
                                            },
                                            _ => value::InputT::new(rv, Mutability::Const)
                                        }
                                    })
                                }).unwrap()
                        },
                        &inputs_mapping,
                    );

                    // The output of the vertex should have slice the same as `the_slice`
                    let output = def_use
                        .value(vid)
                        .map_ref(|ov| {
                            value::OutputT::new(
                                make_value(
                                    ov,
                                    the_slice
                                ),
                                ov.inplace_of().map(|src_vid| {
                                    def_use
                                        .input(vid)
                                        .iter()
                                        .find(|(u, _)| *u == src_vid)
                                        .map(|(u, _)| mutable_inputs[u].clone())
                                        .unwrap()
                                }),
                            )
                        });

                    let chunked_op = ChunkedOp::new(vid, output, node, v.src().clone());

                    ops.emit(chunked_op);
                }
            }
        }

        Prologue {
            ops,
            inputs: todo!(),
            outputs: todo!(),
            intermediates: todo!(),
        }
    }
}
