use zkpoly_common::heap::UsizeId;

use super::super::template::ResidentalValue;
use super::prelude::*;
use super::slice_analysis;
use super::value::Object;

pub struct PrologueBuilder {
    /// The exact slice that the object ID is available in
    available_slices: BTreeMap<ObjectId, Slice>,
}

pub struct Prologue<'s, P> {
    ops: OperationSeq<'s, Object, P>,
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

/// Produce a value for object analysis.
/// The produce value has normalized node, the provided object and the desired device from slice analysis.
fn make_value<P>(sav: &slice_analysis::Value, object: Object) -> ResidentalValue<Option<P>> {
    ResidentalValue::new(
        value::Value::new(
            object,
            sav.device(),
            sav.object().node().with_normalized_p(),
        ),
        None,
    )
}

impl PrologueBuilder {
    pub fn construct<'s, P: UsizeId, Rt: RuntimeType>(
        &mut self,
        cg: &sliceable_subgraph::Cg<'s, Rt>,
        inputs_mapping: impl Fn(type2::VertexId) -> super::super::template::VertexInput<P>,
        def_use: &slice_analysis::DefUse,
        obj_id_allocator: &mut IdAllocator<ObjectId>,
    ) -> Prologue<'s, P> {
        let mut ops = OperationSeq::empty();

        // Note that Subgraph has no dead vertices for now
        for vid in cg.g.vertices() {
            let v = cg.g.vertex(vid);

            use sliceable_subgraph::template::VertexNode::*;
            use type2::template::SliceableNode::*;
            match v.node() {
                Inner(RotateIdx(..)) | TupleGet(..) => {
                    // Their information are already embeded in [`Object`]'s
                }
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

                    let mut mutable_objects = BTreeMap::new();

                    // Rewrite inputs to [`ResidentalValue`]'s for object analysis
                    let node = v.node().relabeled(
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
                                    // - Otherwise, trivially produce value for object analysis
                                    let mut input_objects = vi.iter().map(|v| {
                                        if let Some(input_slice) = v.object().ranges().next() {
                                            let availabele_slice =
                                                &self.available_slices[&v.object().id()];
                                            if availabele_slice == input_slice {
                                                (
                                                    (make_value::<P>(
                                                        v,
                                                        Object::new(
                                                            v.object().id(),
                                                            Some(input_slice.clone()),
                                                        ),
                                                    )),
                                                    false,
                                                )
                                            } else {
                                                let cloned_obj = Object::new(
                                                    v.object().id(),
                                                    Some(input_slice.clone()),
                                                );

                                                ops.emit(Operation::Clone(
                                                    cloned_obj.clone(),
                                                    v.device(),
                                                    Object::new(
                                                        v.object().id(),
                                                        Some(availabele_slice.clone()),
                                                    ),
                                                    Some(input_slice.relative_to(availabele_slice)),
                                                ));

                                                (make_value(v, cloned_obj), true)
                                            }
                                        } else {
                                            (
                                                make_value(v, Object::new(v.object().id(), None)),
                                                false,
                                            )
                                        }
                                    });

                                    // Wrap the objects in [`VertexInput`]
                                    // - Since slices are expected to consume small space, we always make a clone for mutable uses.
                                    //   The new object will have a new object ID and no slice.
                                    match vi {
                                        VertexInput::Single(_, Mutability::Const) => {
                                            VertexInput::Single(
                                                input_objects.next().unwrap().0,
                                                Mutability::Const,
                                            )
                                        }
                                        VertexInput::Single(v, Mutability::Mut) => {
                                            let (rv, cloned) = input_objects.next().unwrap();
                                            if cloned {
                                                VertexInput::Single(rv, Mutability::Mut)
                                            } else {
                                                let new_obj_id = obj_id_allocator.alloc();
                                                let cloned_obj = Object::not_sliced(new_obj_id);

                                                ops.emit(Operation::Clone(
                                                    cloned_obj.clone(),
                                                    v.device(),
                                                    rv.object().clone(),
                                                    None,
                                                ));

                                                mutable_objects.insert(u, cloned_obj.clone());

                                                VertexInput::Single(
                                                    make_value(v, cloned_obj),
                                                    Mutability::Mut,
                                                )
                                            }
                                        }
                                        VertexInput::Tuple(..) => VertexInput::Tuple(
                                            input_objects.map(|(x, _)| x).collect(),
                                        ),
                                    }
                                })
                                .unwrap()
                        },
                        &inputs_mapping,
                    );

                    // The output of the vertex should have slice the same as `the_slice`
                    let output = def_use
                        .value(vid)
                        .iter()
                        .map(|ov| {
                            (
                                make_value(
                                    ov,
                                    Object::new(
                                        ov.object().id(),
                                        if ov.object().can_be_sliced() {
                                            the_slice
                                        } else {
                                            None
                                        },
                                    ),
                                ),
                                ov.inplace_of().map(|src_vid| {
                                    def_use
                                        .input(vid)
                                        .iter()
                                        .find(|(u, _)| *u == src_vid)
                                        .map(|(u, _)| mutable_objects[u].clone())
                                        .unwrap()
                                }),
                            )
                        })
                        .collect();

                    let chunked_op = Operation::Chunked(output, node, v.src().clone());

                    ops.emit(chunked_op);
                }
            }
        }

        Prologue { ops }
    }
}
