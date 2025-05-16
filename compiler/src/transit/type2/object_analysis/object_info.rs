use std::{collections::BTreeMap, marker::PhantomData, ops::Deref};

use zkpoly_common::typ::Typ;
use zkpoly_runtime::args::RuntimeType;

use crate::transit::type3::{Size, SmithereenSize};

use super::{
    template::{Operation, OperationSeq},
    value::Value,
    ObjectId,
};

/// What we know about objects.
/// All appeared objects should can be found here.
pub struct Info<Rt: RuntimeType> {
    /// Their runtime types
    typ: BTreeMap<ObjectId, Typ>,
    _phantom: PhantomData<Rt>,
}

impl<Rt: RuntimeType> Info<Rt> {
    fn add_value(&mut self, value: &Value) {
        self.typ
            .entry(value.object_id())
            .and_modify(|t| {
                if t != &value.node().erase_p() {
                    panic!(
                        "inconsistent value reference to object: {:?} vs {:?}",
                        t,
                        value.node()
                    )
                }
            })
            .or_insert(value.node().erase_p());
    }

    pub fn collect<'s, T, P>(ops: &OperationSeq<'s, T, P>) -> Self
    where
        ObjectId: for<'a> From<&'a T>,
        P: Clone,
    {
        let mut info = Self {
            typ: BTreeMap::new(),
            _phantom: PhantomData,
        };

        for (_, op) in ops.iter() {
            match op {
                Operation::Type2(_, outputs, inputs, temps, _) => {
                    let values = outputs
                        .iter()
                        .map(|rv| rv.deref())
                        .chain(
                            inputs
                                .uses_ref()
                                .map(|vi| vi.iter().map(|rv| rv.deref()))
                                .flatten(),
                        )
                        .chain(temps.iter().map(|rv| rv.deref()));

                    values.for_each(|v| info.add_value(v));
                }
                Operation::CloneSlice(o1, _, o2, slice) => {
                    let (deg, _) = info.typ[o2].unwrap_poly();

                    if slice.len() as usize > deg {
                        panic!(
                            "slice length {} is larger than object degree {}",
                            slice.len(),
                            deg
                        );
                    }

                    info.typ
                        .insert(*o1, Typ::scalar_array(slice.len() as usize));
                }
                Operation::Clone(o1, _, o2) | Operation::Move(o1, _, o2) => {
                    info.typ.insert(*o1, info.typ[o2].clone());
                }
                _ => {}
            }
        }

        info
    }

    pub fn size(&self, object: ObjectId) -> Size {
        Size::Smithereen(SmithereenSize(
            self.typ[&object].size::<Rt::Field, Rt::PointAffine>() as u64,
        ))
    }
}
