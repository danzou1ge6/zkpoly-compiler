use std::{collections::BTreeMap, marker::PhantomData, ops::Deref};

use zkpoly_common::typ::Typ;
use zkpoly_runtime::args::RuntimeType;

use super::{
    size::{IntegralSize, Size, SmithereenSize},
    template::{Operation, OperationSeq},
    value::Value,
    ObjectId,
};

/// What we know about objects.
/// All appeared objects in operation sequence should can be found here.
#[derive(Debug, Clone)]
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
            use Operation::*;
            match op {
                Type2(_, outputs, inputs, temps, _) => {
                    let values = outputs
                        .iter()
                        .map(|(rv, _inplace_of)| rv.deref())
                        .chain(
                            inputs
                                .uses_ref()
                                .map(|vi| vi.iter().map(|rv| rv.deref()))
                                .flatten(),
                        )
                        .chain(temps.iter().map(|rv| rv.deref()));

                    values.for_each(|v| info.add_value(v));
                }
                Clone(o1, _, o2, Some(slice)) => {
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
                Clone(o1, _, o2, None) => {
                    info.typ.insert(*o1, info.typ[o2].clone());
                }
                EjectObject(v_to, _) => {
                    info.add_value(v_to);
                }
                ReclaimObject(v_to, _) | TransferObject(v_to, _) => {
                    info.add_value(v_to.deref());
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

    pub fn typ(&self, object: ObjectId) -> Typ {
        self.typ[&object].clone()
    }

    pub fn sizes<'a>(&'a self) -> impl Iterator<Item = Size> + 'a {
        self.typ.values().map(|t| {
            Size::Smithereen(SmithereenSize(t.size::<Rt::Field, Rt::PointAffine>() as u64))
        })
    }
}
