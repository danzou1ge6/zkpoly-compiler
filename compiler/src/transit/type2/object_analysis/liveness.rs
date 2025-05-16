use std::collections::BTreeMap;

use crate::transit::type3::{Device, DeviceSpecific};

use super::{template::OperationSeq, Index, ObjectId};

/// Distinguishes the instant before or after execution a given operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtModifier {
    Before,
    After,
}

impl std::cmp::PartialOrd for AtModifier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Before, Self::Before) => Some(std::cmp::Ordering::Equal),
            (Self::Before, Self::After) => Some(std::cmp::Ordering::Less),
            (Self::After, Self::Before) => Some(std::cmp::Ordering::Greater),
            (Self::After, Self::After) => Some(std::cmp::Ordering::Equal),
        }
    }
}

impl std::cmp::Ord for AtModifier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Before, Self::Before) => std::cmp::Ordering::Equal,
            (Self::Before, Self::After) => std::cmp::Ordering::Less,
            (Self::After, Self::Before) => std::cmp::Ordering::Greater,
            (Self::After, Self::After) => std::cmp::Ordering::Equal,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UsedBy(BTreeMap<ObjectId, DeviceSpecific<Vec<Index>>>);

impl UsedBy {
    fn add_use(&mut self, index: Index, object: ObjectId, device: Device) {
        self.0
            .entry(object)
            .or_default()
            .get_device_mut(device)
            .push(index)
    }

    pub fn analyze<'s, T, P>(ops: &OperationSeq<'s, T, P>) -> Self
    where
        ObjectId: for<'a> From<&'a T>,
        P: Clone,
    {
        let mut used_by = Self(BTreeMap::new());

        for (index, op) in ops.iter() {
            op.object_uses().for_each(|(object_id, device)| {
                used_by.add_use(index, object_id, device);
            });
        }

        used_by
    }
}
