use std::collections::{BTreeMap, VecDeque};

use crate::{
    driver,
    transit::type3::{Device, DeviceSpecific},
};

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

/// For each object on each device, a list of operation indices where it is used.
///
/// If an object is never used, the currespounding entry in the map may be empty;
/// If an object is never used on some device, the currespounding [`Vec`] can also be empty.
///
/// The vector of indices for each object and device is guaranteed to be sorted.
#[derive(Debug, Clone)]
pub struct UsedBy(BTreeMap<ObjectId, DeviceSpecific<Vec<Index>>>);

impl UsedBy {
    fn add_use(
        &mut self,
        index: Index,
        object: ObjectId,
        device: Device,
        hd_info: &driver::HardwareInfo,
    ) {
        self.0
            .entry(object)
            .or_insert_with(|| DeviceSpecific::default(hd_info.n_gpus()))
            .get_mut(device)
            .push(index)
    }

    /// Caution that uses provided here are not necessarily the real uses during memory planning.
    /// For example, `ReclaimObject`'s device of reclaiming from is only an advice,
    /// but in reality, the reclaiming object may have been popped and we'll turn to its parent device.
    ///
    /// However, it can be guaranteed that the real uses are subset of the uses inferred here,
    /// so it's safe to deallocate object after last inferred use.
    pub fn analyze<'s, T, P>(ops: &OperationSeq<'s, T, P>, hd_info: &driver::HardwareInfo) -> Self
    where
        ObjectId: for<'a> From<&'a T>,
        P: Clone,
    {
        let mut used_by = Self(BTreeMap::new());

        for (index, op) in ops.iter() {
            op.object_uses()
                .chain(op.object_defs())
                .for_each(|(object_id, device)| {
                    used_by.add_use(index, object_id, device, hd_info);
                });
        }

        used_by
    }

    pub fn export_online(self) -> BTreeMap<ObjectId, DeviceSpecific<VecDeque<Index>>> {
        self.0
            .into_iter()
            .map(|(k, v)| (k, v.map(|v| v.into())))
            .collect()
    }
}
