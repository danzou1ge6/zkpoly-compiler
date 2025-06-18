use crate::transit::type2::memory_planning::prelude::*;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct AuxiliaryInfo<'i, Rt: RuntimeType> {
    pc: Index,
    next_uses: BTreeMap<ObjectId, DeviceSpecific<VecDeque<Index>>>,
    planning_devices: BTreeSet<Device>,
    unplanned_devices: BTreeSet<Device>,
    planned_devices: BTreeSet<Device>,
    obj_info: &'i object_info::Info<Rt>,
    n_gpus: usize,
}

impl<'i, Rt: RuntimeType> AuxiliaryInfo<'i, Rt> {
    fn next_uses_queue_of(&mut self, object: ObjectId, device: Device) -> &mut VecDeque<Index> {
        self.next_uses
            .entry(object)
            .or_insert_with(|| DeviceSpecific::default(self.n_gpus))
            .get_mut(device)
    }

    fn pop_next_use_until(
        &mut self,
        object: ObjectId,
        device: Device,
        criterion: impl Fn(Index) -> bool,
    ) -> Option<Index> {
        while let Some(hd) = self.next_uses_queue_of(object, device).pop_front() {
            if criterion(hd) {
                self.next_uses_queue_of(object, device).push_front(hd);
                return Some(hd);
            }
        }
        None
    }

    /// Returns the next use of `object` on `device` after (and including) current pc.
    pub fn query_next_use(&mut self, object: ObjectId, device: Device) -> Option<Index> {
        let pc = self.pc;
        let next_use = self.pop_next_use_until(object, device, |hd| hd >= pc)?;

        Some(next_use)
    }

    /// Mark that we are using the obejct on the device now, so all uses before (and including) now
    /// are popped from the queue.
    /// We'll use the first index after the current pc as the next use, if any.
    pub fn mark_use(&mut self, object: ObjectId, device: Device) -> Option<Index> {
        let pc = self.pc;
        self.pop_next_use_until(object, device, |hd| hd > pc)
    }

    /// Check if `object` is dead on all planning and unplanned devices
    pub fn dead(&mut self, object: ObjectId) -> bool {
        // We only check planning and unplanned devices here,
        // as uses on currently planned devices should have incurred reclaims from currently planning or
        // unplanned device.
        let devices = self
            .planning_devices()
            .chain(self.unplanned_devices())
            .collect::<Vec<_>>();
        self.will_not_be_used_on(object, devices.into_iter())
    }

    /// Check if `object` will be not used on some devices
    pub fn will_not_be_used_on(
        &mut self,
        object: ObjectId,
        devices: impl Iterator<Item = Device>,
    ) -> bool {
        devices
            .into_iter()
            .all(|device| self.query_next_use(object, device).is_none())
    }

    pub fn next_used_device_except(&mut self, object: ObjectId, except: Device) -> Option<Device> {
        let devices = self
            .planning_devices()
            .chain(self.unplanned_devices())
            .collect::<Vec<_>>();
        devices
            .into_iter()
            .find(|device| self.query_next_use(object, *device).is_some() && *device != except)
    }

    pub fn planning_devices<'a>(&'a self) -> impl Iterator<Item = Device> + 'a {
        self.planning_devices.iter().copied()
    }

    pub fn unplanned_devices<'a>(&'a self) -> impl Iterator<Item = Device> + 'a {
        self.unplanned_devices.iter().copied()
    }

    pub fn new(
        used_by: liveness::UsedBy,
        planned_devices: BTreeSet<Device>,
        planning_devices: BTreeSet<Device>,
        unplanned_devices: BTreeSet<Device>,
        obj_info: &'i object_info::Info<Rt>,
        n_gpus: usize,
    ) -> Self {
        Self {
            pc: Index::default(),
            next_uses: used_by.export_online(),
            planned_devices,
            planning_devices,
            unplanned_devices,
            obj_info,
            n_gpus,
        }
    }

    pub fn tick(&mut self, pc: Index) {
        assert!(self.pc == 0.into() || pc > self.pc);
        self.pc = pc;
    }

    pub fn pc(&self) -> Index {
        self.pc
    }

    pub fn is_planning(&self, device: Device) -> bool {
        self.planning_devices.contains(&device)
    }

    pub fn is_unplanned(&self, device: Device) -> bool {
        self.unplanned_devices.contains(&device)
    }

    pub fn is_planned(&self, device: Device) -> bool {
        self.planned_devices.contains(&device)
    }

    pub fn obj_info(&self) -> &'i object_info::Info<Rt> {
        self.obj_info
    }
}
