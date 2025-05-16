static DEBUG: bool = true;

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    ops::Deref,
};

use super_allocator::SuperAllocator;
use zkpoly_common::{
    bijection::Bijection,
    digraph::internal::SubDigraph,
    heap::{Heap, IdAllocator, UsizeId},
    injection::Injection,
    load_dynamic::Libs,
};
use zkpoly_runtime::args::RuntimeType;

use crate::transit::type3::{typ::PolyMeta, Chunk, Device, DeviceSpecific};

use super::{
    super::type3::{
        typ::Typ, Addr, AddrId, AddrMapping, Device as MemoryDevice, Instruction, InstructionNode,
        IntegralSize, RegisterId, Size, SmithereenSize,
    },
    object_analysis::{
        self,
        liveness::{self, AtModifier},
        object_info,
        template::{Operation, OperationSeq, ResidentalValue},
        Index, ObjectId, VertexOutput,
    },
    user_function, VertexId,
};

#[derive(Debug, Clone)]
pub struct Machine<'s, T, P> {
    ops: OperationSeq<'s, T, P>,
}

impl<'s, T, P> Machine<'s, T, P> {
    pub fn new() -> Self {
        Machine {
            ops: OperationSeq::empty(),
        }
    }

    pub fn emit(&mut self, op: Operation<'s, T, P>) {
        self.ops.emit(op);
    }

    pub fn handle<'m>(&'m mut self, device: Device) -> MachineHandle<'m, 's, T, P> {
        MachineHandle { m: self, device }
    }

    pub fn transfer(
        &mut self,
        to_device: Device,
        to_pointer: P,
        t: T,
        from_device: Device,
        from_pointer: P,
    ) {
        self.ops.emit(Operation::Transfer(
            to_device,
            to_pointer,
            t,
            from_device,
            from_pointer,
        ));
    }
}

pub struct MachineHandle<'m, 's, T, P> {
    m: &'m mut Machine<'s, T, P>,
    device: Device,
}

impl<'m, 's, T, P> MachineHandle<'m, 's, T, P> {
    pub fn allocate(&mut self, t: T, pointer: P) {
        self.m.emit(Operation::Allocate(t, self.device, pointer))
    }

    pub fn deallocate(&mut self, t: T, pointer: P) {
        self.m.emit(Operation::Deallocate(t, self.device, pointer))
    }
}

#[derive(Debug, Clone)]
pub struct AuxiliaryInfo {
    liveness: liveness::UsedBy,
}

impl AuxiliaryInfo {
    pub fn next_use(&self, object: ObjectId, device: Device) -> Index {
        todo!()
    }

    pub fn analyze<'s, T, P>(ops: &OperationSeq<'s, T, P>) -> Self
    where
        ObjectId: for<'a> From<&'a T>,
        P: Clone,
    {
        Self {
            liveness: object_analysis::liveness::UsedBy::analyze(ops),
        }
    }
}

pub enum Response<T, P, R> {
    Complete(R),
    Continue(Continuation<T, P, R>)
}

impl<T, P, R> Response<T, P, R> {
    pub fn unwrap_complete(self) -> R {
        match self {
            Response::Complete(r) => r,
            _ => panic!("called unwrap_complete on a continuation")
        }
    }
}


pub struct Continuation<T, P, R> {
    device: Device,
    f: Box<dyn FnMut(&mut Box<dyn AllocatorHandle<T, P>>) -> Response<T, P, R>>
}

/// Handle to an allocator, with which memory can be manipulated.
///
/// See [`Allocator`] for what [`P`] is for.
pub trait AllocatorHandle<T, P> {
    fn device(&self) -> Device;

    fn panic_no_token(&self, t: &T) -> ! where T: std::fmt::Debug {
        panic!("token {:?} not recorded on device {:?}", t, self.device())
    }

    /// Allocate on memory device some token.
    /// Returns the allocated pointer, but this pointer may be immediately invalidated after some other operations
    /// such as `read` or `allocate`.
    fn allocate(&mut self, size: Size, t: &T) -> Response<T, P, P>;

    /// Deallocate the token, which must be recorded on device, otherwise panics.
    /// That is, it can be ejected, but it must has been allocated.
    fn deallocate(&mut self, t: &T) -> Response<T, P, ()>;

    /// Ask the allocator to make sure the requested token is on device.
    /// The token must be recorded on device.
    fn read(&mut self, t: &T) -> Response<T, P, ()>;

    /// Ask the allocator to make sure the requested token on device,
    /// and reassign that space to another token
    /// The token must be recorded on device.
    fn read_write(&mut self, read_t: &T, write_t: &T) -> Response<T, P, ()>;

    /// Get reference of pointer which can be used to access this token.
    /// The token must be recorded on device.
    ///
    /// If the token can't be accessed, return None.
    /// e.g. When some pages of the obejct has been ejected, in case of page
    /// allocator.
    fn access(&self, t: &T) -> Option<P>;
}

trait AllocatorRealizer<T, P> {}

/// A memory allocator.
///
/// # Type Parameters
/// [`P`] is how memory planner keep tracks of pointers.
/// It is introduced because each memory allocator may have different [`Pointer`],
/// but memory planner needs a unified way of recording them.
///
/// [`T`] is smallest unit of data involved in memory management.
///
/// # Associated Types
/// [`Handle`] provides APIs to operate the allocator.
/// [`Realizer`] provides APIs so that abstract operations can be translated to
/// Type3 memory operations.
///
/// Allocator is only responsible for keeping track of the memory device's
/// internal state, but not how the device is manipulated at runtime.
/// Therefore, we need a [`Handle`] with mutable reference to [`Machine`],
/// and [`Machine`] keeps track of the opertions needed to manipulate the device.
pub trait Allocator<T, P> {
    fn handle<'a, 'b, 'c, 'd>(
        &'a mut self,
        machine: MachineHandle<'b, '_, T, P>,
        aux: &'c AuxiliaryInfo,
    ) -> Box<dyn AllocatorHandle<T, P> + 'd>
    where
        'a: 'd, 'b: 'd, 'c: 'd;
}

pub enum Error<'s> {
    VertexInputsOutputsNotAccommodated(VertexId, super::SourceInfo<'s>),
}

pub struct AllocatorCollection<'a, P, T>(BTreeMap<Device, &'a mut dyn Allocator<T, P>>);

impl<'a, P, T> AllocatorCollection<'a, P, T> {
    pub fn get(&mut self, device: Device) -> &mut dyn Allocator<T, P> {
        *self.0.get_mut(&device).unwrap()
    }

    pub fn handle<'b, 'm, 's, 'aux, 'c>(&'b mut self, device: Device, machine: &'m mut Machine<'s, T, P>, aux: &'aux AuxiliaryInfo) -> Box<dyn AllocatorHandle<T, P> + 'c> where 'm: 'c, 'a: 'c, 'aux: 'c, 'b: 'c {
        self.get(device).handle(machine.handle(device), aux)
    }
}

impl<'a, P, T> FromIterator<(Device, &'a mut dyn Allocator<T, P>)> for AllocatorCollection<'a, P, T> {
    fn from_iter<It: IntoIterator<Item = (Device, &'a mut dyn Allocator<T, P>)>>(iter: It) -> Self {
        Self(iter.into_iter().collect())
    }
}

fn plan_devices<'s,'fa, 'a, P, Fa, Rt: RuntimeType>(
    source_ops: OperationSeq<'s, ObjectId, P>,
    obj_info: &object_info::Info<Rt>,
    planned_devices: impl Iterator<Item = Device>,
    planning_devices: impl Iterator<Item = Device>,
    not_planned_devices: impl Iterator<Item = Device>,
    allocators: &'a mut AllocatorCollection<'fa, P, ObjectId>,
) -> Result<OperationSeq<'s, ObjectId, P>, Error<'s>>
where
    P: UsizeId,
    'fa: 'a,
{
    let used_by = liveness::UsedBy::analyze(&source_ops);
    let info = AuxiliaryInfo::analyze(&source_ops);
    let planning_devices: BTreeSet<_> = planning_devices.collect();
    let not_planned_devices: BTreeSet<_> = not_planned_devices.collect();

    let mut top_device: BTreeMap<_, _> = planning_devices.iter().copied().map(|d| (d.parent().unwrap(), super_allocator::SuperAllocator::<ObjectId, P>::new())).collect();
    let top_allocators: AllocatorCollection<_, ObjectId> = top_device.iter_mut().map(|(d, a )| (*d, a as &mut dyn Allocator<ObjectId, P>) ).collect();

    let mut machine = Machine::new();

    for (index, op) in source_ops.into_iter() {
        use Operation::*;
        match op {
            Type2(vid, outputs, node, temps, src) => {
                // Prepare inputs
                for vi in node
                    .uses_ref()
                    .map(|vi| vi.iter())
                    .flatten()
                    .filter(|vi| planning_devices.contains(&vi.device()))
                {
                    // Prepare inputs for any planning device
                    // - If input is already found on needed device, read it
                    // - Otherwise, if input is found on any planning device, send it
                    // - Otherwise, the input must be on a to-plan device, so we read it and let the allocator to reclaim
                    //   the object.
                    //   It can't be only on a planned device, because when a planned device was being planned,
                    //   all use of objects defined on it are transfered when needed.
                    if planning_devices.contains(&vi.device()) {
                        if allocators.handle(vi.device(), &mut machine, &info)
                            .access(&vi.object_id())
                            .is_some()
                        {
                            allocators.handle(vi.device(), &mut machine, &info).read(&vi.object_id());
                        } else if let Some((src_device, src_pointer)) = planning_devices
                            .iter()
                            .filter(|&device| *device != vi.device())
                            .find_map(|&device| {
                                allocators.handle(device, &mut machine, &info)
                                    .access(&vi.object_id())
                                    .map(|p| (device, p))
                            })
                        {
                            let dest_pointer = allocators.handle(vi.device(), &mut machine, &info)
                                .allocate(
                                    Size::from(obj_info.size(vi.object_id())),
                                    &vi.object_id(),
                                );

                            machine.transfer(
                                vi.device(),
                                dest_pointer,
                                vi.object_id(),
                                src_device,
                                src_pointer,
                            );
                        } else {
                            allocators.handle(vi.device(), &mut machine, &info).read(&vi.object_id());
                        }
                    } else if not_planned_devices.contains(&vi.device()) {
                        // Prepare inputs for not-planned devices
                        todo!()
                    }
                }

                // Allocate outputs and temporaries
                for vo in outputs
                    .iter()
                    .chain(temps.iter())
                {
                    if planning_devices.contains(&vo.device()) {
                    allocators.handle(vo.device(), &mut machine, &info)
                        .allocate(Size::from(obj_info.size(vo.object_id())), &vo.object_id());
                    } else if not_planned_devices.contains(&vo.device()) {
                        todo!()
                    }
                }

                let mut pointer_for =
                    |rv: ResidentalValue<Option<P>>| -> Result<ResidentalValue<Option<P>>, Error<'s>> {
                        let pointer = 
                            allocators.handle(rv.device(), &mut machine, &info)
                            .access(&rv.object_id())
                            .ok_or_else(|| {
                                Error::VertexInputsOutputsNotAccommodated(vid, src.clone())
                            })?;
                        if rv.pointer().is_some() {
                            panic!("input/output/temporary {:?} of {:?} at {:?} has its pointer already assigned", rv, vid, index);
                        }
                        Ok(rv.with_pointer(Some(pointer)))
                    };

                // Try to obtain pointer for inputs.
                // If no pointer can be obtained, we know that prepared inputs has been ejected,
                // indicating that memory space is insufficient.
                let node = node.try_relabeled(|ri| {
                    ri.try_map_v(|rv| {
                        if planning_devices.contains(&rv.device()) {
                            pointer_for(rv)
                        } else {
                            Ok(rv)
                        }
                    })
                })?;

                // Try to obtain pointer for outputs.
                // Works the same as inputs.
                let [outputs, temps] = [outputs, temps].map(|rvs| {
                    rvs.into_iter()
                        .map(|rv| {
                            if planning_devices.contains(&rv.device()) {
                                pointer_for(rv)
                            } else {
                                Ok(rv)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()
                });
                let [outputs, temps] = [outputs?, temps?];

                machine.emit(Operation::Type2(vid, outputs, node, temps, src));
            }
            CloneSlice(new_object, device, sliced_object, slice) => {}
            _ => todo!(),
        }
    }

    Ok(machine.ops)
}

pub mod super_allocator;
