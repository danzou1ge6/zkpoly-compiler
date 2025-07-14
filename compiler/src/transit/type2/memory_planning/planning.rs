use crate::{transit::type2::memory_planning::prelude::*};
use allocators::SuperAllocator;
pub use machine::*;

pub mod machine;

/// Get input object to where it is needed.
fn prepare_input<'s, 'a, 'i, P, Rt: RuntimeType>(
    vi: &mut ResidentalValue<Option<P>>,
    planning_devices: &BTreeSet<Device>,
    unplanned_devices: &BTreeSet<Device>,
    obj_info: &object_info::Info<Rt>,
    allocators: &mut AllocatorCollection<'a, 's, ObjectId, P, Rt>,
    machine: &mut Machine<'s, ObjectId, P>,
    info: &mut AuxiliaryInfo<'i, Rt>,
    arbitary_input_device: bool
) -> Result<(), Error<'s>> where P: std::fmt::Debug
{
    let arbitary_input_device = arbitary_input_device && matches!(vi.node(), ValueNode::ScalarArray {..});
    
    let device = vi.device();
    let object = vi.object_id();
    if planning_devices.contains(&device) {
        // Prepare inputs for any planning device
        // - If input is already found on needed device, do nothing
        // - Otherwise, if input is found on any planning device, send it
        // - Otherwise, the input must be on a to-plan device, so we proceed to find such device.
        //   It can't be only on a planned device, because when a planned device was being planned,
        //   all use of objects defined on it are transfered when needed.
        if allocators.handle(device, machine, info)
            .completeness(object)
            .is_one()
        {
            // do nothing
        } else if let Some(src_device) = allocators
            .object_available_on(
                planning_devices .iter() .copied() .filter(|&d| d != device),
                object,
                machine,
                info
            ).pop()
        {
            if arbitary_input_device {
                *vi.device_mut() = src_device;
            } else {
                let resp = allocators
                    .handle(device, machine, info)
                    .claim(&object, obj_info.size(object) , src_device);
                resp.commit(allocators, machine, info)?;
            }

        } else if let Some(src_device) = allocators
            .object_available_on(
                unplanned_devices .iter() .copied(),
                object,
                machine,
                info
            ).pop()
        {
            if arbitary_input_device {
                *vi.device_mut() = src_device;
            } else {
                let resp = allocators
                    .handle(device, machine, info)
                    .claim(&object, obj_info.size(object), src_device);
                resp.commit(allocators, machine, info)?;
            }
        } else {
            panic!("expect to find {:?}, {:?} on either planning device or unplanned device", &object, &device);
        }
    } else if unplanned_devices.contains(&device) {
        // Prepare inputs for unplanned devices
        // - If input is found on a unplanned device, do nothing,
        //   and we'll consider this case when planning that device
        // - If input is found on a planning device, eject object from it.
        //   The input cannot only be found on a planned device, for the same reason as above.
        if let Some(src_device) = allocators
            .object_available_on(
                unplanned_devices .iter() .copied(),
                object,
                machine,
                info
            ).pop()
        {

            if arbitary_input_device {
                *vi.device_mut() = src_device;
            }
        } else if let Some(src_device) = allocators
            .object_available_on(
                planning_devices.iter() .copied(),
                object,
                machine,
                info
            ).pop()
        {
            if arbitary_input_device {
                *vi.device_mut() = src_device;
            } else {
                let resp = allocators
                    .handle(device, machine, info)
                    .claim(&object, obj_info.size(object), src_device);
                resp.commit(allocators, machine, info)?;
            }

        } else {
            panic!("expect to find {:?}, {:?} on either planning device or unplanned device", &object, &device);
        }
    }

    Ok(())
}

/// Run memory planning on `planning_devices`, where `planned_devices` has been planned and `unplanned_devices`
/// has not.
/// 
/// The invariant is that all needed reclaim/reject from and to planned devices are already in the
/// operation sequence `source_ops`.
/// Also, before an object dies on all devices, i.e., will not be used anywhere,
/// it must can be found on some device.
/// 
/// `allocators_x` must contains allocators for all planning devices.
/// 
/// Note that for inplace operation
///   (v1 <- io2, ... ) = type2_vertex (u1, u2, ...)
/// where v1, u1, u2 are [`ResidentalValue`]'s,  u2.object() is io2, we'll assign the same pointer to v1 and u2
fn plan_devices<'s, 'a, P, Rt: RuntimeType>(
    source_ops: OperationSeq<'s, ObjectId, P>,
    obj_info: &object_info::Info<Rt>,
    planned_devices: &BTreeSet<Device>,
    planning_devices: &BTreeSet<Device>,
    unplanned_devices: &BTreeSet<Device>,
    allocators_x: AllocatorCollection<'a, 's, ObjectId, P, Rt>,
    hd_info: &HardwareInfo,
) -> Result<OperationSeq<'s, ObjectId, P>, Error<'s>>
where
    P: UsizeId + 'static,
{
    let mut info_x = AuxiliaryInfo::new(
        liveness::UsedBy::analyze(&source_ops, hd_info),
        planned_devices.clone(),
        planning_devices.clone(),
        unplanned_devices.clone(),
        obj_info,
        hd_info.n_gpus(),
    );

    // Device here does not matter since realization will not be performed
    let mut unplanned_allocators: BTreeMap<_, SuperAllocator<'s, P, Rt, Cpu>> = unplanned_devices
        .iter()
        .map(|d| (*d, SuperAllocator::for_unplanned()))
        .collect();

    // Parent devices of all planning devices, which is expected to be subset of unplanned devices
    let top_device: BTreeSet<_> = planning_devices
        .iter()
        .copied()
        .filter_map(|d|
            d.parent()
        ).collect();
    if !top_device.iter().all(|d| unplanned_devices.contains(d)) {
        panic!("all parent device of planning devices should be a unplanned device");
    }

    // Include allocators for both planning devices and unplanned devices,
    // where unplanned devices' allocators are super allocators with unlimited space.
    let mut allocators_x = unplanned_allocators.iter_mut().fold(allocators_x, |acc, (d, a)| acc.insert(*d, a as &mut (dyn Allocator<'s, ObjectId, P, Rt> + 's)) );

    let mut machine_x = Machine::new();

    let machine = &mut machine_x;
    let aux = &mut info_x;
    let allocators = &mut allocators_x;

    macro_rules! planning {
        ($device:expr) => {
            planning_devices.contains(&$device)
        };
    }
    macro_rules! unplanned {
        ($device:expr) => {
            unplanned_devices.contains(&$device)
        };
    }
    macro_rules! planned {
        ($device:expr) => {
            planned_devices.contains(&$device)
        };
    }
    macro_rules! planning_or_unplanned {
        ($device:expr) => {
            unplanned_devices.contains(&$device) | planning_devices.contains(&$device)
        }
    }
    macro_rules! get_src_pointer {
        ($src_device:expr, $object:expr) => {
            allocators.handle($src_device, machine, aux)
                .access(&$object)
                .expect("i just found pointer to this object on this device")
        }
    }

    for (index, op) in source_ops.into_iter() {
        aux.tick(index);

        use Operation::*;

        let object_uses = op.object_uses().collect::<Vec<_>>();

        // fixme
        println!("{:?}: op = {:?}", index, op);

        match op {
            Type2(vid, mut outputs, mut node, temps, src) => {
                let arbitary_input_supported = match &node {
                    type2::template::VertexNode::Arith {chunking, ..} if chunking.is_some() => true,
                    _ => false
                };
                
                if let Some(vo) = temps.first()
                {
                    if planning_or_unplanned!(vo.device()) {
                        let resp = allocators.handle(vo.device(), machine, aux)
                            .allocate(obj_info.size(vo.object_id()), &vo.object_id());
                        resp.commit(allocators, machine, aux).map_err(|e| e.with_vid_src(vid, src.clone()))?;
                    }
                }

                // Prepare inputs
                for vi in node
                    .uses_mut()
                    .map(|vi| vi.iter_mut())
                    .flatten()
                {
                    prepare_input(
                        vi,
                        planning_devices,
                        unplanned_devices,
                        obj_info,
                        allocators,
                        machine,
                        aux,
                        arbitary_input_supported
                    )?;

                    aux.add_next_use_now(vi.object_id(), vi.device());

                    outputs.iter_mut().find(|(_, inplace_of)| inplace_of.is_some_and(|u| u == vi.object_id()))
                        .map(|(rv, _)| {
                            println!("reset output {:?} on {:?}", vi.object_id(), vi.device());
                            *rv.device_mut() = vi.device();
                        });
                } 

                if let type2::template::VertexNode::Arith {arith, ..} = &node {
                    let (is, os) = arith.space_needed::<Rt::Field>();
                    if is + os > hd_info.cpu().integral_space() as usize {
                        outputs.iter_mut().filter(|(_, inplace_of)| inplace_of.is_none())
                            .filter(|(rv, _)| rv.node().can_on_disk::<Rt::Field, Rt::PointAffine>())
                            .for_each(|(rv, _)| {*rv.device_mut() = Device::Disk; });
                    }
                }

                let node = node;
                let outputs = outputs;

                // Allocate outputs and temporaries.
                // We only care about objects on planning devices and unplanned devices.
                // - If the output took place of some input object,
                //   we first get the input object to where it will later be used,
                //   then later reuse space of the input object after pointers for inputs have been obtained.
                // - If the output object is the same as input, do nothing
                // - Otherwise, allocate new space
                let input_objects = node
                    .uses_ref()
                    .map(|vi|
                        vi.iter().map(|vi| vi.object_id())
                    ).flatten()
                    .collect::<BTreeSet<_>>();
                for (vo, inplace_of) in outputs
                    .iter()
                {
                    if planning_or_unplanned!(vo.device()) {
                        if inplace_of.is_some() || input_objects.contains(&vo.object_id()) {
                            // do nothing
                        } else {
                            let resp = allocators.handle(vo.device(), machine, aux)
                                .allocate(obj_info.size(vo.object_id()), &vo.object_id());
                            resp.commit(allocators, machine, aux)
                                .map_err(|e| e.with_vid_src(vid, src.clone()))?;
                        }
                    }
                }


                // Try to obtain pointer for inputs.
                // We only care about objects on planning devices, and pointers to unplanned devices will be assigned
                // in later phases.
                //
                // If no pointer can be obtained, we know that prepared inputs has been ejected,
                // indicating that memory space is insufficient.
                let node = node.try_relabeled(|ri| {
                    ri.try_map_v(|rv| {
                        if planning!(rv.device()) {
                            let pointer = allocators
                                .handle(rv.device(), machine, aux)
                                .access(&rv.object_id())
                                .ok_or_else(|| {
                                    Error::VertexInputsOutputsNotAccommodated(Some((vid, src.clone())))
                                }).unwrap_or_else(|_| panic!("can not find {:?} on {:?}", rv.object_id(), rv.device()));
                            if rv.pointer().is_some() {
                                panic!("input {:?} of {:?} at {:?} has its pointer already assigned", rv, vid, index);
                            }
                            Ok(rv.with_pointer(Some(pointer)))
                        } else {
                            Ok(rv)
                        }
                    })
                })?;

                // Reuse space for inplace inputs, so that the allocator can find pointer to the output object
                for (vo, inplace_of) in outputs
                    .iter()
                {
                    if planning_or_unplanned!(vo.device()) {
                        if let Some(inplace_of) = *inplace_of {
                            if let Some(target_device) = aux.next_used_device_except(
                            inplace_of, vo.device()
                            ) {
                                if vo.device() == Device::Disk {
                                    // fixme
                                    println!("instead of reuse, make a clone from {:?} to {:?} on {:?} here", inplace_of, vo.object_id(), vo.device());
                                    let r = allocators.handle(vo.device(), machine, aux)
                                        .allocate(obj_info.size(vo.object_id()), &vo.object_id());
                                    let to_pointer = r.commit(allocators, machine, aux)?;

                                    let c = Continuation::provide_object_sliced(vo.device(), inplace_of, vo.device(), vo.object_id(), to_pointer, None);
                                    allocators.apply_continuation(c, machine, aux)?;

                                } else {
                                    // fixme
                                    println!("before reuse, send {:?} from {:?} to {:?}", inplace_of, vo.device(), target_device);

                                    let from_pointer = allocators
                                        .handle(vo.device(), machine, aux)
                                        .access(&inplace_of)
                                        .expect("i just found pointer to this input object on this device");
                                    let c = Continuation::simple_send(
                                        target_device,
                                        vo.device(),
                                        from_pointer,
                                        inplace_of,
                                        obj_info.size(vo.object_id())
                                    );
                                    allocators.apply_continuation(c, machine, aux)?;
                                }
                            }

                            allocators.handle(vo.device(), machine, aux)
                                .reuse(vo.object_id(), inplace_of);
                        }
                    }
                }

                let mut pointer_for =
                    |rv: ResidentalValue<Option<P>>| -> Result<ResidentalValue<Option<P>>, Error<'s>> {
                        let pointer = 
                            allocators.handle(rv.device(), machine, aux)
                                .access(&rv.object_id())
                                .ok_or_else(|| {
                                    Error::VertexInputsOutputsNotAccommodated(Some((vid, src.clone())))
                                }).unwrap();
                        if rv.pointer().is_some() {
                            panic!("output/temporary {:?} of {:?} at {:?} has its pointer already assigned", rv, vid, index);
                        }
                        Ok(rv.with_pointer(Some(pointer)))
                    };

                // Try to obtain pointer for outputs.
                // Works the same as inputs.
                let outputs = outputs.into_iter()
                        .map(|(rv, inplace_of)| {
                            if planning!(rv.device()) {
                                Ok((pointer_for(rv)?, inplace_of))
                            } else {
                                Ok((rv, inplace_of))
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                let temps = temps.into_iter()
                        .map(|rv| {
                            if planning!(rv.device()) {
                                pointer_for(rv)
                            } else {
                                Ok(rv)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                machine.emit(Operation::Type2(vid, outputs, node, temps.clone(), src));

                // Deallocate temporary spaces right away, for clearity
                temps.first()
                    .map(|rv| {
                        if planning!(rv.device()) {
                            assert!(rv.pointer().is_some());
                            allocators.handle(rv.device(), machine, aux)
                                .deallocate(&rv.object_id());
                        }
                    });
            }
            Clone(new_object, device, sliced_object, slice) if planning_or_unplanned!(device) => {

                // Choose a device to slice `sliced_object` from.
                let from_device = if planning!(device) {
                    // When `device` is being planned, the priority is
                    // - `device` , transfer object from it
                    // - Planning device, transfer object from it
                    // - Unplanned device, reclaim object from it
                    if allocators.handle(device, machine, aux).completeness(sliced_object).is_one() {
                        Some(device)
                    } else if let Some(src_device) =  allocators
                        .object_available_on(
                            planning_devices.iter() .copied().filter(|&d| d != device),
                            sliced_object,
                            machine,
                            aux
                        ).pop()
                    {
                        Some(src_device)
                    } else if let Some(src_device) = allocators
                        .object_available_on(
                            unplanned_devices.iter() .copied(),
                            sliced_object,
                            machine,
                            aux
                        ).pop()
                    {
                        Some(src_device)
                    } else {
                        panic!("no planning or unplanneddevice available for object {:?}", sliced_object);
                    }
                } else if unplanned!(device) {
                    // When `device` is unplanned, the priority is
                    // - Unplanned device (including `device`), do nothing in this case
                    // - Planning device, eject object from it.
                    if !allocators
                        .object_available_on(
                            unplanned_devices .iter() .copied(),
                            sliced_object,
                            machine,
                            aux
                        ).is_empty()
                    {
                        // do nothing
                        machine.emit(Clone(new_object, device, sliced_object, slice));
                        None

                    } else if let Some(src_device) = allocators
                        .object_available_on(
                            planning_devices.iter() .copied(),
                            sliced_object,
                            machine,
                            aux
                        ).pop()
                    {
                        Some(src_device)
                    }  else {
                        panic!("no planning or unplanned device available for object {:?}", sliced_object);
                    }
                } else {
                    unreachable!()
                };

                // Allocate space for `new_object`
                let resp = allocators.handle(device, machine, aux)
                    .allocate(obj_info.size(new_object), &new_object);
                resp.commit(allocators, machine, aux)?;

                // For now, disk does not support slicing.
                // So, if we are cloning slice from disk, we first claim whole `sliced_object`,
                // then clone-slice it on `device`.
                if from_device.is_some_and(|from: Device| from == Device::Disk) && slice.is_some() {
                    let from_device = from_device.unwrap();
                    let resp = allocators
                        .handle(device, machine, aux)
                        .claim(&sliced_object, obj_info.size(sliced_object), from_device);
                    resp.commit(allocators, machine, aux)?;

                    let to_pointer = allocators
                        .handle(device, machine, aux)
                        .access(&new_object)
                        .expect("your memory would be too small if it cannot hold two objects of this size");

                    let c = Continuation::provide_object_sliced(
                        device,
                        sliced_object,
                        device,
                        new_object,
                        to_pointer,
                        slice
                    );
                    allocators.apply_continuation(c, machine, aux)?;

                } else if let Some(from_device) = from_device {
                    let to_pointer = allocators
                        .handle(device, machine, aux)
                        .access(&new_object)
                        .expect("your memory would be too small if it cannot hold two objects of this size");

                    let c = Continuation::provide_object_sliced(
                        from_device,
                        sliced_object,
                        device,
                        new_object,
                        to_pointer,
                        slice
                    );
                    allocators.apply_continuation(c, machine, aux)?;
                }
            }
            Clone(_, device, _, _) => {
                panic!("{:?} not found on either planning devices or unplanned devices, but this should indicates that the Clone operation has been filled with pointer", device);
            }
            EjectObject(to_v, from_rv) => {
                // The ejected object must be from some planned device.
                // - If it's ejected to a planning device, we allocate space for it and issue transfer-object instruction.
                // - If it's ejected to a unplanned device, we do nothing.
                assert!(planned!(from_rv.device()));

                if planning!(to_v.device()) {
                    let resp = allocators.handle(to_v.device(), machine, aux)
                        .allocate(obj_info.size(to_v.object_id()), &to_v.object_id());
                    resp.commit(allocators, machine, aux)?;
                    let to_p = allocators.handle(to_v.device(), machine, aux)
                        .access(&to_v.object_id())
                        .expect("i just allocated this object");
                    
                    machine.transfer_object(
                        ResidentalValue::new(to_v, to_p),
                        from_rv
                    );
                } else if unplanned!(to_v.device()) {
                    // do nothing
                    machine.emit(EjectObject(to_v, from_rv));
                } else {
                    panic!("{:?} not found on either planning devices or unplanned devices, but this should indicates that the EjectObject operation has been filled with some pointer", to_v);
                }
            }
            ReclaimObject(to_rv, from_v) => {
                assert!(planned!(to_rv.device()));

                // If the object is found on a planning device, we simply transfer it.
                // Otherwise, it must has been popped from `from_v.device()` , and we'll look for it on
                // unplannded devices.
                // We expect to find it, and we'll leave for later planning phases to decide its source location.
                if let Some(src_device) = allocators
                        .object_available_on(
                            planning_devices.iter() .copied(),
                            from_v.object_id(),
                            machine,
                            aux
                        ).pop()
                {
                    let src_pointer = get_src_pointer!(src_device, from_v.object_id());

                    machine.transfer_object(
                        to_rv,
                        ResidentalValue::new(from_v.with_device(src_device), src_pointer)
                    )

                } else if let Some(src_device) = allocators
                        .object_available_on(
                            unplanned_devices.iter() .copied(),
                            from_v.object_id(),
                            machine,
                            aux
                        ).pop()
                {
                    machine.reclaim_object(to_rv, from_v.with_device(src_device));

                } else {
                    panic!("{:?} not found on either planning devices or unplanned devices, but this should indicates that the ReclaimObject operation has been filled with some pointer", from_v);
                }
            }
            tr@TransferObject(..) => {
                // We already know source and destination pointer, so we just leave the operation as it is.
                machine.ops.emit(tr)
            }
            Eject(to_device, t, source_device, source_pointer) => {
                assert!(planned!(source_device));

                if planning_or_unplanned!(to_device) {
                    let c = Continuation::simple_receive(source_device, source_pointer, to_device, obj_info.size(t), t);
                    allocators.apply_continuation(c, machine, aux)?;
                } else {
                    panic!("should eject to unplanned device, so {:?} must be planning or unplanned, but it is not", to_device);
                }
            }
            Reclaim(to_device, to_pointer, t, _from_device) => {
                assert!(planned!(to_device));

                // If the object is found on a planning device, we simply send object from it.
                // Otherwise, it must has been popped from `from_v.device()` , and we'll look for it on
                // unplannded devices.
                // We expect to find it, and we'll leave for later planning phases to decide its source location.
                if let Some(src_device) = allocators
                        .object_available_on(
                            planning_devices.iter() .copied(),
                            t.into(),
                            machine,
                            aux
                        ).pop()
                {
                    let c = Continuation::simple_provide(to_device, to_pointer, src_device, t);
                    allocators.apply_continuation(c, machine, aux)?;

                } else if let Some(src_device) = allocators
                        .object_available_on(
                            unplanned_devices.iter() .copied(),
                            t.into(),
                            machine,
                            aux
                        ).pop()
                {
                    machine.emit(Reclaim(to_device, to_pointer, t, src_device));
                } else {
                    panic!("{:?} not found on either planning devices or unplanned devices, but this should indicate that the Reclaim operation has been rewritten with some pointer", &t);
                }
            }
            op@Transfer(..) | op@Allocate(..) | op@Deallocate(..) => {
                // We already know source and destination pointer, so we just leave the operation as it is.
                machine.ops.emit(op)
            }
        }

        // Mark objects as used, so their next uses will be updated
        for (object, device) in object_uses.iter().copied() {
            aux.mark_use(object, device);
        }

        // For objects that will not be used on any planning devices, eject them
        // from all planning devices
        for (object, _) in object_uses.iter().copied() {
            if aux.will_not_be_used_on(object, planning_devices.iter().copied()) {
                for &d in planning_devices.iter() {
                    if allocators
                        .handle(d, machine, aux)
                        .completeness(object).is_one()
                    {
                        let from_p = allocators
                            .handle(d, machine, aux)
                            .access(&object)
                            .unwrap();
                        let c = Continuation::simple_eject(
                            d,
                            from_p,
                            object,
                            obj_info.size(object)
                        );
                        allocators.apply_continuation(c, machine, aux)?;

                        let resp = allocators
                            .handle(d, machine, aux)
                            .deallocate(&object);
                        resp.commit(allocators, machine, aux)?;
                    }
                }
            }
        }

        // For objects that will not be used on any unplanned devices nor any planning devices,
        // deallocate them on all planning and unplanned devices
        for (object, _) in object_uses {
            if aux.dead(object) {
                for d in planning_devices.iter().chain(unplanned_devices.iter()) {
                    if allocators
                        .handle(*d, machine, aux)
                        .completeness(object)
                        .is_one() {
                        let resp = allocators
                            .handle(*d, machine, aux)
                            .deallocate(&object);
                        resp.commit(allocators, machine, aux)?;
                    }
                }
            }
        }
    }

    Ok(machine_x.ops)
}

/// Run `plan_devices` repetitively from bottom devices to top devices.
pub fn transform_ops<'s, P, Rt: RuntimeType, Ca, Ga, Da> (
    mut ops: OperationSeq<'s, ObjectId, P>,
    gpu_allocators: &mut [Ga],
    cpu_allocator: &mut Ca,
    disk_allocator: &mut Da,
    obj_info: &object_info::Info<Rt>,
    hd_info: &HardwareInfo
) -> Result<OperationSeq<'s, ObjectId, P>, Error<'s>>
where
    P: UsizeId + 'static,
    Ca: Allocator<'s, ObjectId, P, Rt> + 's,
    Ga: Allocator<'s, ObjectId, P, Rt> + 's,
    Da: Allocator<'s, ObjectId, P, Rt> +'s,
{
    let plan_phases: Vec<BTreeSet<Device>> = vec![
        (0..hd_info.n_gpus()).map(|i| Device::Gpu(i)).collect(),
        [Device::Cpu].into_iter().collect(),
        [Device::Disk].into_iter().collect()
    ];
    
    let mut planned_devices = BTreeSet::new();
    let mut unplanned_devices = plan_phases
        .iter()
        .fold(
            BTreeSet::new(),
            |acc, devices| acc
                .union(devices)
                .into_iter()
                .cloned()
                .collect()
        );
    
    for mut planning_devices in plan_phases.into_iter() {
        unplanned_devices = unplanned_devices.difference(&planning_devices).into_iter().cloned().collect();

        let allocators: AllocatorCollection<ObjectId, P, Rt> = gpu_allocators
            .iter_mut()
            .enumerate()
            .map(|(i, alloc)| (Device::Gpu(i), alloc as &mut dyn Allocator<ObjectId, P, Rt>))
            .chain(std::iter::once((Device::Cpu, cpu_allocator as &mut dyn Allocator<ObjectId, P, Rt>)))
            .chain(std::iter::once((Device::Disk, disk_allocator as &mut dyn Allocator<ObjectId, P, Rt>)))
            .collect();

        // fixme
        println!("begin phase planning {:?}, unplanned {:?}, planned {:?}", &planning_devices, &unplanned_devices, &planned_devices);

        ops = plan_devices(
            ops,
            obj_info,
            &planned_devices,
            &planning_devices,
            &unplanned_devices,
            allocators,
            hd_info,
        )?;

        planned_devices.append(&mut planning_devices);
    }

    Ok(ops)
}
