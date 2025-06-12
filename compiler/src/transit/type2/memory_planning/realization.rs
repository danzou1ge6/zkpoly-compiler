use crate::transit::type2::memory_planning::{prelude::*, Pointer};
use type3::{Instruction, InstructionNode};

pub mod machine;
pub use machine::{Machine, MachineHandle, MemoryBlock, RealizationResponse};

pub fn realize<'s, 'a, T, Rt: RuntimeType>(
    ops: OperationSeq<'s, T, Pointer>,
    mut allocators_x: AllocatorCollection<'a, 's, T, Pointer, Rt>,
    libs: Libs,
    object_id_allocator: IdAllocator<ObjectId>,
    obj_inf: &object_info::Info<Rt>,
    hd_info: &HardwareInfo,
) -> Result<type3::Chunk<'s, Rt>, Error<'s>>
where
    T: std::fmt::Debug,
    ObjectId: for<'o> From<&'o T>,
{
    let mut aux_x = AuxiliaryInfo::new(
        liveness::UsedBy::analyze(&ops, hd_info),
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::new(),
        obj_inf,
        hd_info.n_gpus(),
    );

    let mut machine_x: Machine<'_, Pointer> = Machine::empty();

    let allocators = &mut allocators_x;
    let machine = &mut machine_x;
    let aux = &mut aux_x;

    // fixme
    println!("begin realization");

    for (_, op) in ops.into_iter() {
        use Operation::*;

        // fixme
        println!("op = {:?}", &op);

        match op {
            Type2(vid, outputs, node, temps, src) => {
                // Some inputs are tuples, which we'll free after using them in this vertex
                let mut temporary_tuple_registers = Vec::new();

                // Record mutable inputs, to use in markings for inplace outputs
                let mut mutable_inputs = Vec::new();

                let input_objects = node
                    .uses_ref()
                    .map(|vi| vi.iter().map(|v| v.object_id()))
                    .flatten()
                    .collect::<BTreeSet<_>>();
                // Get registers for inputs
                let node = node.relabeled(|vi| match vi {
                    VertexInput::Single(rv, mutability) => {
                        let r = machine.defined_reg_for(&rv.clone().assume_pointed());
                        if mutability == Mutability::Mut {
                            mutable_inputs.push((rv, r));
                        }
                        r
                    }
                    VertexInput::Tuple(rvs) => {
                        let regs = rvs
                            .into_iter()
                            .map(|rv| machine.defined_reg_for(&rv.assume_pointed()))
                            .collect::<Vec<_>>();
                        let tuple_reg = machine.new_tuple_reg(regs);
                        temporary_tuple_registers.push(tuple_reg);
                        tuple_reg
                    }
                });

                // Get registers for outputs.
                // - If the output is inplace, or if the output is immortal,
                //   its space need no allocation, so the currespounding output register
                //   is not created yet.
                // - If the output is not inplace, the currespounding output register must have been created by
                //   previous memory allocation instruction, which is undefined.
                // - If the output object is the same as some input object,
                //   create a new register for the output.pointing to the same object
                let output_regs =
                    outputs.into_iter().map(|(rv, inplace_of)| {
                        if let Some(inplace_of) = inplace_of {
                            let inplace_reg = mutable_inputs
                                .iter()
                                .find(|(input_rv, _)|
                                    input_rv.object_id() == inplace_of
                                ).map(|(input_rv, input_reg)| {
                                    if input_rv.clone().with_object_id(rv.object_id()) != rv {
                                        panic!("inplace input value must be identical to the output except for object id, but got {:?} and {:?}", input_rv, rv);
                                    }

                                    *input_reg
                                }).unwrap_or_else(|| {
                                    panic!("no mutable input found for {:?}", inplace_of)
                                });
                            (
                                machine.new_reg(rv.clone().assume_pointed()),
                                Some(inplace_reg)
                            )
                        } else if node.immortal() {
                            (
                                machine.new_reg(rv.clone().assume_pointed()),
                                None
                            )
                        } else if input_objects.contains(&rv.object_id()) {
                            (
                                machine.new_reg(rv.clone().assume_pointed()),
                                None
                            )
                        } else {
                            (machine.undefined_reg_for(&rv.clone().assume_pointed()), None)
                        }
                    }).collect::<Vec<_>>();

                // Get registers for outputs
                let temp_regs = temps
                    .iter()
                    .map(|rv| machine.undefined_reg_for(&rv.clone().assume_pointed()))
                    .collect::<Vec<_>>();

                machine.emit(Instruction::new(
                    InstructionNode::Type2 {
                        ids: output_regs,
                        temp: temp_regs,
                        vertex: node,
                        vid,
                    },
                    src,
                ));

                // Free temporary tuple registers
                temporary_tuple_registers.into_iter().for_each(|reg| {
                    machine.free_reg(reg);
                });
            }
            TransferObject(rv_to, rv_from) => {
                machine.transfer_object(rv_to, rv_from);
            }
            Transfer(to_device, to_pointer, t, from_device, from_pointer) => {
                let resp = allocators.realizer(from_device, machine, aux).transfer(
                    &t,
                    &from_pointer,
                    to_device,
                    &to_pointer,
                );
                resp.commit(allocators, machine, aux)?;
            }
            Allocate(t, device, p) => {
                allocators.realizer(device, machine, aux).allocate(&t, &p);
            }
            Deallocate(t, device, p) => {
                allocators.realizer(device, machine, aux).deallocate(&t, &p);
            }
            op => panic!("{:?} is not ready for type3", &op),
        }
    }

    Ok(type3::Chunk {
        instructions: machine_x.instructions,
        register_types: machine_x.reg_booking.export_reg_types(),
        register_devices: machine_x.reg_booking.export_reg_devices(),
        reg_memory_blocks: machine_x.reg_booking.export_memory_blocks(),
        reg_id_allocator: machine_x.reg_booking.reg_id_allocator(),
        obj_id_allocator: object_id_allocator,
        libs,
        _phantom: PhantomData,
    })
}
