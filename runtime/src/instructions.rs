use crate::args::{ConstantId, VariableId};
use crate::devices::{DeviceType, EventId, ThreadId};
use crate::functions::FunctionId;
use zkpoly_common::typ::Typ;

#[derive(Debug, Clone)]
pub enum Instruction {
    Allocate {
        device: DeviceType,
        typ: Typ,
        id: VariableId,
        offset: Option<usize>, // for gpu allocation
    },

    Deallocate {
        // free the underlying memory
        id: VariableId,
    },

    RemoveRegister {
        // only delete the register file
        id: VariableId,
    },

    Transfer {
        src_device: DeviceType,
        dst_device: DeviceType,
        stream: Option<VariableId>,
        src_id: VariableId,
        dst_id: VariableId,
    },

    FuncCall {
        func_id: FunctionId,
        arg_mut: Vec<VariableId>,
        arg: Vec<VariableId>,
    },

    Wait {
        slave: DeviceType,
        stream: Option<VariableId>,
        event: EventId,
    },

    Record {
        stream: Option<VariableId>,
        event: EventId,
    },

    Fork {
        new_thread: ThreadId,
        instructions: Vec<Instruction>,
    },

    Join {
        thread: ThreadId,
    },

    Rotation {
        src: VariableId,
        dst: VariableId,
        shift: i64,
    },

    Slice {
        src: VariableId,
        dst: VariableId,
        start: usize,
        end: usize,
    },

    LoadConstant {
        src: ConstantId,
        dst: VariableId,
    },

    AssembleTuple {
        vars: Vec<VariableId>,
        dst: VariableId,
    },

    Blind {
        dst: VariableId,
        start: usize,
        end: usize,
    },

    Return(VariableId),
    // SetSliceMeta { // for directly operating on the meta data of

    // }
}

pub fn print_instructions(instructions: &Vec<Instruction>, spaces: usize) {
    let prefix = " ".repeat(spaces);
    for (idx, instruct) in instructions.iter().enumerate() {
        if let Instruction::Fork { new_thread, instructions } = instruct {
            println!("{}{}: Fork: {:?}", prefix, idx, new_thread);
            print_instructions(instructions, spaces + 2);
        } else {
            println!("{}{}: {:?}", prefix, idx, instruct);
        }
    }
}