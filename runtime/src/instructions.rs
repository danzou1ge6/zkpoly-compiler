use std::io::Write;

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
    SetSliceMeta {
        // for directly operating on the meta data of
        src: VariableId,
        dst: VariableId,
        offset: usize,
        len: usize,
    },
}

fn print_instructions_indented(
    instructions: &[Instruction],
    spaces: usize,
    writer: &mut impl Write,
) -> std::io::Result<()> {
    let prefix = " ".repeat(spaces);
    for (idx, instruct) in instructions.iter().enumerate() {
        if let Instruction::Fork {
            new_thread,
            instructions,
        } = instruct
        {
            writeln!(writer, "{}{}: Fork: {:?}", prefix, idx, new_thread)?;
            print_instructions_indented(instructions, spaces + 2, writer)?;
        } else {
            writeln!(writer, "{}{}: {:?}", prefix, idx, instruct)?;
        }
    }
    Ok(())
}

pub fn print_instructions(instructions: &[Instruction], writer: &mut impl Write) -> std::io::Result<()> {
    print_instructions_indented(instructions, 0, writer)
}

