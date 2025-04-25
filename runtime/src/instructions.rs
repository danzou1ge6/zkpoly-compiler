use std::io::Write;

use crate::args::{ConstantId, EntryId, RuntimeType, VariableId};
use crate::devices::{DeviceType, EventId, ThreadId};
use crate::functions::{self, FunctionId};
use serde::{Deserialize, Serialize};
use zkpoly_common::typ::Typ;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    GetScalarFromArray {
        src: VariableId,
        dst: VariableId,
        idx: usize,
        stream: Option<VariableId>,
    },

    MoveRegister {
        src: VariableId,
        dst: VariableId,
    },

    LoadInput {
        src: EntryId,
        dst: VariableId,
    },

    AssertEq {
        value: VariableId,
        expected: VariableId,
        msg: Option<String>,
    },

    Print(VariableId, String),

    CopyRegister {
        src: VariableId,
        dst: VariableId,
    },
}

pub fn instruction_label<Rt: RuntimeType>(
    inst: &Instruction,
    ftab: &functions::FunctionTable<Rt>,
) -> String {
    use Instruction::*;
    match inst {
        Allocate { device, offset, .. } => offset.map_or_else(
            || format!("Allocate({:?})", device),
            |_| format!("AllocateGpu"),
        ),
        Deallocate { .. } => "Deallocate".to_string(),
        RemoveRegister { .. } => "RemoveRegister".to_string(),
        Transfer {
            src_device,
            dst_device,
            ..
        } => {
            format!("Transfer({:?}->{:?})", src_device, dst_device)
        }
        FuncCall { func_id, .. } => {
            format!(
                "Call({}: {})",
                usize::from(*func_id),
                &ftab[*func_id].meta.name
            )
        }
        Wait { stream, .. } => {
            stream.map_or_else(|| format!("WaitThread"), |_| format!("WaitStream"))
        }
        Record { stream, .. } => {
            stream.map_or_else(|| format!("Record"), |_| format!("RecordStream"))
        }
        Fork { .. } => "Fork".to_string(),
        Join { .. } => "Join".to_string(),
        Rotation { shift, .. } => format!("Rotation({})", shift),
        Slice { start, end, .. } => format!("Slice({},{})", start, end),
        LoadConstant { .. } => "LoadConstant".to_string(),
        AssembleTuple { .. } => "AssembleTuple".to_string(),
        Blind { start, end, .. } => format!("Blind({}, {})", start, end),
        Return(..) => "Return".to_string(),
        SetSliceMeta { offset, len, .. } => format!("SetSliceMeta({}, {})", offset, len),
        GetScalarFromArray { idx, stream, .. } => stream.map_or_else(
            || format!("IndexPoly({})", idx),
            |_| format!("IndexPolyGpu({})", idx),
        ),
        MoveRegister { .. } => "Move".to_string(),
        LoadInput { .. } => "LoadInput".to_string(),
        AssertEq { .. } => "AssertEq".to_string(),
        Print(_, label) => format!("Print({})", label),
        CopyRegister { .. } => "CopyRegister".to_string(),
    }
}

pub fn static_args(inst: &Instruction) -> Option<String> {
    use Instruction::*;
    match inst {
        Allocate { typ, offset, .. } => Some(offset.map_or_else(
            || format!("{:?}", typ),
            |offset| format!("{:?}, {}", typ, offset),
        )),
        Wait { event, slave, .. } => Some(format!("{:?}, {:?}", event, slave)),
        Record { event, .. } => Some(format!("{:?}", event)),
        Fork { new_thread, .. } => Some(format!("{:?}", new_thread)),
        Join { thread } => Some(format!("{:?}", thread)),
        LoadConstant { src, .. } => Some(format!("{:?}", src)),
        LoadInput { src, .. } => Some(format!("{:?}", src)),
        _ => None,
    }
}

pub fn labeled_mutable_uses(inst: &Instruction) -> Vec<(VariableId, String)> {
    use Instruction::*;
    match inst {
        Allocate { id, .. } => vec![(*id, "".to_string())],
        Deallocate { id, .. } => vec![(*id, "".to_string())],
        RemoveRegister { id, .. } => vec![(*id, "".to_string())],
        Transfer { dst_id, .. } => vec![(*dst_id, "".to_string())],
        FuncCall { arg_mut, .. } => arg_mut.iter().map(|id| (*id, "".to_string())).collect(),
        Wait { .. } => vec![],
        Record { .. } => vec![],
        Fork { .. } => vec![],
        Join { .. } => vec![],
        Rotation { dst, .. } => vec![(*dst, "".to_string())],
        Slice { dst, .. } => vec![(*dst, "".to_string())],
        LoadConstant { dst, .. } => vec![(*dst, "".to_string())],
        AssembleTuple { dst, .. } => vec![(*dst, "".to_string())],
        Blind { dst, .. } => vec![(*dst, "".to_string())],
        Return(..) => vec![],
        SetSliceMeta { dst, .. } => vec![(*dst, "".to_string())],
        GetScalarFromArray { dst, .. } => vec![(*dst, "".to_string())],
        MoveRegister { dst, .. } => vec![(*dst, "".to_string())],
        LoadInput { dst, .. } => vec![(*dst, "".to_string())],
        AssertEq { .. } => vec![],
        Print(..) => vec![],
        CopyRegister { dst, .. } => vec![(*dst, "".to_string())],
    }
}

pub fn labeled_uses(inst: &Instruction) -> Vec<(VariableId, String)> {
    use Instruction::*;
    match inst {
        Allocate { .. } | Deallocate { .. } | RemoveRegister { .. } => vec![],
        Transfer { src_id, .. } => vec![(*src_id, "".to_string())],
        FuncCall { arg, .. } => arg.iter().map(|id| (*id, "".to_string())).collect(),
        Wait { .. } | Record { .. } | Fork { .. } | Join { .. } => vec![],
        Rotation { src, .. } => vec![(*src, "".to_string())],
        Slice { src, .. } => vec![(*src, "".to_string())],
        LoadConstant { .. } => vec![],
        AssembleTuple { vars, .. } => vars.iter().map(|id| (*id, "".to_string())).collect(),
        Blind { .. } => vec![],
        Return(src) => vec![(*src, "".to_string())],
        SetSliceMeta { src, .. } => vec![(*src, "".to_string())],
        GetScalarFromArray { src, .. } => vec![(*src, "".to_string())],
        MoveRegister { src, .. } => vec![(*src, "".to_string())],
        LoadInput { .. } => vec![],
        AssertEq {
            value, expected, ..
        } => vec![
            (*value, "value".to_string()),
            (*expected, "expected".to_string()),
        ],
        Print(value, _) => vec![(*value, "".to_string())],
        CopyRegister { dst, .. } => vec![(*dst, "".to_string())],
    }
}

pub fn stream(inst: &Instruction) -> Option<VariableId> {
    use Instruction::*;
    match inst {
        Transfer { stream, .. }
        | Wait { stream, .. }
        | Record { stream, .. }
        | GetScalarFromArray { stream, .. } => *stream,
        _ => None,
    }
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

pub fn print_instructions(
    instructions: &[Instruction],
    writer: &mut impl Write,
) -> std::io::Result<()> {
    print_instructions_indented(instructions, 0, writer)
}
