use crate::args::VariableId;
use crate::devices::{DeviceType, EventId, ThreadId};
use crate::functions::FunctionId;
use crate::typ::Typ;

pub enum Instruction {
    Allocate {
        device: DeviceType,
        typ: Typ,
        id: VariableId,
        offset: Option<usize>, // for gpu allocation
    },

    Deallocate {
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
}
