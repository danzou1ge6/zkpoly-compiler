use crate::args::{ArgId, VariableId};
use crate::devices::{DeviceType, EventId, StreamId, ThreadId};
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
        stream: Option<StreamId>,
        src_id: VariableId,
        dst_id: VariableId,
    },

    FuncCall {
        device: DeviceType,
        stream: StreamId,
        func_id: FunctionId,
        arg_ids: Vec<VariableId>,
    },

    Wait {
        slave: DeviceType,
        stream: Option<StreamId>,
        event: EventId,
    },

    Record {
        stream: Option<StreamId>,
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
