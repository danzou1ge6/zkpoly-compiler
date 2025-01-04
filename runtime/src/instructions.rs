use crate::devices::{DeviceType, Event, Stream};
use crate::typ::Typ;
use crate::functions::Function;

pub enum Instruction {
    Allocate {
        device: DeviceType,
        stream: Stream,
        typ: Typ,
        id: u32,
    },

    Deallocate {
        stream: Stream,
        id: u32,
    },

    Transfer {
        src_device: DeviceType,
        dst_device: DeviceType,
        stream: Stream,
        src_id: u32,
        dst_id: u32,
    },

    FuncCall {
        device: DeviceType,
        stream: Stream,
        func_id: u32,
        arg_ids: Vec<u32>,
    },

    Sync {
        stream: Stream,
    },

    Wait {
        stream: Stream,
        event: Event,
    },

    Record {
        stream: Stream,
        event: Event,
    },

    Fork {
        new_stream: Stream,
        parent_stream: Stream,
        instructions: Vec<Instruction>,
    }
}
