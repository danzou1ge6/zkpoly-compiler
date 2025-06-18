use std::{collections::HashMap, time::Duration};

use crate::{devices::ThreadId, instructions::Instruction};

#[derive(Clone, Debug)]
pub struct DebugInfo {
    pub instruction: Instruction,
    pub start_duration: Duration, // Duration since the start of the program
    pub end_duration: Duration, // Duration since the start of the program
}

pub struct DebugInfoCollector {
    pub debug_info: Vec<DebugInfo>,
    pub sub_thread_debug_info: HashMap<ThreadId, Box<DebugInfoCollector>>,
}
