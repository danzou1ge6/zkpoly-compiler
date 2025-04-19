use crate::{ast::lowering::UserFunctionId, transit::type2};

use super::{kernel_gen::load_function, Chunk};
use serde::{Deserialize, Serialize};
use zkpoly_common::{heap::Heap, load_dynamic::Libs};
use zkpoly_runtime::{
    args::RuntimeType,
    devices::EventTable,
    functions::{FunctionId, KernelType},
    instructions::Instruction,
};

#[derive(Serialize, Deserialize)]
pub struct ChunkSerializer {
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) f_table: Heap<FunctionId, KernelType>,
    pub(crate) event_table: EventTable,
    pub(crate) n_variables: usize,
    pub(crate) n_threads: usize,
}

impl<Rt: RuntimeType> From<Chunk<Rt>> for ChunkSerializer {
    fn from(value: Chunk<Rt>) -> Self {
        Self {
            instructions: value.instructions,
            f_table: value
                .f_table
                .map_by_ref::<FunctionId, _>(&mut |_, func| func.meta.typ.clone()),
            event_table: value.event_table,
            n_variables: value.n_variables,
            n_threads: value.n_threads,
        }
    }
}

impl ChunkSerializer {
    pub fn deserialize_into_chunk<Rt: RuntimeType>(
        self,
        user_ftable: type2::user_function::Table<Rt>,
    ) -> Chunk<Rt> {
        let mut libs = Libs::new();
        let mut uf_table: Heap<UserFunctionId, _> = user_ftable.map(&mut (|_, f| Some(f)));
        let f_table = self
            .f_table
            .map(&mut |_, kernel_type| load_function(&kernel_type, &mut libs, &mut uf_table));
        let event_table = self.event_table;
        let n_variables = self.n_variables;
        let n_threads = self.n_threads;

        Chunk {
            instructions: self.instructions,
            f_table,
            event_table,
            n_variables,
            n_threads,
            libs: libs,
        }
    }
}
