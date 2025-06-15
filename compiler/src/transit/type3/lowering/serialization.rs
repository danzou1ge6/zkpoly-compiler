use crate::{ast::lowering::UserFunctionId, transit::type2::{self, object_analysis::size::{IntegralSize, LogBlockSizes}}};

use super::{kernel_gen::load_function, Chunk};
use serde::{ser::SerializeStruct, Deserialize, Serialize};
use zkpoly_common::{heap::Heap, load_dynamic::Libs};
use zkpoly_runtime::{
    args::RuntimeType,
    devices::EventTypeTable,
    functions::{FunctionId, KernelType},
    instructions::Instruction,
};

#[derive(Serialize, Deserialize)]
pub struct ChunkDeserializer {
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) f_table: Heap<FunctionId, KernelType>,
    pub(crate) event_table: EventTypeTable,
    pub(crate) n_variables: usize,
    pub(crate) lbss: LogBlockSizes,
    pub(crate) n_threads: usize,
}

impl<Rt: RuntimeType> Serialize for Chunk<Rt> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut struc = serializer.serialize_struct("ChunkDeserializer", 5)?;
        struc.serialize_field("instructions", &self.instructions)?;
        let f_table = self
            .f_table
            .map_by_ref::<FunctionId, _>(&mut |_, func| func.meta.typ.clone());
        struc.serialize_field("f_table", &f_table)?;
        struc.serialize_field("event_table", &self.event_table)?;
        struc.serialize_field("n_variables", &self.n_variables)?;
        struc.serialize_field("n_threads", &self.n_threads)?;
        struc.serialize_field("lbss", &self.lbss)?;
        struc.end()
    }
}

impl ChunkDeserializer {
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
            lbss: self.lbss,
            libs: libs,
        }
    }
}
