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

// impl<Rt: RuntimeType> Serialize for Chunk<Rt> {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer,
//     {
//         let mut map = serializer.serialize_map(Some(5))?;
//         map.serialize_entry("instructions", &self.instructions)?;
//         map.serialize_entry("f_table", &self.f_table.map_by_ref::<FunctionId, _>(&mut |_, func| func.meta.clone()))?;
//         map.serialize_entry("event_table", &self.event_table)?;
//         map.serialize_entry("n_variables", &self.n_variables)?;
//         map.serialize_entry("n_threads", &self.n_threads)?;
//         map.end()
//     }
// }

// struct ChunkVisitor<Rt>(PhantomData<Rt>);

// impl<'de, Rt: RuntimeType> Visitor<'de> for ChunkVisitor<Rt> {
//     type Value = Chunk<Rt>;

//     fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
//         formatter.write_str("a Chunk")
//     }

//     fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
//     where
//         A: serde::de::MapAccess<'de>,
//     {
//         let mut instructions = None;
//         let mut f_table: Option<Heap<FunctionId, FuncMeta>> = None;
//         let mut event_table = None;
//         let mut n_variables = None;
//         let mut n_threads = None;

//         while let Some(key) = map.next_key()? {
//             match key {
//                 "instructions" => instructions = Some(map.next_value()?),
//                 "f_table" => f_table = Some(map.next_value()?),
//                 "event_table" => event_table = Some(map.next_value()?),
//                 "n_variables" => n_variables = Some(map.next_value()?),
//                 "n_threads" => n_threads = Some(map.next_value()?),
//                 _ => return Err(serde::de::Error::unknown_field(key, &["instructions", "f_table", "event_table", "n_variables", "n_threads"])),
//             }
//         }

//         let libs = todo!("Load libs");

//         Ok(Chunk {
//             instructions: instructions.unwrap(),
//             // f_table: f_table.unwrap(),
//             event_table: event_table.unwrap(),
//             n_variables: n_variables.unwrap(),
//             n_threads: n_threads.unwrap(),
//             libs
//         })
//     }
// }

// impl<'de, Rt: RuntimeType> Deserialize<'de> for Chunk<Rt> {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: serde::Deserializer<'de>,
//     {
//         deserializer.deserialize_map(ChunkVisitor(PhantomData))
//     }
// }
