use std::marker::PhantomData;

use super::Chunk;
use serde::{de::Visitor, ser::SerializeMap, Deserialize, Serialize};
use zkpoly_runtime::args::RuntimeType;

impl<Rt: RuntimeType> Serialize for Chunk<Rt> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(5))?;
        map.serialize_entry("instructions", &self.instructions)?;
        map.serialize_entry("f_table", &1)?;
        map.serialize_entry("event_table", &self.event_table)?;
        map.serialize_entry("n_variables", &self.n_variables)?;
        map.serialize_entry("n_threads", &self.n_threads)?;
        todo!("Serialize function table");
        map.end()
    }
}

struct ChunkVisitor<Rt>(PhantomData<Rt>);

impl<'de, Rt: RuntimeType> Visitor<'de> for ChunkVisitor<Rt> {
    type Value = Chunk<Rt>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a Chunk")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut instructions = None;
        let mut f_table = None;
        let mut event_table = None;
        let mut n_variables = None;
        let mut n_threads = None;

        while let Some(key) = map.next_key()? {
            match key {
                "instructions" => instructions = Some(map.next_value()?),
                // "f_table" => f_table = Some(map.next_value()?),
                "event_table" => event_table = Some(map.next_value()?),
                "n_variables" => n_variables = Some(map.next_value()?),
                "n_threads" => n_threads = Some(map.next_value()?),
                _ => return Err(serde::de::Error::unknown_field(key, &["instructions", "f_table", "event_table", "n_variables", "n_threads"])),
            }
        }
        
        let libs = todo!("Load libs");

        Ok(Chunk {
            instructions: instructions.unwrap(),
            f_table: f_table.unwrap(),
            event_table: event_table.unwrap(),
            n_variables: n_variables.unwrap(),
            n_threads: n_threads.unwrap(),
            libs
        })
    }
}

impl<'de, Rt: RuntimeType> Deserialize<'de> for Chunk<Rt> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(ChunkVisitor(PhantomData))
    }
}

