//! Common data structures for Transit IR's

use std::collections::BTreeSet;

use zkpoly_common::digraph::internal::Digraph;

use crate::ast;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Location<'s> {
    pub file: &'s str,
    pub line: u32,
    pub column: u32,
}

impl From<std::panic::Location<'static>> for Location<'static> {
    fn from(value: std::panic::Location<'static>) -> Self {
        Self {
            file: unsafe {
                // SAFETY: s is actually a static string
                let s = value.file().as_ptr();
                let bytes = std::slice::from_raw_parts(s, value.file().len());
                std::str::from_utf8(bytes).unwrap()
            },
            line: value.line(),
            column: value.column(),
        }
    }
}

#[derive(Clone, serde::Serialize)]
pub struct SourceInfo<'s> {
    location: Vec<Location<'s>>,
    name: Option<String>,
}

impl<'s> std::fmt::Debug for SourceInfo<'s> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceInfo")
            .field("location", &self.location[0])
            .field("name", &self.name)
            .finish()
    }
}

mod source_info_deserilize {
    use super::SourceInfo;

    struct Visitor;

    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = SourceInfo<'de>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a SourceInfo")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut name = None;
            let mut location = Vec::new();

            while let Some(key) = map.next_key()? {
                match key {
                    "location" => {
                        location = map.next_value()?;
                    }
                    "name" => {
                        name = map.next_value()?;
                    }
                    _ => {}
                }
            }

            Ok(SourceInfo { location, name })
        }
    }

    impl<'de> serde::Deserialize<'de> for SourceInfo<'de> {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserializer.deserialize_map(Visitor)
        }
    }
}

impl<'s> SourceInfo<'s> {
    pub fn new(location: Vec<Location<'s>>, name: Option<String>) -> Self {
        Self { location, name }
    }
}

impl From<ast::SourceInfo> for SourceInfo<'_> {
    fn from(value: ast::SourceInfo) -> Self {
        Self {
            location: vec![value.loc.into()],
            name: value.name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize, serde::Serialize)]
pub enum PolyInit {
    Zeros,
    Ones,
}

/// Computation Graph of a Transit IR function.
/// [`V`]: vertex
/// [`I`]: vertex ID
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cg<I, V> {
    pub(crate) output: I,
    pub(crate) g: Digraph<I, V>,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum HashTyp {
    WriteProof,
    NoWriteProof,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Vertex<N, T, S>(N, T, S);

impl<N, T, S> Vertex<N, T, S> {
    pub fn new(v_node: N, v_typ: T, v_src: S) -> Self {
        Self(v_node, v_typ, v_src)
    }
    pub fn node(&self) -> &N {
        &self.0
    }
    pub fn typ(&self) -> &T {
        &self.1
    }
    pub fn src(&self) -> &S {
        &self.2
    }
    pub fn node_mut(&mut self) -> &mut N {
        &mut self.0
    }
    pub fn typ_mut(&mut self) -> &mut T {
        &mut self.1
    }
    pub fn src_mut(&mut self) -> &mut S {
        &mut self.2
    }

    pub fn map_typ<T2>(self, f: impl FnOnce(T) -> T2) -> Vertex<N, T2, S> {
        Vertex(self.0, f(self.1), self.2)
    }
}

// pub mod type1;
pub mod type2;
pub mod type3;
