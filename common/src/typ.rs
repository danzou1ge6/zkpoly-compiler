use serde::{ser::SerializeTuple, Deserialize, Serialize};
use std::any;

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub struct Slice(u64, u64);

impl Slice {
    pub fn new(start: u64, len: u64) -> Self {
        Slice(start, len)
    }

    pub fn len(&self) -> u64 {
        self.1
    }

    pub fn begin(&self) -> u64 {
        self.0
    }

    pub fn end(&self) -> u64 {
        self.0 + self.1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyType {
    Coef,
    Lagrange,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyMeta {
    Sliced(Slice),
    Rotated(i32),
}

impl PolyMeta {
    pub fn plain() -> Self {
        PolyMeta::Rotated(0)
    }

    pub fn offset_and_len(&self, deg: u64) -> (u64, u64) {
        use PolyMeta::*;
        match self {
            Sliced(Slice(start, len)) => (*start, *len),
            Rotated(rot) => (((deg as i64 + *rot as i64) % (deg as i64)) as u64, deg),
        }
    }

    pub fn len(&self, deg: u64) -> u64 {
        self.offset_and_len(deg).1
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AnyTypeId {
    type_id: any::TypeId,
}

impl std::fmt::Debug for AnyTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.type_id.fmt(f)
    }
}

impl From<any::TypeId> for AnyTypeId {
    fn from(id: any::TypeId) -> Self {
        AnyTypeId { type_id: id }
    }
}

impl From<AnyTypeId> for any::TypeId {
    fn from(value: AnyTypeId) -> Self {
        value.type_id
    }
}

impl Serialize for AnyTypeId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let [u1, u2] = unsafe { std::mem::transmute::<_, [u64; 2]>(self.type_id) };
        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element(&u1)?;
        tup.serialize_element(&u2)?;
        tup.end()
    }
}

struct AnyTypeIdVisitor;

impl<'de> serde::de::Visitor<'de> for AnyTypeIdVisitor {
    type Value = AnyTypeId;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a tuple of 2 u64")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let u1 = seq.next_element()?.map_or_else(
            || Err(serde::de::Error::invalid_length(0, &"2 elements")),
            |x| Ok(x),
        )?;
        let u2 = seq.next_element()?.map_or_else(
            || Err(serde::de::Error::invalid_length(1, &"2 elements")),
            |x| Ok(x),
        )?;

        Ok(AnyTypeId {
            type_id: unsafe { std::mem::transmute::<[u64; 2], any::TypeId>([u1, u2]) },
        })
    }
}

impl<'de> Deserialize<'de> for AnyTypeId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_tuple(2, AnyTypeIdVisitor)
    }
}

pub mod template {
    use super::AnyTypeId;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum Typ<P> {
        ScalarArray { len: usize, meta: P },
        PointBase { len: usize },
        Scalar,
        Transcript,
        Point,
        Tuple,
        Any(AnyTypeId, usize),
        Stream,
        GpuBuffer(usize),
    }

    impl<P> Typ<P> {
        pub fn compatible(&self, other: &Self) -> bool
        where
            P: Eq,
        {
            use Typ::*;
            match (self, other) {
                (ScalarArray { len: deg1, .. }, ScalarArray { len: deg2, .. }) => deg1 == deg2,
                (otherwise1, otherwise2) => otherwise1 == otherwise2,
            }
        }

        pub fn unwrap_poly(&self) -> (usize, &P) {
            use Typ::*;
            match self {
                ScalarArray { len, meta } => (*len, meta),
                _ => panic!("expected ScalarArray"),
            }
        }
    }
}

pub type Typ = template::Typ<()>;

impl template::Typ<()> {
    pub fn scalar_array(len: usize) -> Self {
        template::Typ::ScalarArray { len, meta: () }
    }
}

impl template::Typ<PolyMeta> {
    pub fn normalized(&self) -> Self {
        match self {
            Self::ScalarArray { len, .. } => Self::ScalarArray {
                len: *len,
                meta: PolyMeta::Rotated(0),
            },
            otherwise => otherwise.clone(),
        }
    }

    pub fn erase_p(&self) -> Typ {
        use template::Typ::*;
        match self {
            ScalarArray { len, .. } => ScalarArray {
                len: *len,
                meta: (),
            },
            PointBase { len } => PointBase { len: *len },
            Scalar => Scalar,
            Transcript => Transcript,
            Point => Point,
            Tuple => Tuple,
            Any(tid, len) => Any(*tid, *len),
            Stream => Stream,
            GpuBuffer(len) => GpuBuffer(*len),
        }
    }
}
