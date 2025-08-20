use serde::{ser::SerializeTuple, Deserialize, Serialize};
use std::any;

/// A slice that is allowed to wrap around the sliced array.
/// `.0` is the offset of the slice, and `.1` is the length of the slice.
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

    pub fn is_plain(&self, deg: u64) -> bool {
        self.0 == 0 && self.1 == deg
    }

    pub fn rotated(&self, delta: i64, deg: u64) -> Self {
        let offset = self.begin() as i64 + delta;
        Slice(offset.rem_euclid(deg as i64) as u64, self.1)
    }

    pub fn is_contained_in(&self, other: &Self) -> bool {
        self.begin() >= other.begin() && self.end() <= other.end()
    }

    pub fn relative_to(&self, other: &Self) -> Self {
        if !self.is_contained_in(other) {
            panic!(
                "slice {:?} is not contained in {:?}, cannot get relative slice",
                self, other
            );
        }
        Slice(self.0 - other.0, self.1)
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum PolyType {
    Coef,
    Lagrange,
}

pub type PolyMeta = Slice;

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
    use pasta_curves::{arithmetic::CurveAffine, group::ff::Field};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
    pub enum Typ<P> {
        ScalarArray { len: usize, meta: P },
        PointBase { len: usize },
        Scalar,
        Transcript,
        Point,
        Tuple,
        Any(AnyTypeId, usize),
        Stream,
        GpuBuffer(usize, P),
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

        pub fn poly(&self) -> Option<(&usize, &P)> {
            use Typ::*;
            match self {
                ScalarArray { len, meta } => Some((len, meta)),
                _ => None,
            }
        }

        pub fn unwrap_poly(&self) -> (usize, &P) {
            use Typ::*;
            match self {
                ScalarArray { len, meta } => (*len, meta),
                _ => panic!("expected ScalarArray"),
            }
        }

        pub fn size<F: Field, C: CurveAffine>(&self) -> usize {
            use Typ::*;
            let fs = std::mem::size_of::<F>();
            let es = std::mem::size_of::<C>();
            match self {
                ScalarArray { len, .. } => fs * *len,
                PointBase { len } => es * *len,
                Scalar => fs,
                Transcript => fs,
                Point => es,
                Tuple => panic!("size of Tuple is not defined"),
                Any(_, len) => *len,
                Stream => panic!("size of Stream is not defined"),
                GpuBuffer(len, ..) => *len,
            }
        }

        pub fn is_gpu_buffer(&self) -> bool {
            use Typ::*;
            match self {
                GpuBuffer(..) => true,
                _ => false,
            }
        }

        pub fn unwrap_gpu_buffer(&self) -> (usize, &P) {
            use Typ::*;
            match self {
                GpuBuffer(len, meta) => (*len, meta),
                _ => panic!("expected GpuBuffer"),
            }
        }

        pub fn can_on_disk<F, Po>(&self) -> bool {
            use Typ::*;
            match self {
                ScalarArray { len, .. } => *len * std::mem::size_of::<F>() >= 4096,
                PointBase { len } => *len * std::mem::size_of::<Po>() >= 4096,
                _ => false,
            }
        }
    }
}

pub type Typ = template::Typ<()>;

impl template::Typ<()> {
    pub fn scalar_array(len: usize) -> Self {
        template::Typ::ScalarArray { len, meta: () }
    }

    pub fn with_normalized_p(&self) -> template::Typ<PolyMeta> {
        use template::Typ::*;
        match self {
            ScalarArray { len, .. } => ScalarArray {
                len: *len,
                meta: Slice::new(0, *len as u64),
            },
            PointBase { len } => PointBase { len: *len },
            Scalar => Scalar,
            Transcript => Transcript,
            Point => Point,
            Tuple => Tuple,
            Any(tid, len) => Any(*tid, *len),
            Stream => Stream,
            GpuBuffer(len, ..) => GpuBuffer(*len, Slice::new(0, *len as u64)),
        }
    }

    pub fn with_slice(&self, slice: Option<Slice>) -> template::Typ<PolyMeta> {
        use template::Typ::*;
        match self {
            ScalarArray { len, .. } => ScalarArray {
                len: *len,
                meta: slice.unwrap(),
            },
            PointBase { len } => PointBase { len: *len },
            Scalar => Scalar,
            Transcript => Transcript,
            Point => Point,
            Tuple => Tuple,
            Any(tid, len) => Any(*tid, *len),
            Stream => Stream,
            GpuBuffer(len, ..) => GpuBuffer(*len, Slice::new(0, *len as u64)),
        }
    }
}

impl template::Typ<PolyMeta> {
    pub fn normalized(&self) -> Self {
        match self {
            Self::ScalarArray { len, .. } => Self::ScalarArray {
                len: *len,
                meta: Slice::new(0, *len as u64),
            },
            Self::GpuBuffer(len, _) => Self::GpuBuffer(*len, Slice::new(0, *len as u64)),
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
            GpuBuffer(len, ..) => GpuBuffer(*len, ()),
        }
    }

    pub fn plain_scalar_array(len: usize) -> Self {
        Self::ScalarArray {
            len,
            meta: Slice::new(0, len as u64),
        }
    }
}
