use std::any;

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub struct Slice(u64, u64);

impl Slice {
    pub fn new(start: u64, end: u64) -> Self {
        Slice(start, end)
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
            Rotated(rot) => ((deg as i64 + *rot as i64) as u64, deg),
        }
    }
}

pub mod template {
    use super::any;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Typ<P> {
        ScalarArray { len: usize, meta: P },
        PointBase { len: usize },
        Scalar,
        Transcript,
        Point,
        Rng,
        Tuple,
        Any(any::TypeId, usize),
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
            Rng => Rng,
            Tuple => Tuple,
            Any(id, len) => Any(*id, *len),
            Stream => Stream,
            GpuBuffer(len) => GpuBuffer(*len),
        }
    }
}
