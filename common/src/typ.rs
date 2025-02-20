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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyType {
    Coef,
    Lagrange,
    ExtendedLagrange,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Typ {
    ScalarArray { len: usize, meta: PolyMeta },
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

impl Typ {
    pub fn compatible(&self, other: &Self) -> bool {
        use Typ::*;
        match (self, other) {
            (ScalarArray { len: deg1, .. }, ScalarArray { len: deg2, .. }) => deg1 == deg2,
            (otherwise1, otherwise2) => otherwise1 == otherwise2,
        }
    }

    pub fn unwrap_poly(&self) -> (usize, &PolyMeta) {
        use Typ::*;
        match self {
            ScalarArray { len, meta } => (*len, meta),
            otherwise => panic!("Expected ScalarArray, got {:?}", otherwise),
        }
    }

    pub fn normalized(&self) -> Self {
        match self {
            Typ::ScalarArray { len, .. } => Typ::ScalarArray {
                len: *len,
                meta: PolyMeta::Rotated(0),
            },
            otherwise => otherwise.clone(),
        }
    }
}
