use std::any;


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slice(u64, u64);

impl Slice {
    pub fn new(start: u64, end: u64) -> Self {
        Slice(start, end)
    }

    pub fn union(&self, other: &Self) -> Self {
        Slice(self.0.min(other.0), self.1.max(other.1))
    }

    pub fn union_with(&mut self, other: &Self) {
        self.0 = self.0.min(other.0);
        self.1 = self.1.max(other.1);
    }

    pub fn is_within(&self, other: &Self) -> bool {
        self.0 >= other.0 && self.1 <= other.1
    }

    pub fn relative_of(&self, outer: &Self) -> Self {
        assert!(self.is_within(outer));
        Slice(self.0 - outer.0, self.1 - outer.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyType {
    Coef,
    Lagrange,
    ExtendedLagrange,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyMeta {
    pub slice: Slice,
    pub rot: i32
}

impl PolyMeta {
    pub fn plain(len: usize) -> Self {
        PolyMeta {
            slice: Slice(0, len as u64),
            rot: 0,
        }
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
            (ScalarArray {len: deg1, ..}, ScalarArray {len: deg2, .. }) => deg1 == deg2,
            (otherwise1, otherwise2) => otherwise1 == otherwise2
        }
    }
}
