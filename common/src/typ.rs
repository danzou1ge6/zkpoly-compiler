use std::any;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyType {
    Coef,
    Lagrange,
    ExtendedLagrange,
}

#[derive(Debug, Clone)]
pub enum Typ {
    ScalarArray { len: usize },
    PointBase { len: usize },
    Scalar,
    Transcript,
    Point,
    Rng,
    Tuple(Vec<Typ>),
    Array(Box<Typ>, usize),
    Any(any::TypeId, usize),
    Stream,
    GpuBuffer(usize),
}
