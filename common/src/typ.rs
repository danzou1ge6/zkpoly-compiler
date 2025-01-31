use std::any;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyType {
    Coef,
    Lagrange,
    ExtendedLagrange,
}

#[derive(Debug, Clone)]
pub enum Typ {
    ScalarArray { typ: PolyType, len: usize },
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

impl Typ {
    pub fn coef(len: usize) -> Self {
        Typ::ScalarArray {
            typ: PolyType::Coef,
            len,
        }
    }

    pub fn lagrange(len: usize) -> Self {
        Typ::ScalarArray {
            typ: PolyType::Lagrange,
            len,
        }
    }

    pub fn extended_lagrange(len: usize) -> Self {
        Typ::ScalarArray {
            typ: PolyType::ExtendedLagrange,
            len,
        }
    }
}
