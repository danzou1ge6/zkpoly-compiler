use crate::poly::PolyType;
use std::any;

#[derive(Debug, Clone)]
pub enum Typ {
    Poly { typ: PolyType, log_n: u32 },
    PointBase { log_n: u32 },
    Scalar,
    Transcript,
    Point,
    Rng,
    Tuple(Vec<Typ>),
    Array(Box<Typ>, usize),
    Any(any::TypeId, usize),
    Stream,
}

impl Typ {
    pub fn coef(log_n: u32) -> Self {
        Typ::Poly {
            typ: PolyType::Coef,
            log_n,
        }
    }

    pub fn lagrange(log_n: u32) -> Self {
        Typ::Poly {
            typ: PolyType::Lagrange,
            log_n,
        }
    }

    pub fn extended_lagrange(log_n: u32) -> Self {
        Typ::Poly {
            typ: PolyType::ExtendedLagrange,
            log_n,
        }
    }
}
