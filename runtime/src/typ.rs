use std::any;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyRepr {
    Coef(u64),
    Lagrange(u64),
    ExtendedLagrange(u64),
}

#[derive(Debug, Clone)]
pub enum Typ {
    Poly(PolyRepr),
    Scalar,
    Transcript,
    Point,
    Rng,
    Tuple(Vec<Typ>),
    Array(Box<Typ>, usize),
    Any(any::TypeId, usize),
}

impl Typ {
    pub fn coef(n: u64) -> Self {
        Typ::Poly(PolyRepr::Coef(n))
    }

    pub fn lagrange(n: u64) -> Self {
        Typ::Poly(PolyRepr::Lagrange(n))
    }

    pub fn extended_lagrange(n: u64) -> Self {
        Typ::Poly(PolyRepr::ExtendedLagrange(n))
    }
}