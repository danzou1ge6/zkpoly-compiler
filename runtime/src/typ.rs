use std::any;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyRepr {
    Coef{log_n: u32},
    Lagrange{log_n: u32},
    ExtendedLagrange{log_n: u32},
}

#[derive(Debug, Clone)]
pub enum Typ {
    Poly(PolyRepr),
    Scalar,
    Transcript,
    Point,
    Tuple(Vec<Typ>),
    Array(Box<Typ>, usize),
    Any(any::TypeId, usize),
}

impl Typ {
    pub fn coef(log_n: u32) -> Self {
        Typ::Poly(PolyRepr::Coef{log_n})
    }

    pub fn lagrange(log_n: u32) -> Self {
        Typ::Poly(PolyRepr::Lagrange{log_n})
    }

    pub fn extended_lagrange(log_n: u32) -> Self {
        Typ::Poly(PolyRepr::ExtendedLagrange{log_n})
    }
}