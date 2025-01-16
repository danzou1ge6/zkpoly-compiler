use std::{any, marker::PhantomData};
use std::mem::size_of;
use zkpoly_runtime::{args::RuntimeType, poly::PolyType};

#[derive(Debug, Clone)]
pub enum Typ<Rt: RuntimeType> {
    Poly { typ: PolyType, deg: u64 },
    PointBase { log_n: u32 },
    Scalar,
    Transcript,
    Point,
    Rng,
    Tuple(Vec<Typ<Rt>>),
    Array(Box<Typ<Rt>>, usize),
    Any(any::TypeId, usize),
    _Phantom(PhantomData<Rt>)
}

impl<Rt> Typ<Rt> where Rt: RuntimeType {
    pub fn size(&self) -> u64 {
        use Typ::*;
        match self {
            Poly { deg, .. } => *deg * size_of::<Rt::Field>() as u64,
            PointBase { log_n } => (1 << log_n) * 2 * size_of::<Rt::Field>() as u64,
            Scalar => size_of::<Rt::Field>() as u64,
            Transcript => 0,
            Point => 2 * size_of::<Rt::Field>() as u64,
            Rng => 0,
            Tuple(ts) => ts.iter().map(|t| t.size()).sum(),
            Array(t, len) => t.size() * *len as u64,
            Any(_, size) => *size as u64,
            _Phantom(_) => unreachable!(),
        }
    }

    pub fn unwrap_poly(&self) -> (PolyType, u64) {
        use Typ::*;
        match self {
            Poly { typ, deg } => (typ.clone(), *deg),
            _ => panic!("called unwrap_poly on non-poly type"),
        }
    }
}
