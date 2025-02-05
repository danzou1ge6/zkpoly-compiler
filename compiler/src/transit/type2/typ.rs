use std::mem::size_of;
use std::{any, marker::PhantomData};
use zkpoly_common::typ::PolyType;
use zkpoly_runtime::args::RuntimeType;

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
    _Phantom(PhantomData<Rt>),
}

pub enum Size {
    Single(u64),
    Tuple(Vec<u64>),
    Array(u64, usize),
}

impl Size {
    pub fn total(&self) -> u64 {
        use Size::*;
        match self {
            Single(s) => *s,
            Tuple(ss) => ss.iter().sum(),
            Array(s, len) => *s * *len as u64,
        }
    }

    pub fn iter<'s>(&'s self) -> Box<dyn Iterator<Item = &u64> + 's> {
        use Size::*;
        match self {
            Single(s) => Box::new(std::iter::once(s)),
            Tuple(ss) => Box::new(ss.iter()),
            Array(s, len) => Box::new(std::iter::repeat(s).take(*len)),
        }
    }

    pub fn unwrap_single(&self) -> u64 {
        use Size::*;
        match self {
            Single(s) => *s,
            _ => panic!("unwrap_single: not a single"),
        }
    }

    pub fn len(&self) -> usize {
        use Size::*;
        match self {
            Single(_) => 1,
            Tuple(ss) => ss.len(),
            Array(_, len) => *len,
        }
    }
}

impl<Rt> Typ<Rt>
where
    Rt: RuntimeType,
{
    pub fn size(&self) -> Size {
        use Typ::*;
        match self {
            Poly { deg, .. } => Size::Single(*deg * size_of::<Rt::Field>() as u64),
            PointBase { log_n } => Size::Single((1 << log_n) * 2 * size_of::<Rt::Field>() as u64),
            Scalar => Size::Single(size_of::<Rt::Field>() as u64),
            Transcript => unimplemented!(),
            Point => Size::Single(2 * size_of::<Rt::Field>() as u64),
            Rng => unimplemented!(),
            Tuple(ts) => Size::Tuple(
                ts.iter()
                    .map(|t| match t.size() {
                        Size::Single(s) => s,
                        Size::Tuple(_) => panic!("tuple of tuple is not supported"),
                        Size::Array(_, _) => panic!("tuple of array is not supported"),
                    })
                    .collect(),
            ),
            Array(t, len) => match t.size() {
                Size::Single(s) => Size::Array(s, *len),
                Size::Tuple(_) => panic!("array of tuple is not supported"),
                Size::Array(_, _) => panic!("array of array is not supported"),
            },
            Any(_, size) => Size::Single(*size as u64),
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

    pub fn iter<'s>(&'s self) -> Box<dyn Iterator<Item = &'s Self> + 's> {
        match self {
            Typ::Tuple(ts) => Box::new(ts.iter()),
            Typ::Array(t, n) => Box::new(std::iter::repeat(t.as_ref()).take(*n)),
            _ => Box::new(std::iter::once(self)),
        }
    }

    pub fn stack_allocable(&self) -> bool {
        use Typ::*;
        match self {
            Poly { .. } => false,
            PointBase { .. } => false,
            Scalar => true,
            Transcript => false,
            Point => true,
            Rng => true,
            Tuple(..) => true,
            Array(..) => true,
            Any(..) => false,
            _Phantom(_) => unreachable!(),
        }
    }
}
