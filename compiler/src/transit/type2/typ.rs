use std::mem::size_of;
use zkpoly_common::typ::PolyType;
use zkpoly_runtime::args::RuntimeType;

pub mod template {
    use zkpoly_common::typ::AnyTypeId;
    use zkpoly_runtime::args::Variable;

    use super::RuntimeType;
    use std::{fmt::Debug, marker::PhantomData};

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub enum Typ<Rt: RuntimeType, P> {
        Poly(P),
        PointBase { log_n: u32 },
        Scalar,
        Transcript,
        Point,
        Tuple(Vec<Typ<Rt, P>>),
        Array(Box<Typ<Rt, P>>, usize),
        Any(AnyTypeId, usize),
        _Phantom(PhantomData<Rt>),
    }

    impl<Rt: RuntimeType, P> PartialEq for Typ<Rt, P>
    where
        P: Eq,
    {
        fn eq(&self, other: &Self) -> bool {
            use Typ::*;
            match (self, other) {
                (Poly(p1), Poly(p2)) => p1 == p2,
                (PointBase { log_n: l1 }, PointBase { log_n: l2 }) => l1 == l2,
                (Scalar, Scalar) => true,
                (Transcript, Transcript) => true,
                (Point, Point) => true,
                (Tuple(ts1), Tuple(ts2)) => ts1 == ts2,
                (Array(t1, l1), Array(t2, l2)) => t1 == t2 && l1 == l2,
                (Any(t1, s1), Any(t2, s2)) => t1 == t2 && s1 == s2,
                _ => false,
            }
        }
    }

    impl<Rt: RuntimeType, P> Eq for Typ<Rt, P> where P: Eq {}

    impl<Rt: RuntimeType, P: Debug> Typ<Rt, P> {
        pub fn match_arg(&self, other: &Variable<Rt>) {
            use Variable::*;
            match (self, other) {
                (Typ::Scalar, Scalar(..)) => {}
                (Typ::Poly(..), ScalarArray(..)) => {}
                (Typ::PointBase { .. }, PointArray(..)) => {}
                (Typ::Transcript, Transcript(..)) => {}
                (Typ::Point, Point(..)) => {}
                (Typ::Any(..), Any(..)) => {}
                (Typ::Tuple(ts), Tuple(vs)) => {
                    if ts.len() != vs.len() {
                        panic!("tuple length mismatch");
                    }
                    for (t, v) in ts.iter().zip(vs.iter()) {
                        t.match_arg(v);
                    }
                }
                (Typ::Array(t, len), Tuple(vs)) => {
                    if *len != vs.len() {
                        panic!("array length mismatch");
                    }
                    for v in vs.iter() {
                        t.match_arg(v);
                    }
                }
                _ => panic!("expected {:?}, got {:?}", self, other),
            }
        }
    }

    impl<Rt: RuntimeType, P> Typ<Rt, P> {
        pub fn unwrap_poly(&self) -> &P {
            use Typ::*;
            match self {
                Poly(p) => p,
                _ => panic!("called unwrap_poly on non-poly type"),
            }
        }

        pub fn try_unwrap_poly(&self) -> Option<&P> {
            use Typ::*;
            match self {
                Poly(p) => Some(p),
                _ => None,
            }
        }

        pub fn try_unwrap_point_base(&self) -> Option<u32> {
            use Typ::*;
            match self {
                PointBase { log_n } => Some(*log_n),
                _ => None,
            }
        }

        pub fn hashable(&self) -> bool {
            use Typ::*;
            match self {
                Scalar | Poly(..) | Point => true,
                _ => false,
            }
        }

        pub fn try_unwrap_tuple(&self) -> Option<&Vec<Typ<Rt, P>>> {
            use Typ::*;
            match self {
                Tuple(ts) => Some(ts),
                _ => None,
            }
        }

        pub fn try_unwrap_array(&self) -> Option<(&Typ<Rt, P>, usize)> {
            use Typ::*;
            match self {
                Array(t, l) => Some((t.as_ref(), *l)),
                _ => None,
            }
        }

        pub fn unwrap_array(&self) -> (&Typ<Rt, P>, usize) {
            use Typ::*;
            match self {
                Array(t, l) => (t.as_ref(), *l),
                _ => panic!("called unwrap_array on non-array type"),
            }
        }
    }
}

pub type Typ<Rt: RuntimeType> = template::Typ<Rt, (PolyType, u64)>;

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

    pub fn iter<'s>(&'s self) -> Box<dyn Iterator<Item = &'s u64> + 's> {
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
    pub fn lagrange(deg: u64) -> Self {
        Typ::Poly((PolyType::Lagrange, deg))
    }

    pub fn coef(deg: u64) -> Self {
        Typ::Poly((PolyType::Coef, deg))
    }

    pub fn size(&self) -> Size {
        use template::Typ::*;
        match self {
            Poly((_, deg)) => Size::Single(*deg * size_of::<Rt::Field>() as u64),
            PointBase { log_n } => Size::Single((1 << log_n) * 2 * size_of::<Rt::Field>() as u64),
            Scalar => Size::Single(size_of::<Rt::Field>() as u64),
            Transcript => Size::Single(size_of::<Rt::Trans>() as u64),
            Point => Size::Single(2 * size_of::<Rt::Field>() as u64),
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

    pub fn iter<'s>(&'s self) -> Box<dyn Iterator<Item = &'s Self> + 's> {
        match self {
            Typ::Tuple(ts) => Box::new(ts.iter()),
            Typ::Array(t, n) => Box::new(std::iter::repeat(t.as_ref()).take(*n)),
            _ => Box::new(std::iter::once(self)),
        }
    }

    pub fn stack_allocable(&self) -> bool {
        use template::Typ::*;
        match self {
            Tuple(..) | Array(..) => true,
            _ => false,
        }
    }
}
