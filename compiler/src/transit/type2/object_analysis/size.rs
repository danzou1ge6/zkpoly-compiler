use crate::utils::{log2, log2_ceil};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct IntegralSize(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmithereenSize(pub u64);

impl IntegralSize {
    pub fn double(self) -> Self {
        Self(self.0 + 1)
    }
}
impl From<IntegralSize> for SmithereenSize {
    fn from(size: IntegralSize) -> Self {
        Self(2u64.pow(size.0))
    }
}

impl TryFrom<SmithereenSize> for IntegralSize {
    type Error = ();
    fn try_from(value: SmithereenSize) -> Result<Self, Self::Error> {
        if let Some(l) = value.0.checked_ilog2() {
            if 2u64.pow(l) == value.0 {
                Ok(IntegralSize(l))
            } else {
                Err(())
            }
        } else {
            Err(())
        }
    }
}

impl TryFrom<Size> for IntegralSize {
    type Error = ();
    fn try_from(value: Size) -> Result<Self, Self::Error> {
        match value {
            Size::Integral(size) => Ok(size),
            Size::Smithereen(ss) => ss.try_into(),
        }
    }
}

impl IntegralSize {
    pub fn ceiling(size: SmithereenSize) -> Self {
        Self(log2_ceil(size.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    Integral(IntegralSize),
    Smithereen(SmithereenSize),
}

impl std::fmt::Display for Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Size::Integral(size) => write!(f, "2^{}", size.0),
            Size::Smithereen(size) => write!(f, "{}", size.0),
        }
    }
}
impl Size {
    pub fn new(s: u64) -> Self {
        let ss = SmithereenSize(s);
        if let Ok(is) = IntegralSize::try_from(ss) {
            Self::Integral(is)
        } else {
            Self::Smithereen(ss)
        }
    }

    pub fn unwrap_integral(self) -> IntegralSize {
        match self {
            Size::Integral(size) => size,
            Size::Smithereen(..) => panic!("unwrap_integral on Size::Smithereen"),
        }
    }
}

impl From<u64> for Size {
    fn from(size: u64) -> Self {
        Self::new(size)
    }
}

impl From<Size> for u64 {
    fn from(value: Size) -> Self {
        match value {
            Size::Integral(is) => 2u64.pow(is.0),
            Size::Smithereen(SmithereenSize(ss)) => ss,
        }
    }
}

impl std::ops::Div<u64> for Size {
    type Output = Size;
    fn div(self, rhs: u64) -> Self::Output {
        match self {
            Size::Integral(IntegralSize(is)) => {
                if let Some(log) = log2(rhs) {
                    Self::Integral(IntegralSize(is - log))
                } else {
                    panic!("can only divide by power of 2")
                }
            }
            Size::Smithereen(SmithereenSize(ss)) => Self::Smithereen(SmithereenSize(ss / rhs)),
        }
    }
}

impl std::ops::Mul<u64> for Size {
    type Output = Size;
    fn mul(self, rhs: u64) -> Self::Output {
        match self {
            Size::Integral(IntegralSize(is)) => {
                if let Some(log) = log2(rhs) {
                    Self::Integral(IntegralSize(is + log))
                } else {
                    panic!("can only multiply by power of 2")
                }
            }
            Size::Smithereen(SmithereenSize(ss)) => Self::Smithereen(SmithereenSize(ss * rhs)),
        }
    }
}
