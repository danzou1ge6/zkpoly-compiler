use super::type3::template::GpuAddr;


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Addr(pub(crate) u64);

impl Addr {
    pub fn offset(self, x: u64) -> Addr {
        Addr(self.0 + x)
    }

    pub fn unoffset(self, x: u64) -> Addr {
        Addr(self.0 - x)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl From<Addr> for GpuAddr {
    fn from(value: Addr) -> Self {
        GpuAddr::Offset(value.0.into())
    }
}
