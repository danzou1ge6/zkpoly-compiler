
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Addr(pub(crate) u64);

impl Addr {
    pub fn offset(self, x: u64) -> Addr {
        Addr(self.0 + x)
    }

    pub fn unoffset(self, x: u64) -> Addr {
        Addr(self.0 - x)
    }

    pub fn get(self) -> usize {
        self.0 as usize
    }
}

