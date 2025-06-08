use crate::transit::type2::memory_planning::prelude::*;


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

define_usize_id!(AddrId);

pub type AddrMapping = Heap<AddrId, (Addr, Size)>;
