use crate::transit::type2::memory_planning::prelude::*;

pub mod gpu_allocator;
pub mod regretting_integral;
pub mod smithereens_allocator;
pub mod super_allocator;
pub mod constant_pool;
pub mod page_allocator;
pub mod smithereens_wrapper;

pub mod cpu_allocator {
    pub use super::super_allocator::SuperAllocator as CpuAllocator;
}


pub use gpu_allocator::GpuAllocator;
pub use super_allocator::SuperAllocator;
pub use cpu_allocator::CpuAllocator;
pub use constant_pool::ConstantPool;

struct OffsettedAddrMapping<'a> {
    mapping: &'a mut AddrMapping,
    offset: u64,
}

impl<'a> OffsettedAddrMapping<'a> {
    pub fn new(mapping: &'a mut AddrMapping, offset: u64) -> Self {
        Self { mapping, offset }
    }
}

impl<'a> AddrMappingHandler for OffsettedAddrMapping<'a> {
    fn add(&mut self, addr: Addr, size: Size) -> AddrId {
        self.mapping.push((addr.offset(self.offset), size))
    }

    fn update(&mut self, id: AddrId, addr: Addr, size: Size) {
        self.mapping[id] = (addr.offset(self.offset), size)
    }

    fn get(&self, id: AddrId) -> (Addr, Size) {
        let (addr, size) = self.mapping[id];
        (addr.unoffset(self.offset), size)
    }
}

trait AddrMappingHandler {
    fn get(&self, id: AddrId) -> (Addr, Size);
    fn update(&mut self, id: AddrId, addr: Addr, size: Size);
    fn add(&mut self, addr: Addr, size: Size) -> AddrId;
}

const SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD: u64 = 2u64.pow(16);
const LOG_MIN_INTEGRAL_SIZE: u32 = 10;

pub fn normalize_size(size: Size) -> Size {
    match size {
        Size::Integral(is) => {
            if is.0 < LOG_MIN_INTEGRAL_SIZE {
                Size::Smithereen(SmithereenSize(2u64.pow(is.0)))
            } else {
                Size::Integral(is)
            }
        }
        Size::Smithereen(ss) => {
            if let Ok(is) = IntegralSize::try_from(ss) {
                if is.0 < LOG_MIN_INTEGRAL_SIZE {
                    Size::Smithereen(SmithereenSize(2u64.pow(is.0)))
                } else {
                    Size::Integral(is)
                }
            } else if ss.0 >= SMITHEREEN_CEIL_TO_INTEGRAL_THRESHOLD {
                Size::Integral(IntegralSize::ceiling(ss))
            } else {
                Size::Smithereen(ss)
            }
        }
    }
}

pub fn collect_integral_sizes(sizes: impl Iterator<Item = Size>) -> Vec<IntegralSize> {
    let mut integral_sizes = BTreeSet::<IntegralSize>::new();
    for size in sizes {
        if let Size::Integral(is) = normalize_size(size) {
            integral_sizes.insert(is);
        }
    }

    integral_sizes.into_iter().collect()
}
