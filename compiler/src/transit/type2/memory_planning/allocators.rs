use crate::transit::type2::memory_planning::prelude::*;

pub mod slab_allocator;
pub mod slab_pool;
pub mod smithereens_pool;
pub mod super_allocator;
pub mod constant_wrapper;
pub mod page_allocator;
pub mod smithereens_wrapper;

pub use slab_allocator::SlabAllocator;
pub use super_allocator::SuperAllocator;
pub use constant_wrapper::Wrapper as ConstantWrapper;
pub use smithereens_wrapper::Wrapper as SmithereenWrapper;

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
