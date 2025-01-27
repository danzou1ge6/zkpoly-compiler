//! A module for planning memory allocations of 2's exponential sizes.

use super::{Addr, IntegralSize, Instant, AddrId, AddrMappingHandler, Size};

pub mod one_shot;
pub mod regretting;
