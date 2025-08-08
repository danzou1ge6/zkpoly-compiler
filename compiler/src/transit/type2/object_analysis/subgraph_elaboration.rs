//! Freshly created sliceable subgraph still operates on complete polynomials.
//! Here we must tailor them for operating on polynomial slices.

mod prelude {
    pub use std::collections::BTreeMap;
    pub use std::collections::BTreeSet;
    pub use std::ops::Deref;
    pub use std::ops::DerefMut;

    pub use crate::transit::type2;
    pub use crate::transit::type3;

    pub use super::super::{
        lower_typ,
        value::{self, ObjectId, OutputValue, VertexInput, VertexOutput},
    };
    pub use super::super::{Operation, OperationSeq};
    pub use sliceable_subgraph::{Vertex, VertexId, VertexNode};
    pub use type2::sliceable_subgraph;
    pub use type2::Device;
    pub use type2::{template::SliceableNode, ConstantId};
    pub use zkpoly_common::{arith::Mutability, heap::IdAllocator};
    pub use zkpoly_common::{
        heap::Heap,
        typ::{Slice, Typ},
    };
    pub use zkpoly_runtime::args::RuntimeType;
}

use prelude::*;

mod body;
mod prologue;
mod slice_analysis;

struct Chunker {
    size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ChunkNumber(u64);

impl Deref for ChunkNumber {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Chunker {
    fn new(size: u64) -> Self {
        Self { size }
    }

    fn output_range(&self) -> Slice {
        Slice::new(0, self.size)
    }

    fn chunks_for(&self, slices: impl Iterator<Item = Slice>) -> BTreeSet<ChunkNumber> {
        slices.fold(BTreeSet::new(), |mut acc, slice| {
            let start_chunk = slice.begin() / self.size;
            let end_chunk = (slice.end() - 1) / self.size;
            for c in start_chunk..=end_chunk {
                acc.insert(ChunkNumber(c));
            }
            acc
        })
    }

    fn including_slice(&self, slices: impl Iterator<Item = Slice>) -> Slice {
        let mut min = self.size;
        let mut max = 0;
        for slice in slices {
            min = min.min(slice.begin());
            max = max.max(slice.end());
        }

        if min >= max {
            panic!("empty slices or ill-formed slices");
        }

        let start = min / self.size * self.size;
        let len = crate::utils::div_ceil_u64(max - start, self.size) * self.size;
        Slice::new(start, len)
    }
}
