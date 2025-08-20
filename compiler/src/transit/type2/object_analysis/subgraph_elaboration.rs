//! Freshly created sliceable subgraph still operates on complete polynomials.
//! Here we must tailor them for operating on polynomial slices.

mod prelude {
    pub use std::collections::BTreeMap;
    pub use std::ops::{Deref, DerefMut};

    pub use crate::transit::type2;
    pub use crate::transit::type3;

    pub use super::super::OperationSeq;
    pub use super::super::{
        lower_typ,
        template::{ChunkedNode, ChunkedOp, MemoryOp, ResidentalAtom, SubgraphOperation},
        value::{self, ObjectId},
    };
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
    chunk_size: u64,
    deg: u64,
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
    fn new(size: u64, deg: u64) -> Self {
        Self {
            chunk_size: size,
            deg,
        }
    }

    fn output_range(&self) -> Slice {
        Slice::new(0, self.chunk_size)
    }

    /// Return the smallest slice containing `slices`, and with begin and end aligned to chunk boundaries
    fn including_chunks(&self, slices: impl Iterator<Item = Slice>) -> Slice {
        let including = self.including_slice(slices);
        let (min, max) = (including.begin(), including.end());

        let start = min / self.chunk_size * self.chunk_size;
        let len = crate::utils::div_ceil_u64(max - start, self.chunk_size) * self.chunk_size;
        Slice::new(start, len)
    }

    /// Return the smalles slice containing `slices`
    fn including_slice(&self, slices: impl Iterator<Item = Slice>) -> Slice {
        let slices: Vec<Slice> = slices.collect();

        let begin1 = slices
            .iter()
            .map(|s| s.begin())
            .max()
            .expect("no slice given");
        let begin2 = slices.iter().map(|s| s.begin()).min().unwrap();

        let len1 = slices
            .iter()
            .map(|s| (s.end() as i64 - begin1 as i64).rem_euclid(self.deg as i64) as u64)
            .max()
            .unwrap();
        let len2 = slices
            .iter()
            .map(|s| (s.end() as i64 - begin2 as i64).rem_euclid(self.deg as i64) as u64)
            .max()
            .unwrap();

        [Slice::new(begin1, len1), Slice::new(begin2, len2)]
            .iter()
            .min_by_key(|s| s.len())
            .unwrap()
            .clone()
    }

    fn chunk_size(&self) -> u64 {
        self.chunk_size
    }

    fn rotate(&self, slice: &Slice, offset: i64) -> Slice {
        slice.rotated(offset, self.deg)
    }

    fn new_slice(&self, begin: u64, len: u64) -> Slice {
        Slice::new(begin % self.deg, len)
    }
}
