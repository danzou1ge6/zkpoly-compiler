use std::collections::BTreeMap;

use super::{processed_type3::ProcessedType3, type3};
use zkpoly_common::heap::Heap;
use zkpoly_memory_pool::CpuMemoryPool;
use zkpoly_runtime::args::{ConstantId, RuntimeType};

use super::{type2, DebugOptions, Error};

pub struct FreshType3<'s, Rt: RuntimeType> {
    pub(super) chunk: type3::Chunk<'s, Rt>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) allocator: CpuMemoryPool,
    pub(super) constants_device: Heap<ConstantId, type3::Device>,
    pub(super) execution_devices: BTreeMap<type2::VertexId, type2::Device>,
}

impl<'s, Rt: RuntimeType> FreshType3<'s, Rt> {
    pub fn apply_passes<'a>(
        self,
        options: &'a DebugOptions,
    ) -> Result<ProcessedType3<'s, Rt>, Error<'s, Rt>> {
        // - Extend Rewritting
        let t3chunk = options.log_suround(
            "Rewritting Extend and NewPoly",
            || Ok(type3::rewrite_extend::rewrite(self.chunk)),
            "Done.",
        )?;

        if options.debug_extend_rewriting {
            let mut f =
                std::fs::File::create(options.debug_dir.join("type3_extend_rewriting.html"))
                    .unwrap();
            type3::pretty_print::prettify(&t3chunk, |vid| self.execution_devices[&vid], &mut f)
                .unwrap();
        }

        Ok(ProcessedType3 {
            chunk: t3chunk,
            uf_table: self.uf_table,
            constant_table: self.constant_table,
            allocator: self.allocator,
            constants_device: self.constants_device,
            execution_devices: self.execution_devices,
        })
    }
}
