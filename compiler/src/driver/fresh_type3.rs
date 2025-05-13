use super::{processed_type3::ProcessedType3, type3};
use std::io::Write;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_runtime::args::RuntimeType;

use super::{
    ast, check_type2_dag, debug_partial_typed_type2, debug_type2, processed_type2::ProcessedType2,
    type2, DebugOptions, Error, HardwareInfo, PanicJoinHandler,
};

pub struct FreshType3<Rt: RuntimeType> {
    pub(super) chunk: type3::Chunk<'static, Rt>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) allocator: PinnedMemoryPool,
}

impl<Rt: RuntimeType> FreshType3<Rt> {
    pub fn apply_passes(
        self,
        options: &DebugOptions,
    ) -> Result<ProcessedType3<Rt>, Error<'static, Rt>> {
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
            type3::pretty_print::prettify(&t3chunk, &mut f).unwrap();
        }

        Ok(ProcessedType3 {
            chunk: t3chunk,
            uf_table: self.uf_table,
            constant_table: self.constant_table,
            allocator: self.allocator,
        })
    }
}
