use super::fresh_type3::FreshType3;
use std::io::Write;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_memory_pool::CpuMemoryPool;
use zkpoly_runtime::args::{self, RuntimeType};

use super::{
    ast, check_type2_dag, debug_partial_typed_type2, debug_type2, debug_type2_def_use,
    debug_type2_with_seq, type2, type3, DebugOptions, Error, HardwareInfo, PanicJoinHandler,
    SubDigraph,
};

pub struct Artifect<Rt: RuntimeType> {
    pub(super) chunk: type3::lowering::Chunk<Rt>,
    pub(super) constant_table: args::ConstantTable<Rt>,
    pub(super) allocator: CpuMemoryPool,
}

impl<Rt: RuntimeType> Artifect<Rt> {
    pub fn dump(&self, dir: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::create_dir_all(dir.as_ref())?;

        let mut chunk_f = std::fs::File::create(dir.as_ref().join("chunk.json"))?;
        serde_json::to_writer_pretty(&mut chunk_f, &self.chunk)?;

        let mut ct_header_f = std::fs::File::create(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header = args::serialization::Header::build(&self.constant_table);
        serde_json::to_writer_pretty(&mut ct_header_f, &ct_header)?;

        let mut ct_f = std::fs::File::create(dir.as_ref().join("constants.bin"))?;
        ct_header.dump_entries_data(&self.constant_table, &mut ct_f)?;

        Ok(())
    }

    pub fn prepare_dispatcher(
        self,
        gpu_allocator: Vec<zkpoly_cuda_api::mem::CudaAllocator>,
        rng: zkpoly_runtime::async_rng::AsyncRng,
    ) -> zkpoly_runtime::runtime::Runtime<Rt> {
        zkpoly_runtime::runtime::Runtime::new(
            self.chunk.instructions,
            self.chunk.n_variables,
            self.constant_table,
            self.chunk.f_table,
            self.chunk.event_table,
            self.chunk.n_threads,
            self.allocator,
            gpu_allocator,
            rng,
            self.chunk.libs,
        )
    }

    pub fn allocator(&mut self) -> &mut CpuMemoryPool {
        &mut self.allocator
    }
}
