use zkpoly_memory_pool::static_allocator::CpuStaticAllocator;
use zkpoly_runtime::{
    args::{self, RuntimeType},
    devices::instantizate_event_table,
};

use super::type3;

pub struct Artifect<Rt: RuntimeType> {
    pub(super) chunk: type3::lowering::Chunk<Rt>,
    pub(super) constant_table: args::ConstantTable<Rt>,
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
        mut self,
        cpu_allocator: CpuStaticAllocator,
        gpu_allocator: Vec<zkpoly_cuda_api::mem::CudaAllocator>,
        rng: zkpoly_runtime::async_rng::AsyncRng,
        gpu_offset: i32,
    ) -> zkpoly_runtime::runtime::Runtime<Rt> {
        self.chunk = self.chunk.adjust_gpu_device_ids(gpu_offset);
        zkpoly_runtime::runtime::Runtime::new(
            self.chunk.instructions,
            self.chunk.n_variables,
            self.constant_table,
            self.chunk.f_table,
            instantizate_event_table(self.chunk.event_table),
            self.chunk.n_threads,
            cpu_allocator,
            gpu_allocator,
            vec![], // TODO: disk allocators
            vec![], // TODO: page allocators
            rng,
            self.chunk.libs,
        )
    }
}
