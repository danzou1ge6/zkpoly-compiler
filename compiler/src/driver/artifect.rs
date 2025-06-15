use std::{collections::HashMap, sync::Arc};

use zkpoly_memory_pool::{buddy_disk_pool::DiskMemoryPool, static_allocator::CpuStaticAllocator};
use zkpoly_runtime::{
    args::{self, RuntimeType},
    devices::instantizate_event_table,
};

use crate::transit::type2::object_analysis::size::IntegralSize;

use super::{type3, HardwareInfo};

#[derive(Clone)]
pub struct Artifect<Rt: RuntimeType> {
    pub(super) chunk: type3::lowering::Chunk<Rt>,
    pub(super) constant_table: args::ConstantTable<Rt>,
}

pub struct Pools {
    pub cpu: CpuStaticAllocator,
    pub gpu: HashMap<i32, zkpoly_cuda_api::mem::CudaAllocator>,
    pub disk: DiskMemoryPool,
}

impl Pools {
    pub fn on(hd_info: &HardwareInfo, memory_check: bool, max_block: usize) -> Self {
        Self {
            cpu: hd_info.cpu_allocator(memory_check),
            gpu: hd_info.gpu_allocators(memory_check),
            disk: hd_info.disk_allocator(max_block),
        }
    }
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

    pub fn create_pools(&self, hd_info: &HardwareInfo, memory_check: bool) -> Pools {
        let max_lbs = self.chunk.lbss.max();
        let max_bs: usize = max_lbs.into();

        Pools::on(hd_info, memory_check, max_bs)
    }

    pub fn prepare_dispatcher(
        self,
        pools: Pools,
        rng: zkpoly_runtime::async_rng::AsyncRng,
        gpu_mapping: Arc<dyn Fn(i32) -> i32 + Send + Sync>,
    ) -> zkpoly_runtime::runtime::Runtime<Rt> {
        // self.chunk = self.chunk.adjust_gpu_device_ids(gpu_offset);
        zkpoly_runtime::runtime::Runtime::new(
            self.chunk.instructions,
            self.chunk.n_variables,
            self.constant_table,
            self.chunk.f_table,
            instantizate_event_table(self.chunk.event_table, gpu_mapping.clone()),
            self.chunk.n_threads,
            pools.cpu,
            pools.gpu,
            pools.disk,
            rng,
            gpu_mapping,
            self.chunk.libs,
        )
    }
}
