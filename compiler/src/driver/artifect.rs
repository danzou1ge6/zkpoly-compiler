use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use zkpoly_common::{devices::DeviceType, heap::Heap, load_dynamic::Libs};
use zkpoly_memory_pool::{buddy_disk_pool::DiskMemoryPool, static_allocator::CpuStaticAllocator};
use zkpoly_runtime::{
    args::{self, move_constant_table, ConstantId, RuntimeType},
    devices::instantizate_event_table,
};

use crate::{driver::MemoryInfo, transit::type2};

use super::{type3, ConstantPool, Versions};

#[derive(Clone)]
pub(crate) struct Version<Rt: RuntimeType> {
    pub chunk: type3::lowering::Chunk<Rt>,
    pub statistics: type2::memory_planning::Statistics,
}

/// The compilation artifect ready to run by the runtime.
#[derive(Clone)]
pub struct Artifect<Rt: RuntimeType> {
    pub(super) versions: Versions<Version<Rt>>,
    pub(super) constant_table: args::ConstantTable<Rt>,
    pub(super) libs: Libs,
}

/// The compilation artifect almost ready to run by the runtime.
/// At this point, constants can be dumped to disk so [`SemiArtifect`]
/// can be serialized to file, whereas [`Artifect`] cannot.
pub struct SemiArtifect<Rt: RuntimeType> {
    pub(super) versions: Versions<Version<Rt>>,
    pub(super) constant_table: args::ConstantTable<Rt>,
    pub(super) constant_devices: Heap<ConstantId, type3::Device>,
    pub(super) libs: Libs,
}

impl<Rt: RuntimeType> SemiArtifect<Rt> {
    /// Finish the last step and obtain [`Artifect`].
    pub fn finish(self, constant_pool: &mut ConstantPool) -> Artifect<Rt> {
        // - Move Constants to where they should be
        let constant_table = move_constant_table(
            self.constant_table,
            &self
                .constant_devices
                .map_by_ref(&mut |_, t3t| DeviceType::from(t3t.clone())),
            &mut constant_pool.cpu,
            constant_pool.disk.as_mut(),
        );

        Artifect {
            versions: self.versions,
            constant_table,
            libs: self.libs,
        }
    }

    /// Write everything in the [`Artifect`] to file, including instructions and constants.
    /// To load the dump, use `load_artifect` method on [`super::FreshType2`].
    pub fn dump(
        &self,
        dir: impl AsRef<std::path::Path>,
        constant_pool: &mut ConstantPool,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(dir.as_ref())?;

        self.versions
            .dump(&dir, |f, cpu, Version { chunk, statistics }| {
                serde_json::to_writer_pretty(f, &(cpu, chunk, statistics))?;
                Ok(())
            })?;

        let mut ct_header_f = std::fs::File::create(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header = args::serialization::Header::build(&self.constant_table);
        serde_json::to_writer_pretty(&mut ct_header_f, &ct_header)?;

        let mut ct_f = std::fs::File::create(dir.as_ref().join("constants.bin"))?;
        ct_header.dump_entries_data(&self.constant_table, &mut ct_f, &mut constant_pool.cpu)?;

        Ok(())
    }
}

/// The memory pools on each memory device, used for running the artifect.
pub struct Pools {
    pub cpu: CpuStaticAllocator,
    pub gpu: HashMap<i32, zkpoly_cuda_api::mem::CudaAllocator>,
    pub disk: Arc<Mutex<DiskMemoryPool>>,
}

pub type GpuMapping = Arc<dyn Fn(i32) -> i32 + Send + Sync>;

impl<Rt: RuntimeType> Artifect<Rt> {
    /// The versions included in the artifect.
    /// Each version runs on different CPU memory usage.
    pub fn versions(&self) -> impl Iterator<Item = &MemoryInfo> {
        self.versions.iter()
    }

    /// The maximum block size that will be used if using a buddy allocator.
    pub fn max_bs(&self) -> usize {
        let max_lbs = self
            .versions
            .iter_items()
            .map(|(_, v)| v.chunk.lbss.max())
            .max()
            .expect("no versions");
        let max_bs: usize = max_lbs.into();

        max_bs
    }

    /// Make a kernel dispatcher (the runtime) to run this artifect.
    ///
    /// `version` selects the artifect version that runs on certain CPU memory usage;
    /// No requirements are imposed on `pools`' contents, it's only a scratchboard for one dispatcher run.
    pub fn prepare_dispatcher<'d>(
        &self,
        version: &MemoryInfo,
        pools: Pools,
        rng: zkpoly_runtime::async_rng::AsyncRng,
        gpu_mapping: GpuMapping,
    ) -> zkpoly_runtime::runtime::Runtime<Rt> {
        let version = self
            .versions
            .ref_of(version)
            .unwrap_or_else(|| panic!("no version found for {:?}", version));

        // self.chunk = self.chunk.adjust_gpu_device_ids(gpu_offset);
        zkpoly_runtime::runtime::Runtime::new(
            version.chunk.instructions.clone(),
            version.chunk.n_variables,
            self.constant_table.clone(),
            version.chunk.f_table.clone(),
            instantizate_event_table(version.chunk.event_table.clone(), gpu_mapping.clone()),
            version.chunk.n_threads,
            pools.cpu,
            pools.gpu,
            pools.disk,
            rng,
            gpu_mapping,
            self.libs.clone(),
        )
    }
}
