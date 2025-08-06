use super::artifect::SemiArtifect;
use std::{io::Write, path::PathBuf};
use zkpoly_common::{heap::Heap, load_dynamic::Libs};
use zkpoly_runtime::args::{ConstantId, RuntimeType};

use super::{
    artifect, cudaDeviceSynchronize, cuda_check, fresh_type3, type2, type3, DebugOptions, Error,
    HardwareInfo, Versions,
};

/// The Type3 IR after applying various Type3 passes.
pub struct ProcessedType3<'s, Rt: RuntimeType> {
    pub(super) versions: Versions<fresh_type3::Version<'s, Rt>>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) constants_device: Heap<ConstantId, type3::Device>,
    pub(super) libs: Libs,
}

impl<'s, Rt: RuntimeType> ProcessedType3<'s, Rt> {
    /// Run the final kernel generation and other passes to obtain [`SemiArtifect`].
    /// `kernel_dir` controls in which directory the compiler puts temporary source files
    /// and the compiled libraries.
    pub fn to_artifect(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        kernel_dir: PathBuf,
    ) -> Result<SemiArtifect<Rt>, Error<'s, Rt>> {
        let Self {
            versions,
            uf_table: t2uf_tab,
            constant_table: t2const_tab,
            constants_device,
            mut libs,
        } = self;

        let versions = versions.map(|cpu, version, helper| {
            let hardware_info = hardware_info.clone().with_cpu(cpu.clone());
            let hardware_info = &hardware_info;

            // - Track Splitting
            let track_tasks = options.log_suround(
                helper.log_prologue("Splitting tracks"),
                || Ok(type3::track_splitting::split(&version.chunk, hardware_info)),
                "Done.",
            )?;

            if options.debug_track_splitting {
                let mut f = std::fs::File::create(
                    options
                        .debug_dir
                        .join(helper.debug_filename("type3_track_splitting.txt")),
                )
                .unwrap();
                write!(f, "{:?}", &track_tasks).unwrap();
            }

            // To Runtime Instructions

            // - Emitting Multithread Chunk
            let (mt_chunk, f_table, event_table, stream2variable_id, variable_id_allocator, lbss) =
                options.log_suround(
                    helper.log_prologue("Emitting Multithread Chunk"),
                    || {
                        Ok(type3::lowering::emit_multithread_instructions(
                            &track_tasks,
                            version.chunk,
                            t2uf_tab.clone(),
                            &mut libs,
                            kernel_dir.join(helper.dirname("version")),
                        ))
                    },
                    "Done.",
                )?;

            if options.debug_multithread_instructions {
                let path = options
                    .debug_dir
                    .join(helper.debug_filename("multithread_instructions.html"));
                let mut f = std::fs::File::create(&path).unwrap();
                type3::lowering::pretty_print::print(
                    &mt_chunk,
                    &stream2variable_id,
                    &f_table,
                    &mut f,
                )
                .unwrap();
            }

            // - Serialize Multithread Chunk
            let rt_chunk = options.log_suround(
                helper.log_prologue("Lowering Type3 to Runtime Instructions"),
                || {
                    Ok(type3::lowering::lower(
                        mt_chunk,
                        f_table,
                        event_table,
                        stream2variable_id,
                        variable_id_allocator,
                        lbss,
                    ))
                },
                "Done.",
            )?;

            if options.debug_instructions {
                let mut f = std::fs::File::create(
                    options
                        .debug_dir
                        .join(helper.debug_filename("instructions.txt")),
                )
                .unwrap();
                zkpoly_runtime::instructions::print_instructions(&rt_chunk.instructions, &mut f)
                    .unwrap();
            }

            Ok(artifect::Version {
                chunk: rt_chunk,
                statistics: version.memory_statistics,
            })
        })?;

        unsafe {
            cuda_check!(cudaDeviceSynchronize());
        }

        Ok(SemiArtifect {
            versions,
            constant_table: t2const_tab,
            constant_devices: constants_device,
            libs,
        })
    }
}
