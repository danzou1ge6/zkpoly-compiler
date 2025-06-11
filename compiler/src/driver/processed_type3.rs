use super::artifect::Artifect;
use std::io::Write;
use zkpoly_memory_pool::CpuMemoryPool;
use zkpoly_runtime::args::RuntimeType;

use super::{
    cudaDeviceSynchronize, cuda_check, type2, type3, DebugOptions, Error, HardwareInfo,
    PanicJoinHandler,
};

pub struct ProcessedType3<'s, Rt: RuntimeType> {
    pub(super) chunk: type3::Chunk<'s, Rt>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) allocator: CpuMemoryPool,
}

impl<'s, Rt: RuntimeType> ProcessedType3<'s, Rt> {
    pub fn to_artifect(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        _ctx: &PanicJoinHandler,
    ) -> Result<(Artifect<Rt>, CpuMemoryPool), Error<'s, Rt>> {
        let Self {
            chunk: t3chunk,
            uf_table: t2uf_tab,
            constant_table: t2const_tab,
            allocator,
        } = self;

        // - Track Splitting
        let track_tasks = options.log_suround(
            "Splitting tracks",
            || Ok(type3::track_splitting::split(&t3chunk, hardware_info)),
            "Done.",
        )?;

        if options.debug_track_splitting {
            let mut f =
                std::fs::File::create(options.debug_dir.join("type3_track_splitting.txt")).unwrap();
            write!(f, "{:?}", &track_tasks).unwrap();
        }

        // To Runtime Instructions

        // - Emitting Multithread Chunk
        let (mt_chunk, f_table, event_table, stream2variable_id, variable_id_allocator, libs) =
            options.log_suround(
                "Emitting Multithread Chunk",
                || {
                    Ok(type3::lowering::emit_multithread_instructions(
                        &track_tasks,
                        t3chunk,
                        t2uf_tab,
                    ))
                },
                "Done.",
            )?;

        if options.debug_multithread_instructions {
            let path = options.debug_dir.join("multithread_instructions.html");
            let mut f = std::fs::File::create(&path).unwrap();
            type3::lowering::pretty_print::print(&mt_chunk, &stream2variable_id, &f_table, &mut f)
                .unwrap();
        }

        // - Serialize Multithread Chunk
        let rt_chunk = options.log_suround(
            "Lowering Type3 to Runtime Instructions",
            || {
                Ok(type3::lowering::lower(
                    mt_chunk,
                    f_table,
                    event_table,
                    stream2variable_id,
                    variable_id_allocator,
                    libs,
                ))
            },
            "Done.",
        )?;

        if options.debug_instructions {
            let mut f = std::fs::File::create(options.debug_dir.join("instructions.txt")).unwrap();
            zkpoly_runtime::instructions::print_instructions(&rt_chunk.instructions, &mut f)
                .unwrap();
        }
        unsafe {
            cuda_check!(cudaDeviceSynchronize());
        }

        Ok((Artifect {
            chunk: rt_chunk,
            constant_table: t2const_tab,
        }, allocator))
    }
}
