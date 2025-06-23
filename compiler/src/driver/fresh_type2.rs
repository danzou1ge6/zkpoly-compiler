use std::io::{Read, Write};
use zkpoly_common::load_dynamic::Libs;
use zkpoly_memory_pool::{buddy_disk_pool::DiskMemoryPool, CpuMemoryPool};
use zkpoly_runtime::args::{self, RuntimeType};

use super::{
    artifect::{Artifect, SemiArtifect},
    ast, check_type2_dag, debug_partial_typed_type2, debug_type2,
    processed_type2::ProcessedType2,
    type2, type3, DebugOptions, Error, HardwareInfo, PanicJoinHandler,
};

pub struct FreshType2<'s, Rt: RuntimeType> {
    prog: type2::Program<'s, Rt>,
}

impl<'s, Rt: RuntimeType> FreshType2<'s, Rt> {
    pub fn from_ast(
        ast: impl ast::TypeEraseable<Rt>,
        options: &DebugOptions,
        allocator: CpuMemoryPool,
        ctx: &PanicJoinHandler,
    ) -> Result<Self, Error<'s, Rt>> {
        let (ast_cg, output_vid) = options.log_suround(
            "Lowering AST to Type2...",
            || Ok(ast::lowering::Cg::new(ast, allocator)),
            "Done.",
        )?;

        if options.debug_fresh_type2 {
            ctx.add(debug_type2(
                options.debug_dir.join("type2_fresh.dot"),
                &ast_cg.g,
                output_vid,
                options.type2_visualizer,
            ));
        }

        if options.debug_user_function_table {
            let mut f =
                std::fs::File::create(&options.debug_dir.join("user_functions.txt")).unwrap();
            write!(f, "{:#?}", ast_cg.user_function_table).unwrap();
        }

        // Type inference
        let t2prog =
            match options.log_suround("Inferring Type...", || ast_cg.lower(output_vid), "Done.") {
                Ok(t2prog) => Ok(t2prog),
                Err((e, g)) => {
                    let fpath = options.debug_dir.join("type2_type_inference.dot");
                    debug_partial_typed_type2(
                        fpath,
                        &g,
                        e.vid,
                        output_vid,
                        options.type2_visualizer,
                    )
                    .map(|j| j.join().unwrap());

                    Err(Error::Typ(e))
                }
            }?;

        if !check_type2_dag(
            options.debug_dir.join("type2_type_inference.dot"),
            &t2prog.cg.g,
            output_vid,
            options.type2_visualizer,
        ) {
            return Err(Error::NotDag);
        }
        Ok(Self { prog: t2prog })
    }

    pub fn apply_passes(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        ctx: &PanicJoinHandler,
        disk_allocator: &mut DiskMemoryPool
    ) -> Result<ProcessedType2<'s, Rt>, Error<'static, Rt>> {
        let type2::Program {
            cg: t2cg,
            consant_table: mut t2const_tab,
            user_function_table: t2uf_tab,
            memory_pool: mut allocator,
        } = self.prog;

        let mut libs = Libs::new();

        if options.debug_type_inference {
            ctx.add(debug_type2(
                options.debug_dir.join("type2_type_inference.dot"),
                &t2cg.g,
                t2cg.output,
                options.type2_visualizer,
            ));
        }

        // Apply Type2 passes
        // - INTT Mending: Append division by n to end of each INTT
        let t2cg = options.log_suround(
            "Applying INTT Mending...",
            || Ok(type2::intt_mending::mend(t2cg, &mut t2const_tab)),
            "Done.",
        )?;

        if !check_type2_dag(
            options.debug_dir.join("type2_intt_mending.dot"),
            &t2cg.g,
            t2cg.output,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after INTT Mending");
        }

        if options.debug_intt_mending {
            ctx.add(debug_type2(
                options.debug_dir.join("type2_intt_mending.dot"),
                &t2cg.g,
                t2cg.output,
                options.type2_visualizer,
            ));
        }

        // - Precompute NTT and MSM constants
        let t2cg = options.log_suround(
            "Precomputing constants for NTT and MSM",
            || {
                Ok(type2::precompute::precompute(
                    t2cg,
                    hardware_info.smallest_gpu_memory_integral_limit() as usize,
                    &mut libs,
                    &mut allocator,
                    disk_allocator,
                    &mut t2const_tab,
                    hardware_info.disk_available()
                ))
            },
            "Done.",
        )?;

        if !check_type2_dag(
            options.debug_dir.join("type2_precompute.dot"),
            &t2cg.g,
            t2cg.output,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after Precomputing");
        }

        if options.debug_precompute {
            ctx.add(debug_type2(
                options.debug_dir.join("type2_precompute.dot"),
                &t2cg.g,
                t2cg.output,
                options.type2_visualizer,
            ));
        }

        // - Manage Inversions: Rewrite inversions of scalars and polynomials to dedicated operators
        let t2cg = options.log_suround(
            "Managing inversions",
            || Ok(type2::manage_inverse::manage_inverse(t2cg)),
            "Done.",
        )?;

        if !check_type2_dag(
            options.debug_dir.join("type2_manage_invers.dot"),
            &t2cg.g,
            t2cg.output,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after Managing Inversions");
        }

        if options.debug_manage_invers {
            ctx.add(debug_type2(
                options.debug_dir.join("type2_manage_invers.dot"),
                &t2cg.g,
                t2cg.output,
                options.type2_visualizer,
            ));
        }

        // - Common Subexpression Elimination
        let t2cg = options.log_suround(
            "Eliminating common subexpressions",
            || Ok(type2::common_subexpression_elimination::cse(t2cg)),
            "Done.",
        )?;

        if !check_type2_dag(
            options.debug_dir.join("type2_cse.dot"),
            &t2cg.g,
            t2cg.output,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after CSE");
        }

        if options.debug_cse {
            ctx.add(debug_type2(
                options.debug_dir.join("type2_cse.dot"),
                &t2cg.g,
                t2cg.output,
                options.type2_visualizer,
            ));
        }

        // - Arithmetic Kernel Fusion
        let t2cg = options.log_suround(
            "Fusing arithmetic kernels",
            || Ok(type2::kernel_fusion::fuse_arith(t2cg, &hardware_info)),
            "Done.",
        )?;

        if !check_type2_dag(
            options.debug_dir.join("type2_kernel_fusion.dot"),
            &t2cg.g,
            t2cg.output,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after Arithmetic Kernel Fusion");
        }

        if options.debug_kernel_fusion {
            ctx.add(debug_type2(
                options.debug_dir.join("type2_kernel_fusion.dot"),
                &t2cg.g,
                t2cg.output,
                options.type2_visualizer,
            ));
        }

        Ok(ProcessedType2 {
            cg: t2cg,
            constant_table: t2const_tab,
            uf_table: t2uf_tab,
            libs,
            allocator,
        })
    }

    pub fn load_artifect(
        mut self,
        dir: impl AsRef<std::path::Path>,
        disk_allocator: &mut DiskMemoryPool,
    ) -> std::io::Result<(Artifect<Rt>, CpuMemoryPool)> {
        let mut chunk_f = std::fs::File::open(dir.as_ref().join("chunk.json"))?;
        let rt_chunk_deserializer: type3::lowering::serialization::ChunkDeserializer =
            serde_json::from_reader(&mut chunk_f)?;
        let rt_chunk = rt_chunk_deserializer.deserialize_into_chunk(self.prog.user_function_table);

        let mut ct_header_f = std::fs::File::open(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header: args::serialization::Header = serde_json::from_reader(&mut ct_header_f)?;

        let mut ct_f = std::fs::File::open(dir.as_ref().join("constants.bin"))?;
        let mut allocator = self.prog.memory_pool;
        ct_header.load_constant_table(
            &mut self.prog.consant_table,
            &mut ct_f,
            &mut allocator,
            disk_allocator,
        )?;

        Ok((
            Artifect {
                chunk: rt_chunk,
                constant_table: self.prog.consant_table,
            },
            allocator,
        ))
    }

    pub fn load_processed_type2<'de>(
        mut self,
        str_buf: &'de mut String,
        dir: impl AsRef<std::path::Path>,
        disk_allocator: &mut DiskMemoryPool,
    ) -> std::io::Result<ProcessedType2<'de, Rt>>
    where
        Rt: serde::Serialize + serde::Deserialize<'de>,
    {
        let mut cg_f = std::fs::File::open(dir.as_ref().join("cg.json"))?;
        cg_f.read_to_string(str_buf)?;
        let cg: type2::Cg<'de, Rt> = serde_json::from_str(str_buf)?;

        let mut ct_header_f = std::fs::File::open(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header: args::serialization::Header = serde_json::from_reader(&mut ct_header_f)?;

        let mut ct_f = std::fs::File::open(dir.as_ref().join("constants.bin"))?;
        let mut allocator = self.prog.memory_pool;
        ct_header.load_constant_table(
            &mut self.prog.consant_table,
            &mut ct_f,
            &mut allocator,
            disk_allocator,
        )?;

        Ok(ProcessedType2 {
            cg,
            constant_table: self.prog.consant_table,
            uf_table: self.prog.user_function_table,
            libs: Libs::new(),
            allocator,
        })
    }

    pub fn to_artifect(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        disk_allocator: &mut DiskMemoryPool,
        ctx: &PanicJoinHandler,
    ) -> Result<(Artifect<Rt>, CpuMemoryPool), Error<'s, Rt>> {
        Ok(self
            .to_semi_artifect(options, hardware_info, disk_allocator , ctx)?
            .finish(disk_allocator))
    }

    pub fn to_semi_artifect(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        disk_allocator: &mut DiskMemoryPool,
        ctx: &PanicJoinHandler,
    ) -> Result<SemiArtifect<Rt>, Error<'s, Rt>> {
        self.apply_passes(options, hardware_info, ctx, disk_allocator)?
            .to_type3(options, hardware_info, ctx)?
            .apply_passes(options)?
            .to_artifect(options, hardware_info, ctx)
    }
}
