use std::{io::Write, path::PathBuf};
use zkpoly_common::load_dynamic::Libs;
use zkpoly_runtime::args::{self, RuntimeType};

use crate::driver::artifect;

use super::{
    artifect::{Artifect, SemiArtifect},
    ast, check_type2_unsubgraphed_dag,
    processed_type2::ProcessedType2,
    type2::{self, pretty::unsubgraphed_cg::debug_graph},
    type3,
    unfused_type2::UnfusedType2,
    ConstantPool, DebugOptions, Error, HardwareInfo, MemoryInfo, PanicJoinHandler, Versions,
};

/// Type2 intermediate representation built from AST.
pub struct FreshType2<'s, Rt: RuntimeType> {
    prog: type2::Program<'s, Rt>,
}

impl<'s, Rt: RuntimeType> FreshType2<'s, Rt> {
    /// Build from AST. Runs type inference here.
    pub fn from_ast(
        ast: impl ast::TypeEraseable<Rt>,
        options: &DebugOptions,
        _ctx: &PanicJoinHandler,
    ) -> Result<Self, Error<'s, Rt>> {
        let (ast_cg, output_vid) = options.log_suround(
            "Lowering AST to Type2...",
            || Ok(ast::lowering::Cg::new(ast)),
            "Done.",
        )?;

        if options.debug_fresh_type2 {
            type2::pretty::partial_typed::debug_graph(
                &ast_cg.g,
                options.debug_dir.join("type2_fresh.dot"),
            );
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
                    type2::pretty::type_error::debug_graph(&g, e.vid, &fpath)
                        .expect("write debug file error");

                    Err(Error::Typ(e))
                }
            }?;

        if !check_type2_unsubgraphed_dag(
            options.debug_dir.join("type2_type_inference.dot"),
            &t2prog.cg,
            options.type2_visualizer,
        ) {
            return Err(Error::NotDag);
        }
        Ok(Self { prog: t2prog })
    }

    /// Apply passes to obtain [`UnfusedType2`].
    ///
    /// This method needs mutable access to `constant_pool`, because due to precomputation for MSM and NTT
    /// new constants may be added.
    pub fn apply_passes(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        constant_pool: &mut ConstantPool,
        _ctx: &PanicJoinHandler,
    ) -> Result<UnfusedType2<'s, Rt>, Error<'static, Rt>> {
        let type2::Program {
            cg: t2cg,
            consant_table: mut t2const_tab,
            user_function_table: t2uf_tab,
        } = self.prog;

        let mut libs = Libs::new();

        if options.debug_type_inference {
            debug_graph(&t2cg, options.debug_dir.join("type2_type_inference.dot"));
        }

        // Apply Type2 passes
        // - INTT Mending: Append division by n to end of each INTT
        let t2cg = options.log_suround(
            "Applying INTT Mending...",
            || Ok(type2::intt_mending::mend(t2cg, &mut t2const_tab)),
            "Done.",
        )?;

        if !check_type2_unsubgraphed_dag(
            options.debug_dir.join("type2_intt_mending.dot"),
            &t2cg,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after INTT Mending");
        }

        if options.debug_intt_mending {
            debug_graph(&t2cg, options.debug_dir.join("type2_intt_mending.dot"));
        }

        // - Precompute NTT and MSM constants
        let t2cg = options.log_suround(
            "Precomputing constants for NTT and MSM",
            || {
                Ok(type2::precompute::precompute(
                    t2cg,
                    hardware_info.smallest_gpu_memory_integral_limit() as usize,
                    &mut libs,
                    &mut constant_pool.cpu,
                    constant_pool.disk.as_mut(),
                    &mut t2const_tab,
                ))
            },
            "Done.",
        )?;

        if !check_type2_unsubgraphed_dag(
            options.debug_dir.join("type2_precompute.dot"),
            &t2cg,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after Precomputing");
        }

        if options.debug_precompute {
            debug_graph(&t2cg, options.debug_dir.join("type2_precompute.dot"));
        }

        // - Manage Inversions: Rewrite inversions of scalars and polynomials to dedicated operators
        let t2cg = options.log_suround(
            "Managing inversions",
            || Ok(type2::manage_inverse::manage_inverse(t2cg)),
            "Done.",
        )?;

        if !check_type2_unsubgraphed_dag(
            options.debug_dir.join("type2_manage_invers.dot"),
            &t2cg,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after Managing Inversions");
        }

        if options.debug_manage_invers {
            debug_graph(&t2cg, options.debug_dir.join("type2_manage_invers.dot"));
        }

        // - Common Subexpression Elimination
        let t2cg = options.log_suround(
            "Eliminating common subexpressions",
            || {
                Ok(type2::common_subexpression_elimination::cse(
                    t2cg, &t2uf_tab,
                ))
            },
            "Done.",
        )?;

        if !check_type2_unsubgraphed_dag(
            options.debug_dir.join("type2_cse.dot"),
            &t2cg,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after CSE");
        }

        if options.debug_cse {
            debug_graph(&t2cg, options.debug_dir.join("type2_cse.dot"));
        }

        Ok(UnfusedType2 {
            cg: t2cg,
            constant_table: t2const_tab,
            uf_table: t2uf_tab,
            libs,
        })
    }

    /// Base on fresh Type2, directly load [`Artifect`] from file system without need for further passes.
    /// During loading, stored constants are written to `constant_pool`.
    pub fn load_artifect(
        mut self,
        dir: impl AsRef<std::path::Path>,
        constant_pool: &mut ConstantPool,
    ) -> std::io::Result<Artifect<Rt>> {
        let mut libs = Libs::new();

        let versions = Versions::load(&dir, |f| {
            let (cpu, chunk, statistics): (
                MemoryInfo,
                type3::lowering::serialization::ChunkDeserializer,
                type2::memory_planning::Statistics,
            ) = serde_json::from_reader(f)?;
            let chunk =
                chunk.deserialize_into_chunk(self.prog.user_function_table.clone(), &mut libs);
            Ok((cpu, artifect::Version { chunk, statistics }))
        })?;

        let mut ct_header_f = std::fs::File::open(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header: args::serialization::Header = serde_json::from_reader(&mut ct_header_f)?;

        let mut ct_f = std::fs::File::open(dir.as_ref().join("constants.bin"))?;
        ct_header.load_constant_table(
            &mut self.prog.consant_table,
            &mut ct_f,
            &mut constant_pool.cpu,
            constant_pool.disk.as_mut(),
        )?;

        Ok(Artifect {
            versions,
            constant_table: self.prog.consant_table,
            libs,
        })
    }

    pub fn load_processed_type2<'de>(
        mut self,
        dir: impl AsRef<std::path::Path>,
        buffers: &'de mut Vec<String>,
        constant_pool: &mut ConstantPool,
    ) -> std::io::Result<ProcessedType2<'de, Rt>>
    where
        Rt: serde::Serialize + serde::Deserialize<'de>,
    {
        let cg = Versions::load_buffered(&dir, buffers, |buf| {
            let (cpu, cg): (MemoryInfo, type2::Cg<'de, Rt>) = serde_json::from_str(buf)?;
            Ok((cpu, cg))
        })?;

        let mut ct_header_f = std::fs::File::open(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header: args::serialization::Header = serde_json::from_reader(&mut ct_header_f)?;

        let mut ct_f = std::fs::File::open(dir.as_ref().join("constants.bin"))?;
        ct_header.load_constant_table(
            &mut self.prog.consant_table,
            &mut ct_f,
            &mut constant_pool.cpu,
            constant_pool.disk.as_mut(),
        )?;

        Ok(ProcessedType2 {
            cg,
            constant_table: self.prog.consant_table,
            uf_table: self.prog.user_function_table,
            libs: Libs::new(),
        })
    }

    /// Apply all passes to obtain [`Artifect`].
    ///
    /// For meanings of each argument, refer to lowering methods on [`FreshType2`], [`UnfusedType2`], etc.
    pub fn to_artifect(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        constant_pool: &mut ConstantPool,
        versions_cpu_memory_divisions: impl Iterator<Item = u32>,
        ctx: &PanicJoinHandler,
        kernel_dir: PathBuf,
    ) -> Result<Artifect<Rt>, Error<'s, Rt>> {
        Ok(self
            .to_semi_artifect(
                options,
                hardware_info,
                constant_pool,
                versions_cpu_memory_divisions,
                ctx,
                kernel_dir,
            )?
            .finish(constant_pool))
    }

    /// Apply all passes to obtain [`SemiArtifect`].
    ///
    /// For meanings of each argument, refer to lowering methods on [`FreshType2`], [`UnfusedType2`], etc.
    pub fn to_semi_artifect(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        constant_pool: &mut ConstantPool,
        versions_cpu_memory_divisions: impl Iterator<Item = u32>,
        ctx: &PanicJoinHandler,
        kernel_dir: PathBuf,
    ) -> Result<SemiArtifect<Rt>, Error<'s, Rt>> {
        self.apply_passes(options, hardware_info, constant_pool, ctx)?
            .fuse(options, hardware_info, versions_cpu_memory_divisions, ctx)?
            .to_type3(options, hardware_info, constant_pool, ctx)?
            .apply_passes(options)?
            .to_artifect(options, hardware_info, kernel_dir)
    }
}
