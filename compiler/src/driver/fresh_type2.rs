use std::io::{Read, Write};
use zkpoly_common::load_dynamic::Libs;
use zkpoly_runtime::args::{self, RuntimeType};

use super::{
    ast, check_type2_dag, processed_type2::ProcessedType2, type2, ConstantPool, DebugOptions,
    Error, HardwareInfo, PanicJoinHandler,
};

pub struct FreshType2<'s, Rt: RuntimeType> {
    prog: type2::Program<'s, Rt>,
}

impl<'s, Rt: RuntimeType> FreshType2<'s, Rt> {
    pub fn from_ast(
        ast: impl ast::TypeEraseable<Rt>,
        options: &DebugOptions,
        ctx: &PanicJoinHandler,
    ) -> Result<Self, Error<'s, Rt>> {
        let (ast_cg, output_vid) = options.log_suround(
            "Lowering AST to Type2...",
            || Ok(ast::lowering::Cg::new(ast)),
            "Done.",
        )?;

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
        constant_pool: &mut ConstantPool,
        ctx: &PanicJoinHandler,
    ) -> Result<ProcessedType2<'s, Rt>, Error<'static, Rt>> {
        let type2::Program {
            cg: t2cg,
            consant_table: mut t2const_tab,
            user_function_table: t2uf_tab,
        } = self.prog;

        let mut libs = Libs::new();

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

        if !check_type2_dag(
            options.debug_dir.join("type2_precompute.dot"),
            &t2cg.g,
            t2cg.output,
            options.type2_visualizer,
        ) {
            panic!("graph is not a DAG after Precomputing");
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

        Ok(ProcessedType2 {
            cg: t2cg,
            constant_table: t2const_tab,
            uf_table: t2uf_tab,
            libs,
        })
    }
}
