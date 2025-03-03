use std::{fmt::Write, path::PathBuf};

use zkpoly_common::load_dynamic::Libs;
use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_runtime::args::{self, RuntimeType};

use crate::{
    ast,
    transit::{type2, type3},
};

#[derive(Debug, Clone)]
pub struct Options {
    debug_dir: PathBuf,
    debug_fresh_type2: bool,
    debug_intt_mending: bool,
    debug_precompute: bool,
    debug_manage_invers: bool,
    debug_kernel_fusion: bool,
    debug_graph_scheduling: bool,
    debug_fresh_type3: bool,
    debug_instructions: bool,
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    gpu_memory_limit: u64,
}

fn debug_type2<'s, Rt: RuntimeType>(f: PathBuf, cg: &type2::Cg<'s, Rt>) {
    let mut f = std::fs::File::create(f).unwrap();
    type2::pretty_print::write_graph(cg, &mut f).unwrap();
}

pub fn ast2inst<Rt: RuntimeType>(
    ast: impl ast::TypeEraseable<Rt>,
    allocator: PinnedMemoryPool,
    options: &Options,
    hardware_info: &HardwareInfo,
) -> Result<(type3::lowering::Chunk<Rt>, args::ConstantTable<Rt>), ast::lowering::Error<'static, Rt>>
{
    // First from AST to Type2
    let (ast_cg, ast_output_vid) = ast::lowering::Cg::new(ast, allocator);
    let t2prog = ast_cg.lower(ast_output_vid)?;

    let type2::Program {
        cg: t2cg,
        consant_table: mut t2const_tab,
        user_function_table: t2uf_tab,
        memory_pool: mut allocator,
    } = t2prog;
    let mut libs = Libs::new();

    if options.debug_fresh_type2 {
        debug_type2(options.debug_dir.join("type2_fresh.dot"), &t2cg);
    }

    // Apply Type2 passes
    // - INTT Mending: Append division by n to end of each INTT
    let t2cg = type2::intt_mending::mend(t2cg, &mut t2const_tab);

    if options.debug_intt_mending {
        debug_type2(options.debug_dir.join("type2_intt_mending.dot"), &t2cg);
    }

    // - Precompute NTT and MSM constants
    let t2cg = type2::precompute::precompute(
        t2cg,
        hardware_info.gpu_memory_limit as usize,
        &mut libs,
        &mut allocator,
        &mut t2const_tab,
    );

    if options.debug_precompute {
        debug_type2(options.debug_dir.join("type2_precompute.dot"), &t2cg);
    }

    // - Manage Inversions: Rewrite inversions of scalars and polynomials to dedicated operators
    let t2cg = type2::manage_inverse::manage_inverse(t2cg);

    if options.debug_manage_invers {
        debug_type2(options.debug_dir.join("type2_manage_invers.dot"), &t2cg);
    }

    // - Arithmetic Kernel Fusion
    let t2cg = type2::kernel_fusion::fuse_arith(t2cg);

    if options.debug_kernel_fusion {
        debug_type2(options.debug_dir.join("type2_kernel_fusion.dot"), &t2cg);
    }

    // - Graph Scheduling
    let (seq, _) = type2::graph_scheduling::schedule(&t2cg);

    if options.debug_graph_scheduling {
        let mut f =
            std::fs::File::create(options.debug_dir.join("type2_graph_scheduled.dot")).unwrap();
        type2::pretty_print::write_graph_with_seq(&t2cg, &mut f, seq.iter().cloned()).unwrap();
    }

    // To Type3 through Memory Planning
    let t3chunk =
        type2::memory_planning::plan(hardware_info.gpu_memory_limit, &t2cg, &seq, &t2uf_tab)
            .expect("The computation graph is using too much smithereen space");

    if options.debug_fresh_type3 {
        let mut f = std::fs::File::create(options.debug_dir.join("type3_fresh.dot")).unwrap();
        type3::pretty_print::write_graph(&t3chunk, &mut f).unwrap();
    }

    // To Runtime Instructions
    let rt_chunk = type3::lowering::lower(t3chunk, t2uf_tab);
    let rt_const_tab = type3::lowering::lower_constants(t2const_tab);

    if options.debug_instructions {
        let mut f = std::fs::File::create(options.debug_dir.join("instructions.txt")).unwrap();
        zkpoly_runtime::instructions::print_instructions(&rt_chunk.instructions, &mut f).unwrap();
    }

    Ok((rt_chunk, rt_const_tab))
}
