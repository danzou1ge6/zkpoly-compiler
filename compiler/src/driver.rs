use std::{path::PathBuf, process::Command};

use zkpoly_common::{digraph::internal::Digraph, load_dynamic::Libs};
use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_runtime::args::{self, RuntimeType};

use crate::{
    ast,
    transit::{type2, type3},
};

#[derive(Debug, Clone)]
pub struct DebugOptions {
    debug_dir: PathBuf,
    debug_fresh_type2: bool,
    debug_type_inference: bool,
    debug_intt_mending: bool,
    debug_precompute: bool,
    debug_manage_invers: bool,
    debug_kernel_fusion: bool,
    debug_graph_scheduling: bool,
    debug_fresh_type3: bool,
    debug_instructions: bool,
    log: bool,
}

impl DebugOptions {
    pub fn all(debug_dir: PathBuf) -> Self {
        Self {
            debug_dir,
            debug_fresh_type2: true,
            debug_intt_mending: true,
            debug_type_inference: true,
            debug_precompute: true,
            debug_manage_invers: true,
            debug_kernel_fusion: true,
            debug_graph_scheduling: true,
            debug_fresh_type3: true,
            debug_instructions: true,
            log: false,
        }
    }

    pub fn with_log(mut self, log: bool) -> Self {
        self.log = log;
        self
    }

    fn log_suround<R, E>(
        &self,
        prologue: &str,
        f: impl FnOnce() -> Result<R, E>,
        epilogue: &str,
    ) -> Result<R, E> {
        if self.log {
            println!("[Compiler]{}", prologue);
        }
        let r = f()?;
        if self.log {
            println!("[Compiler]{}", epilogue);
        }
        Ok(r)
    }
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub gpu_memory_limit: u64,
}

fn debug_type2<'s, Ty: std::fmt::Debug>(
    fpath: PathBuf,
    g: &Digraph<type2::VertexId, type2::partial_typed::Vertex<'s, Ty>>,
    output_vid: type2::VertexId,
) {
    let mut f = std::fs::File::create(&fpath).unwrap();
    type2::pretty_print::write_graph(g, output_vid, &mut f).unwrap();
    drop(f);

    compile_dot(&fpath);
}

fn compile_dot(fpath: &PathBuf) {
    let _ = Command::new("dot")
        .arg("-Tsvg")
        .arg(fpath)
        .arg("-o")
        .arg(fpath.with_extension("svg"))
        .spawn();
}

fn check_type2_dag<'s, Rt: RuntimeType>(
    fpath: PathBuf,
    g: &Digraph<type2::VertexId, type2::Vertex<'s, Rt>>,
    output_vid: type2::VertexId,
) -> bool {
    if let Some(cycle) = g.try_find_cycle() {
        let mut f = std::fs::File::create(&fpath).unwrap();
        type2::pretty_print::write_graph_with_vertices_colored(g, output_vid, &mut f, |vid, _| {
            if cycle[vid] {
                Some("#e74c3c")
            } else {
                None
            }
        })
        .unwrap();
        drop(f);

        compile_dot(&fpath);
        false
    } else {
        true
    }
}

#[derive(Debug, Clone)]
pub enum Error<'s, Rt: RuntimeType> {
    Typ(ast::lowering::Error<'s, Rt>),
    NotDag,
}

pub fn ast2inst<Rt: RuntimeType>(
    ast: impl ast::TypeEraseable<Rt>,
    allocator: PinnedMemoryPool,
    options: &DebugOptions,
    hardware_info: &HardwareInfo,
) -> Result<(type3::lowering::Chunk<Rt>, args::ConstantTable<Rt>), Error<'static, Rt>> {
    // First from AST to Type2
    let (ast_cg, output_vid) = options.log_suround(
        "Lowering AST to Type2...",
        || Ok(ast::lowering::Cg::new(ast, allocator)),
        "Done.",
    )?;

    if options.debug_fresh_type2 {
        debug_type2(
            options.debug_dir.join("type2_fresh.dot"),
            &ast_cg.g,
            output_vid,
        );
    }

    // Type inference
    let t2prog =
        match options.log_suround("Inferring Type...", || ast_cg.lower(output_vid), "Done.") {
            Ok(t2prog) => Ok(t2prog),
            Err((e, g)) => {
                let fpath = options.debug_dir.join("type2_type_inference.dot");
                let mut f = std::fs::File::create(&fpath).unwrap();
                type2::pretty_print::write_optinally_typed_graph(&g, e.vid, output_vid, &mut f)
                    .unwrap();
                drop(f);

                compile_dot(&fpath);
                Err(Error::Typ(e))
            }
        }?;

    if !check_type2_dag(
        options.debug_dir.join("type2_type_inference.dot"),
        &t2prog.cg.g,
        output_vid,
    ) {
        return Err(Error::NotDag);
    }

    let type2::Program {
        cg: t2cg,
        consant_table: mut t2const_tab,
        user_function_table: t2uf_tab,
        memory_pool: mut allocator,
    } = t2prog;
    let mut libs = Libs::new();

    if options.debug_type_inference {
        debug_type2(
            options.debug_dir.join("type2_type_inference.dot"),
            &t2cg.g,
            output_vid,
        );
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
        output_vid,
    ) {
        panic!("graph is not a DAG after INTT Mending");
    }

    if options.debug_intt_mending {
        debug_type2(
            options.debug_dir.join("type2_intt_mending.dot"),
            &t2cg.g,
            output_vid,
        );
    }

    // - Precompute NTT and MSM constants
    // let t2cg = options.log_suround(
    //     "Precomputing constants for NTT and MSM",
    //     || {
    //         Ok(type2::precompute::precompute(
    //             t2cg,
    //             hardware_info.gpu_memory_limit as usize,
    //             &mut libs,
    //             &mut allocator,
    //             &mut t2const_tab,
    //         ))
    //     },
    //     "Done.",
    // )?;

    if !check_type2_dag(
        options.debug_dir.join("type2_precompute.dot"),
        &t2cg.g,
        output_vid,
    ) {
        panic!("graph is not a DAG after Precomputing");
    }

    if options.debug_precompute {
        debug_type2(
            options.debug_dir.join("type2_precompute.dot"),
            &t2cg.g,
            output_vid,
        );
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
        output_vid,
    ) {
        panic!("graph is not a DAG after Managing Inversions");
    }

    if options.debug_manage_invers {
        debug_type2(
            options.debug_dir.join("type2_manage_invers.dot"),
            &t2cg.g,
            output_vid,
        );
    }

    // - Arithmetic Kernel Fusion
    let t2cg = options.log_suround(
        "Fusing arithmetic kernels",
        || Ok(type2::kernel_fusion::fuse_arith(t2cg)),
        "Done.",
    )?;

    if !check_type2_dag(
        options.debug_dir.join("type2_kernel_fusion.dot"),
        &t2cg.g,
        output_vid,
    ) {
        panic!("graph is not a DAG after Arithmetic Kernel Fusion");
    }

    if options.debug_kernel_fusion {
        debug_type2(
            options.debug_dir.join("type2_kernel_fusion.dot"),
            &t2cg.g,
            output_vid,
        );
    }

    // - Graph Scheduling
    let (seq, _) = options.log_suround(
        "Scheduling graph",
        || Ok(type2::graph_scheduling::schedule(&t2cg)),
        "Done.",
    )?;

    if options.debug_graph_scheduling {
        let mut f =
            std::fs::File::create(options.debug_dir.join("type2_graph_scheduled.dot")).unwrap();
        type2::pretty_print::write_graph_with_seq(&t2cg.g, &mut f, seq.iter().cloned()).unwrap();
    }

    panic!("abort");

    // To Type3 through Memory Planning
    let t3chunk =
        options.log_suround(
            "Planning memory",
            || {
                Ok(type2::memory_planning::plan(
                    hardware_info.gpu_memory_limit,
                    &t2cg,
                    &seq,
                    &t2uf_tab,
                )
                .expect("The computation graph is using too much smithereen space"))
            },
            "Done.",
        )?;

    if options.debug_fresh_type3 {
        let mut f = std::fs::File::create(options.debug_dir.join("type3_fresh.dot")).unwrap();
        type3::pretty_print::write_graph(&t3chunk, &mut f).unwrap();
    }

    // To Runtime Instructions
    let rt_chunk = options.log_suround(
        "Lowering Type3 to Runtime Instructions",
        || Ok(type3::lowering::lower(t3chunk, t2uf_tab)),
        "Done.",
    )?;
    let rt_const_tab = type3::lowering::lower_constants(t2const_tab);

    if options.debug_instructions {
        let mut f = std::fs::File::create(options.debug_dir.join("instructions.txt")).unwrap();
        zkpoly_runtime::instructions::print_instructions(&rt_chunk.instructions, &mut f).unwrap();
    }

    Ok((rt_chunk, rt_const_tab))
}
