use std::{io::Write, path::PathBuf, process::Command, thread::JoinHandle};

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

static GRAPHVIZ_INTERACTIVE_HTML_1: &'static str = r#"
<!DOCTYPE html>
<html>
<head>
</head>
<body>
    <div id="graph-container">
"#;

static GRAPHVIZ_INTERACTIVE_HTML_2: &'static str = r#"
   </div>

    <script>
    function setEdgeColor(edge, color) {
      const elements = edge.querySelectorAll('path, polygon, text');
      elements.forEach(element => {
        if (element.tagName.toLowerCase() === 'path') {
          element.setAttribute('stroke', color);
        }
        if (element.tagName.toLowerCase() === 'polygon') {
          element.setAttribute('stroke', color);
          element.setAttribute('fill', color);
        }
        if (element.tagName.toLowerCase() === 'text') {
          element.setAttribute('fill', color);
        }
      });
    }
    document.querySelectorAll('.node').forEach(node => {
      let polygon =  node.querySelector('polygon')
      node.addEventListener('click', en => {
        document.querySelectorAll('.edge').forEach(edge => setEdgeColor(edge, '#0F0F0F0F'))
        document.querySelectorAll('.' + node.id + '-neighbour').forEach(edge => setEdgeColor(edge, 'black'))
      })
    })
    </script>
</body>
</html>
"#;

fn debug_type2<'s, Ty: std::fmt::Debug>(
    fpath: PathBuf,
    g: &Digraph<type2::VertexId, type2::partial_typed::Vertex<'s, Ty>>,
    output_vid: type2::VertexId,
) -> JoinHandle<()> {
    let mut f = std::fs::File::create(&fpath).unwrap();
    type2::pretty_print::write_graph(g, output_vid, &mut f).unwrap();
    drop(f);

    compile_dot(fpath)
}

fn compile_dot(fpath: PathBuf) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let output_file = fpath.with_extension("html");
        println!("[Visualizer] Compiling dot {}", fpath.to_string_lossy());

        match Command::new("dot").arg("-Tsvg").arg(&fpath).output() {
            Ok(out) => {
                let mut f = std::fs::File::create(&output_file).unwrap();
                f.write_all(GRAPHVIZ_INTERACTIVE_HTML_1.as_bytes()).unwrap();
                f.write_all(&out.stdout).unwrap();
                f.write_all(GRAPHVIZ_INTERACTIVE_HTML_2.as_bytes()).unwrap();
                println!(
                    "[Visualizer] Dot compiled {} successfully",
                    fpath.to_string_lossy()
                );
            }
            Err(e) => println!(
                "[Visualizer] Dot compiling {} exited with error: {:?}",
                fpath.to_string_lossy(),
                &e
            ),
        }
    })
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

        compile_dot(fpath).join().unwrap();
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

struct Ctx(Vec<JoinHandle<()>>);

impl Ctx {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn join(&mut self) {
        std::mem::take(&mut self.0).into_iter().for_each(|j| j.join().unwrap());
    }

    fn add(&mut self, j: JoinHandle<()>) {
        self.0.push(j);
    }

    fn abort(&mut self, msg: &str) {
        println!("[Compiler] Aborting: {}", msg);
        self.join();
        panic!("abort");
    }
}

pub fn ast2inst<Rt: RuntimeType>(
    ast: impl ast::TypeEraseable<Rt>,
    allocator: PinnedMemoryPool,
    options: &DebugOptions,
    hardware_info: &HardwareInfo,
) -> Result<(type3::lowering::Chunk<Rt>, args::ConstantTable<Rt>), Error<'static, Rt>> {
    let mut ctx = Ctx::new();

    // First from AST to Type2
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
        ));
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

                compile_dot(fpath);
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
        ctx.add(debug_type2(
            options.debug_dir.join("type2_type_inference.dot"),
            &t2cg.g,
            output_vid,
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
        output_vid,
    ) {
        ctx.abort("graph is not a DAG after INTT Mending");
    }

    if options.debug_intt_mending {
        ctx.add(debug_type2(
            options.debug_dir.join("type2_intt_mending.dot"),
            &t2cg.g,
            output_vid,
        ));
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

    // if !check_type2_dag(
    //     options.debug_dir.join("type2_precompute.dot"),
    //     &t2cg.g,
    //     output_vid,
    // ) {
    //     ctx.abort("graph is not a DAG after Precomputing");
    // }

    // if options.debug_precompute {
    //     ctx.add(debug_type2(
    //         options.debug_dir.join("type2_precompute.dot"),
    //         &t2cg.g,
    //         output_vid,
    //     ));
    // }

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
        ctx.abort("graph is not a DAG after Managing Inversions");
    }

    if options.debug_manage_invers {
        ctx.add(debug_type2(
            options.debug_dir.join("type2_manage_invers.dot"),
            &t2cg.g,
            output_vid,
        ));
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
        ctx.abort("graph is not a DAG after Arithmetic Kernel Fusion");
    }

    if options.debug_kernel_fusion {
        ctx.add(debug_type2(
            options.debug_dir.join("type2_kernel_fusion.dot"),
            &t2cg.g,
            output_vid,
        ));
    }

    // - Graph Scheduling
    let (seq, _) = options.log_suround(
        "Scheduling graph",
        || Ok(type2::graph_scheduling::schedule(&t2cg)),
        "Done.",
    )?;

    if options.debug_graph_scheduling {
        let path = options.debug_dir.join("type2_graph_scheduled.dot");
        let mut f =
            std::fs::File::create(&path).unwrap();
        type2::pretty_print::write_graph_with_seq(&t2cg.g, &mut f, seq.iter().cloned()).unwrap();
        ctx.add(compile_dot(path));
    }

    ctx.abort("Let's leave further passes for tomorrow");

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

    ctx.abort("Let's leave further passes for tomorrow");

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
