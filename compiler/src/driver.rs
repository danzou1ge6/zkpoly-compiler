use std::{
    io::Write,
    panic::PanicHookInfo,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use zkpoly_common::{
    digraph::internal::{Digraph, SubDigraph},
    load_dynamic::Libs,
};
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
    debug_obj_def_use: bool,
    debug_obj_liveness: bool,
    debug_obj_gpu_next_use: bool,
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
            debug_obj_def_use: true,
            debug_obj_liveness: true,
            debug_obj_gpu_next_use: true,
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
            println!("[Compiler] {}", prologue);
        }
        let r = f()?;
        if self.log {
            println!("[Compiler] {}", epilogue);
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

#[derive(Clone)]
struct Ctx(
    Arc<
        Mutex<(
            Vec<JoinHandle<()>>,
            Box<dyn FnOnce(&PanicHookInfo<'_>) + Send + Sync>,
        )>,
    >,
);

impl Ctx {
    fn new() -> Self {
        let hook = std::panic::take_hook();
        let r = Self(Arc::new(Mutex::new((Vec::new(), hook))));
        let r1 = r.clone();

        std::panic::set_hook(Box::new(move |pi| {
            r.abort(pi);
        }));

        r1
    }

    fn join(&self) {
        std::mem::take::<Vec<_>>(self.0.lock().unwrap().0.as_mut())
            .into_iter()
            .for_each(|j| j.join().unwrap());
    }

    fn add(&self, j: JoinHandle<()>) {
        self.0.lock().unwrap().0.push(j);
    }

    fn abort(&self, pi: &std::panic::PanicHookInfo<'_>) {
        self.join();
        let mut hook: Box<dyn FnOnce(&PanicHookInfo) + Send + Sync> = Box::new(|_| ());
        std::mem::swap(&mut self.0.lock().unwrap().1, &mut hook);
        hook(pi);
    }
}

pub fn ast2inst<Rt: RuntimeType>(
    ast: impl ast::TypeEraseable<Rt>,
    allocator: PinnedMemoryPool,
    options: &DebugOptions,
    hardware_info: &HardwareInfo,
) -> Result<(type3::lowering::Chunk<Rt>, args::ConstantTable<Rt>), Error<'static, Rt>> {
    let ctx = Ctx::new();

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
    let libs = Libs::new();

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
        panic!("graph is not a DAG after INTT Mending");
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
    //     panic!("graph is not a DAG after Precomputing");
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
        panic!("graph is not a DAG after Managing Inversions");
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
        panic!("graph is not a DAG after Arithmetic Kernel Fusion");
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
        let mut f = std::fs::File::create(&path).unwrap();
        type2::pretty_print::write_graph_with_seq(&t2cg.g, &mut f, seq.iter().cloned()).unwrap();
        ctx.add(compile_dot(path));
    }

    let g = SubDigraph::new(&t2cg.g, t2cg.g.connected_component(output_vid));

    // - Decide Device
    let devices = options.log_suround(
        "Deciding device",
        || Ok(type2::decide_device::decide(&g)),
        "Done.",
    )?;

    // - Object Analysis
    let (mut obj_def, mut obj_id_allocator) = options.log_suround(
        "Analyzing object definitions",
        || {
            Ok(type2::object_analysis::analyze_def(&g, &seq, |vid| {
                devices[&vid]
            }))
        },
        "Done.",
    )?;

    let vertex_inputs = options.log_suround(
        "Planning vertex inputs",
        || {
            Ok(type2::object_analysis::plan_vertex_inputs(
                &g,
                &mut obj_def,
                |vid| devices[&vid],
                &mut obj_id_allocator,
            ))
        },
        "Done",
    )?;

    if options.debug_obj_def_use {
        let fpath = options.debug_dir.join("type2_object_def_use.dot");
        let mut f = std::fs::File::create(&fpath).unwrap();
        type2::pretty_print::write_graph_with_optional_seq(
            &t2cg.g,
            &mut f,
            seq.iter().copied(),
            true,
            |vid, _| match devices[&vid] {
                type3::Device::Cpu => Some("#FFFFFF"),
                type3::Device::Gpu => Some("#A5D6A7"),
                type3::Device::Stack => Some("#90CAF9"),
            },
            |vid, _| Some(format!("{:?}", &obj_def.values[&vid])),
            |vid, v| {
                Some(
                    v.uses()
                        .zip(vertex_inputs.input_of(vid).map(|vv| format!("{:?}", vv)))
                        .collect(),
                )
            },
        )
        .unwrap();
        ctx.add(compile_dot(fpath));
    }

    let obj_def = obj_def;
    let (obj_dies_after, obj_dies_after_reversed) = options.log_suround(
        "Analyzing object lifetimes",
        || {
            let d =
                type2::object_analysis::analyze_die_after(&seq, &devices, &obj_def, &vertex_inputs);
            let r = d.reversed();
            Ok((d, r))
        },
        "Done",
    )?;

    if options.debug_obj_liveness {
        let fpath = options.debug_dir.join("type2_object_liveness.txt");
        let mut f = std::fs::File::create(&fpath).unwrap();
        write!(f, "{:?}", &obj_dies_after).unwrap();
    }

    let (obj_used_by, obj_gpu_next_use) = options.log_suround(
        "Analyzing next uses of objects on GPU",
        || {
            let obj_used_by = type2::object_analysis::analyze_used_by(&seq, &vertex_inputs);
            let obj_gpu_next_use =
                type2::object_analysis::analyze_gpu_next_use(&seq, &vertex_inputs, &obj_used_by);

            Ok((obj_used_by, obj_gpu_next_use))
        },
        "Done",
    )?;

    if options.debug_obj_gpu_next_use {
        let fpath = options.debug_dir.join("type2_object_gpu_next_use.txt");
        let mut f = std::fs::File::create(&fpath).unwrap();
        write!(f, "{:?}\n", &obj_used_by).unwrap();
        write!(f, "{:?}\n", &obj_gpu_next_use).unwrap();
    }

    // To Type3 through Memory Planning
    let t3chunk = options.log_suround(
        "Planning memory",
        || {
            Ok(type2::memory_planning::plan(
                hardware_info.gpu_memory_limit,
                &t2cg,
                &g,
                &seq,
                &obj_dies_after,
                &obj_dies_after_reversed,
                &obj_def,
                &obj_gpu_next_use,
                &vertex_inputs,
                &devices,
                &t2uf_tab,
                &mut obj_id_allocator,
                libs,
            )
            .expect("The computation graph is using too much smithereen space"))
        },
        "Done.",
    )?;

    if options.debug_fresh_type3 {
        let mut f = std::fs::File::create(options.debug_dir.join("type3_fresh.html")).unwrap();
        type3::pretty_print::prettify(&t3chunk, &mut f).unwrap();
    }

    panic!("Let's leave further passes for tomorrow");

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
