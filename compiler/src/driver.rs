use std::{
    collections::BTreeMap,
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
    debug_user_function_table: bool,
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
    debug_extend_rewriting: bool,
    debug_track_splitting: bool,
    debug_multithread_instructions: bool,
    debug_instructions: bool,
    type2_visualizer: Type2DebugVisualizer,
    log: bool,
}

impl DebugOptions {
    pub fn all(debug_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&debug_dir).unwrap();
        Self {
            debug_dir,
            debug_user_function_table: true,
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
            debug_extend_rewriting: true,
            debug_track_splitting: true,
            debug_multithread_instructions: true,
            debug_instructions: true,
            type2_visualizer: Type2DebugVisualizer::Graphviz,
            log: false,
        }
    }

    pub fn none(debug_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&debug_dir).unwrap();
        Self {
            debug_dir,
            debug_user_function_table: true,
            debug_fresh_type2: false,
            debug_intt_mending: false,
            debug_type_inference: false,
            debug_precompute: false,
            debug_manage_invers: false,
            debug_kernel_fusion: false,
            debug_graph_scheduling: false,
            debug_obj_def_use: false,
            debug_obj_liveness: false,
            debug_obj_gpu_next_use: false,
            debug_fresh_type3: false,
            debug_extend_rewriting: false,
            debug_track_splitting: false,
            debug_multithread_instructions: false,
            debug_instructions: false,
            type2_visualizer: Type2DebugVisualizer::Graphviz,
            log: false,
        }
    }

    pub fn with_type3(mut self, switch: bool) -> Self {
        self.debug_fresh_type3 = switch;
        self.debug_extend_rewriting = switch;
        self
    }

    pub fn with_inst(mut self, switch: bool) -> Self {
        self.debug_multithread_instructions = switch;
        self.debug_instructions = switch;
        self
    }

    pub fn with_type2_visualizer(mut self, type2_visualizer: Type2DebugVisualizer) -> Self {
        self.type2_visualizer = type2_visualizer;
        self
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type2DebugVisualizer {
    Cytoscape,
    Graphviz,
}

fn debug_type2<'s, Ty: std::fmt::Debug>(
    fpath: PathBuf,
    g: &Digraph<type2::VertexId, type2::partial_typed::Vertex<'s, Ty>>,
    output_vid: type2::VertexId,
    visualizer: Type2DebugVisualizer,
) -> Option<JoinHandle<()>> {
    match visualizer {
        Type2DebugVisualizer::Cytoscape => {
            let fpath = fpath.with_extension("html");
            let mut f = std::fs::File::create(&fpath).unwrap();
            type2::visualize::write_graph(g, output_vid, &mut f).unwrap();
            None
        }
        Type2DebugVisualizer::Graphviz => {
            let mut f = std::fs::File::create(&fpath).unwrap();
            type2::pretty_print::write_graph(g, output_vid, &mut f).unwrap();
            drop(f);

            Some(compile_dot(fpath))
        }
    }
}

fn debug_type2_def_use<'s, Rt: std::fmt::Debug + RuntimeType>(
    fpath: PathBuf,
    g: &Digraph<type2::VertexId, type2::Vertex<'s, Rt>>,
    devices: &BTreeMap<type2::VertexId, type2::Device>,
    seq: &[type2::VertexId],
    obj_def: &type2::object_analysis::ObjectsDef,
    vertex_inputs: &type2::object_analysis::ObjectUse,
    visualizer: Type2DebugVisualizer,
) -> Option<JoinHandle<()>> {
    let fpath = match visualizer {
        Type2DebugVisualizer::Cytoscape => fpath.with_extension("html"),
        Type2DebugVisualizer::Graphviz => fpath,
    };
    let mut f = std::fs::File::create(&fpath).unwrap();

    let vf: Box<dyn Fn(_, _, _, _, _, _, _, _) -> _> = match visualizer {
        Type2DebugVisualizer::Cytoscape => {
            Box::new(type2::visualize::write_graph_with_optional_seq)
        }
        Type2DebugVisualizer::Graphviz => {
            Box::new(type2::pretty_print::write_graph_with_optional_seq)
        }
    };

    vf(
        g,
        &mut f,
        seq.last().copied(),
        seq.iter().copied(),
        true,
        |vid, _| match devices[&vid] {
            type2::Device::Cpu => Some("#FFFFFF"),
            type2::Device::Gpu => Some("#A5D6A7"),
            type2::Device::PreferGpu => {
                panic!("PreferGpu should has been resolved during deciding device")
            }
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

    if let Type2DebugVisualizer::Graphviz = visualizer {
        Some(compile_dot(fpath))
    } else {
        None
    }
}

fn debug_type2_with_seq<'s, Rt: std::fmt::Debug + RuntimeType>(
    fpath: PathBuf,
    g: &Digraph<type2::VertexId, type2::Vertex<'s, Rt>>,
    seq: &[type2::VertexId],
    visualizer: Type2DebugVisualizer,
) -> Option<JoinHandle<()>> {
    match visualizer {
        Type2DebugVisualizer::Cytoscape => {
            let fpath = fpath.with_extension("html");
            let mut f = std::fs::File::create(&fpath).unwrap();
            type2::visualize::write_graph_with_seq(g, &mut f, seq.iter().cloned()).unwrap();
            None
        }
        Type2DebugVisualizer::Graphviz => {
            let mut f = std::fs::File::create(&fpath).unwrap();
            type2::pretty_print::write_graph_with_seq(g, &mut f, seq.iter().cloned()).unwrap();
            Some(compile_dot(fpath))
        }
    }
}

fn debug_partial_typed_type2<'s, Ty: std::fmt::Debug>(
    fpath: PathBuf,
    g: &Digraph<type2::VertexId, type2::partial_typed::Vertex<'s, Option<Ty>>>,
    error_vid: type2::VertexId,
    output_vid: type2::VertexId,
    visualizer: Type2DebugVisualizer,
) -> Option<JoinHandle<()>> {
    match visualizer {
        Type2DebugVisualizer::Cytoscape => {
            let fpath = fpath.with_extension("html");
            let mut f = std::fs::File::create(&fpath).unwrap();
            type2::visualize::write_optinally_typed_graph(g, error_vid, output_vid, &mut f)
                .unwrap();
            None
        }
        Type2DebugVisualizer::Graphviz => {
            let mut f = std::fs::File::create(&fpath).unwrap();
            type2::pretty_print::write_optinally_typed_graph(g, error_vid, output_vid, &mut f)
                .unwrap();
            Some(compile_dot(fpath))
        }
    }
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
    visualizer: Type2DebugVisualizer,
) -> bool {
    if let Some(cycle) = g.try_find_cycle() {
        let mut f = std::fs::File::create(&fpath).unwrap();

        match visualizer {
            Type2DebugVisualizer::Cytoscape => {
                type2::visualize::write_graph_with_vertices_colored(
                    g,
                    output_vid,
                    &mut f,
                    |vid, _| {
                        if cycle[vid] {
                            Some("#e74c3c")
                        } else {
                            None
                        }
                    },
                )
                .unwrap();
            }
            Type2DebugVisualizer::Graphviz => {
                type2::pretty_print::write_graph_with_vertices_colored(
                    g,
                    output_vid,
                    &mut f,
                    |vid, _| {
                        if cycle[vid] {
                            Some("#e74c3c")
                        } else {
                            None
                        }
                    },
                )
                .unwrap();
                drop(f);

                compile_dot(fpath).join().unwrap()
            }
        }
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
pub struct PanicJoinHandler(
    Arc<
        Mutex<(
            Vec<JoinHandle<()>>,
            Box<dyn FnOnce(&PanicHookInfo<'_>) + Send + Sync>,
        )>,
    >,
);

impl PanicJoinHandler {
    pub fn new() -> Self {
        let hook = std::panic::take_hook();
        let hook: Box<dyn FnOnce(&PanicHookInfo) + Send + Sync> = Box::new(move |info| hook(info));
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

    fn add(&self, j: Option<JoinHandle<()>>) {
        if let Some(j) = j {
            self.0.lock().unwrap().0.push(j);
        }
    }

    fn abort(&self, pi: &std::panic::PanicHookInfo<'_>) {
        self.join();
        let mut hook: Box<dyn FnOnce(&PanicHookInfo) + Send + Sync> = Box::new(|_| ());
        std::mem::swap(&mut self.0.lock().unwrap().1, &mut hook);
        hook(pi);
    }
}

pub fn ast2type2<Rt: RuntimeType>(
    ast: impl ast::TypeEraseable<Rt>,
    options: &DebugOptions,
    allocator: PinnedMemoryPool,
    ctx: &PanicJoinHandler,
) -> Result<type2::Program<'static, Rt>, Error<'static, Rt>> {
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
            options.type2_visualizer,
        ));
    }

    if options.debug_user_function_table {
        let mut f = std::fs::File::create(&options.debug_dir.join("user_functions.txt")).unwrap();
        write!(f, "{:#?}", ast_cg.user_function_table).unwrap();
    }

    // Type inference
    let t2prog =
        match options.log_suround("Inferring Type...", || ast_cg.lower(output_vid), "Done.") {
            Ok(t2prog) => Ok(t2prog),
            Err((e, g)) => {
                let fpath = options.debug_dir.join("type2_type_inference.dot");
                debug_partial_typed_type2(fpath, &g, e.vid, output_vid, options.type2_visualizer)
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
    Ok(t2prog)
}

pub fn ast2inst<Rt: RuntimeType>(
    ast: impl ast::TypeEraseable<Rt>,
    allocator: PinnedMemoryPool,
    options: &DebugOptions,
    hardware_info: &HardwareInfo,
    ctx: &PanicJoinHandler,
) -> Result<
    (
        type3::lowering::Chunk<Rt>,
        args::ConstantTable<Rt>,
        PinnedMemoryPool,
    ),
    Error<'static, Rt>,
> {
    let t2prog = ast2type2(ast, options, allocator, &ctx)?;
    type2_to_inst(t2prog, options, hardware_info, &ctx)
}

pub fn type2_to_inst<Rt: RuntimeType>(
    t2prog: type2::Program<'static, Rt>,
    options: &DebugOptions,
    hardware_info: &HardwareInfo,
    ctx: &PanicJoinHandler,
) -> Result<
    (
        type3::lowering::Chunk<Rt>,
        args::ConstantTable<Rt>,
        PinnedMemoryPool,
    ),
    Error<'static, Rt>,
> {
    let type2::Program {
        cg: t2cg,
        consant_table: mut t2const_tab,
        user_function_table: t2uf_tab,
        memory_pool: mut allocator,
    } = t2prog;

    let output_vid = t2cg.output;

    let mut libs = Libs::new();

    if options.debug_type_inference {
        ctx.add(debug_type2(
            options.debug_dir.join("type2_type_inference.dot"),
            &t2cg.g,
            output_vid,
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
        output_vid,
        options.type2_visualizer,
    ) {
        panic!("graph is not a DAG after INTT Mending");
    }

    if options.debug_intt_mending {
        ctx.add(debug_type2(
            options.debug_dir.join("type2_intt_mending.dot"),
            &t2cg.g,
            output_vid,
            options.type2_visualizer,
        ));
    }

    // - Precompute NTT and MSM constants
    let t2cg = options.log_suround(
        "Precomputing constants for NTT and MSM",
        || {
            Ok(type2::precompute::precompute(
                t2cg,
                hardware_info.gpu_memory_limit as usize,
                &mut libs,
                &mut allocator,
                &mut t2const_tab,
            ))
        },
        "Done.",
    )?;

    if !check_type2_dag(
        options.debug_dir.join("type2_precompute.dot"),
        &t2cg.g,
        output_vid,
        options.type2_visualizer,
    ) {
        panic!("graph is not a DAG after Precomputing");
    }

    if options.debug_precompute {
        ctx.add(debug_type2(
            options.debug_dir.join("type2_precompute.dot"),
            &t2cg.g,
            output_vid,
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
        output_vid,
        options.type2_visualizer,
    ) {
        panic!("graph is not a DAG after Managing Inversions");
    }

    if options.debug_manage_invers {
        ctx.add(debug_type2(
            options.debug_dir.join("type2_manage_invers.dot"),
            &t2cg.g,
            output_vid,
            options.type2_visualizer,
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
        options.type2_visualizer,
    ) {
        panic!("graph is not a DAG after Arithmetic Kernel Fusion");
    }

    if options.debug_kernel_fusion {
        ctx.add(debug_type2(
            options.debug_dir.join("type2_kernel_fusion.dot"),
            &t2cg.g,
            output_vid,
            options.type2_visualizer,
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
        ctx.add(debug_type2_with_seq(
            path,
            &t2cg.g,
            &seq,
            options.type2_visualizer,
        ));
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
        ctx.add(debug_type2_def_use(
            fpath,
            &t2cg.g,
            &devices,
            &seq,
            &obj_def,
            &vertex_inputs,
            options.type2_visualizer,
        ));
    }

    let obj_def = obj_def;
    let (obj_dies_after, obj_dies_after_reversed) = options.log_suround(
        "Analyzing object lifetimes",
        || {
            let d = type2::object_analysis::analyze_die_after(
                &seq,
                &obj_def,
                &vertex_inputs,
                &vertex_inputs,
            );
            let r = d.reversed();
            Ok((d, r))
        },
        "Done",
    )?;

    if options.debug_obj_liveness {
        let fpath = options.debug_dir.join("type2_object_liveness.txt");
        let mut f = std::fs::File::create(&fpath).unwrap();
        obj_dies_after.iter().for_each(|(d, m)| {
            write!(f, "{:?}\n{:?}\n", d, m).unwrap();
        });
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
            .expect("memory plan failed"))
        },
        "Done.",
    )?;

    if options.debug_fresh_type3 {
        let mut f = std::fs::File::create(options.debug_dir.join("type3_fresh.html")).unwrap();
        type3::pretty_print::prettify(&t3chunk, &mut f).unwrap();
    }

    // - Extend Rewritting
    let t3chunk = options.log_suround(
        "Rewritting Extend and NewPoly",
        || Ok(type3::rewrite_extend::rewrite(t3chunk)),
        "Done.",
    )?;

    if options.debug_extend_rewriting {
        let mut f =
            std::fs::File::create(options.debug_dir.join("type3_extend_rewriting.html")).unwrap();
        type3::pretty_print::prettify(&t3chunk, &mut f).unwrap();
    }

    // - Track Splitting
    let track_tasks = options.log_suround(
        "Splitting tracks",
        || Ok(type3::track_splitting::split(&t3chunk)),
        "Done.",
    )?;

    if options.debug_track_splitting {
        let mut f =
            std::fs::File::create(options.debug_dir.join("type3_track_splitting.txt")).unwrap();
        write!(f, "{:?}", &track_tasks).unwrap();
    }

    // To Runtime Instructions

    // - Emitting Multithread Chunk
    let (mt_chunk, f_table, event_table, stream2variable_id, variable_id_allocator, libs) = options
        .log_suround(
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
    let rt_const_tab = type3::lowering::lower_constants(t2const_tab);

    if options.debug_instructions {
        let mut f = std::fs::File::create(options.debug_dir.join("instructions.txt")).unwrap();
        zkpoly_runtime::instructions::print_instructions(&rt_chunk.instructions, &mut f).unwrap();
    }

    Ok((rt_chunk, rt_const_tab, allocator))
}

pub fn dump_artifect<Rt: RuntimeType>(
    rt_chunk: &type3::lowering::Chunk<Rt>,
    rt_const_tab: &args::ConstantTable<Rt>,
    dir: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir.as_ref())?;

    let mut chunk_f = std::fs::File::create(dir.as_ref().join("chunk.json"))?;
    serde_json::to_writer_pretty(&mut chunk_f, &rt_chunk)?;

    let mut ct_header_f = std::fs::File::create(dir.as_ref().join("constants-manifest.json"))?;
    let ct_header = args::serialization::Header::build(rt_const_tab);
    serde_json::to_writer_pretty(&mut ct_header_f, &ct_header)?;

    let mut ct_f = std::fs::File::create(dir.as_ref().join("constants.bin"))?;
    ct_header.dump_entries_data(rt_const_tab, &mut ct_f)?;

    Ok(())
}

pub fn load_artifect<Rt: RuntimeType>(
    t2prog: type2::Program<'static, Rt>,
    dir: impl AsRef<std::path::Path>,
) -> std::io::Result<(
    type3::lowering::Chunk<Rt>,
    args::ConstantTable<Rt>,
    PinnedMemoryPool,
)> {
    let uf_table = t2prog.user_function_table;
    let ct_table = t2prog.consant_table;
    let mut allocator = t2prog.memory_pool;
    let (rt_chunk, rt_const_tab) = load_artifect_(uf_table, ct_table, &mut allocator, dir)?;
    Ok((rt_chunk, rt_const_tab, allocator))
}

fn load_artifect_<Rt: RuntimeType>(
    uf_table: type2::user_function::Table<Rt>,
    ct_table: ast::lowering::ConstantTable<Rt>,
    allocator: &mut PinnedMemoryPool,
    dir: impl AsRef<std::path::Path>,
) -> std::io::Result<(type3::lowering::Chunk<Rt>, args::ConstantTable<Rt>)> {
    let mut chunk_f = std::fs::File::open(dir.as_ref().join("chunk.json"))?;
    let rt_chunk_deserializer: type3::lowering::serialization::ChunkDeserializer =
        serde_json::from_reader(&mut chunk_f)?;
    let rt_chunk = rt_chunk_deserializer.deserialize_into_chunk(uf_table);

    let mut ct_header_f = std::fs::File::open(dir.as_ref().join("constants-manifest.json"))?;
    let ct_header: args::serialization::Header = serde_json::from_reader(&mut ct_header_f)?;

    let mut rt_const_tab = type3::lowering::lower_constants(ct_table);

    let mut ct_f = std::fs::File::open(dir.as_ref().join("constants.bin"))?;
    ct_header.load_constant_table(&mut rt_const_tab, &mut ct_f, allocator)?;

    Ok((rt_chunk, rt_const_tab))
}

pub fn prepare_vm<Rt: RuntimeType>(
    rt_chunk: type3::lowering::Chunk<Rt>,
    rt_const_tab: args::ConstantTable<Rt>,
    mem_allocator: PinnedMemoryPool,
    inputs: args::EntryTable<Rt>,
    pool: zkpoly_runtime::runtime::ThreadPool,
    gpu_allocator: Vec<zkpoly_cuda_api::mem::CudaAllocator>,
    rng: zkpoly_runtime::async_rng::AsyncRng,
) -> zkpoly_runtime::runtime::Runtime<Rt> {
    zkpoly_runtime::runtime::Runtime::new(
        rt_chunk.instructions,
        rt_chunk.n_variables,
        rt_const_tab,
        inputs,
        rt_chunk.f_table,
        pool,
        rt_chunk.event_table,
        rt_chunk.n_threads,
        mem_allocator,
        gpu_allocator,
        rng,
        rt_chunk.libs,
    )
}
