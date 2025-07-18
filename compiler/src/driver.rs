use std::{
    collections::{BTreeMap, HashMap},
    io::{Read, Write},
    panic::PanicHookInfo,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use zkpoly_common::digraph::internal::{Digraph, SubDigraph};
use zkpoly_cuda_api::{bindings::cudaDeviceSynchronize, cuda_check, mem::CudaAllocator};
use zkpoly_memory_pool::{
    buddy_disk_pool::DiskMemoryPool, static_allocator::CpuStaticAllocator, BuddyDiskPool,
};
use zkpoly_runtime::args::RuntimeType;

use crate::{
    ast,
    transit::{type2, type3},
    utils::human_readable_size,
};

pub use ast::ConstantPool;

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
    debug_fresh_type3: bool,
    debug_memory_planning_statistics: bool,
    debug_extend_rewriting: bool,
    debug_track_splitting: bool,
    debug_multithread_instructions: bool,
    debug_instructions: bool,
    debug_cse: bool,
    debug_arith_decide_mutable: bool,
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
            debug_cse: true,
            debug_arith_decide_mutable: true,
            debug_fresh_type3: true,
            debug_memory_planning_statistics: true,
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
            debug_cse: false,
            debug_intt_mending: false,
            debug_type_inference: false,
            debug_precompute: false,
            debug_manage_invers: false,
            debug_kernel_fusion: false,
            debug_arith_decide_mutable: false,
            debug_graph_scheduling: false,
            debug_obj_def_use: false,
            debug_obj_liveness: false,
            debug_fresh_type3: false,
            debug_memory_planning_statistics: false,
            debug_extend_rewriting: false,
            debug_track_splitting: false,
            debug_multithread_instructions: false,
            debug_instructions: false,
            type2_visualizer: Type2DebugVisualizer::Graphviz,
            log: false,
        }
    }

    pub fn minimal(debug_dir: PathBuf) -> Self {
        let r = Self::none(debug_dir);
        Self {
            debug_user_function_table: true,
            debug_obj_def_use: true,
            debug_obj_liveness: true,
            debug_memory_planning_statistics: true,
            debug_extend_rewriting: true,
            debug_instructions: true,
            ..r
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

    fn log_suround<R, E, S1, S2>(
        &self,
        prologue: S1,
        f: impl FnOnce() -> Result<R, E>,
        epilogue: S2,
    ) -> Result<R, E>
    where
        S1: AsRef<str>,
        S2: AsRef<str>,
    {
        if self.log {
            println!("[Compiler] {}", prologue.as_ref());
        }
        let r = f()?;
        if self.log {
            println!("[Compiler] {}", epilogue.as_ref());
        }
        Ok(r)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct MemoryInfo {
    pub(crate) memory_limit: u64,
    pub(crate) smithereen_space: u64,
}

impl MemoryInfo {
    pub fn new(memory_limit: u64, smithereen_space: u64) -> Self {
        Self {
            memory_limit,
            smithereen_space,
        }
    }

    pub fn memory_limit(&self) -> u64 {
        self.memory_limit
    }

    pub fn smithereen_space(&self) -> u64 {
        self.smithereen_space
    }

    pub fn integral_space(&self) -> u64 {
        self.memory_limit - self.smithereen_space
    }

    pub fn page_number(&self, page_size: u64) -> u64 {
        self.integral_space() / page_size
    }
}

#[derive(Debug, Clone)]
pub struct DiskMemoryInfo {
    disk_path: Option<PathBuf>,
}

impl DiskMemoryInfo {
    pub fn new(disk_path: Option<PathBuf>) -> Self {
        Self { disk_path }
    }

    pub fn disk_path(&self) -> Option<&PathBuf> {
        self.disk_path.as_ref()
    }

    pub fn has_path(&self) -> bool {
        self.disk_path.is_none()
    }
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    gpus: Vec<MemoryInfo>,
    cpu: MemoryInfo,
    disk: Vec<DiskMemoryInfo>,
    page_size: Option<u64>,
}

impl HardwareInfo {
    pub fn n_gpus(&self) -> usize {
        self.gpus.len()
    }

    pub fn gpus(&self) -> impl Iterator<Item = &MemoryInfo> {
        self.gpus.iter()
    }

    pub fn disks(&self) -> impl Iterator<Item = &DiskMemoryInfo> {
        self.disk.iter()
    }

    pub fn cpu(&self) -> &MemoryInfo {
        &self.cpu
    }

    pub fn smallest_gpu_memory_integral_limit(&self) -> u64 {
        self.gpus()
            .map(|gpu| gpu.memory_limit - gpu.smithereen_space)
            .min()
            .expect("no GPU")
    }

    pub fn new(cpu: MemoryInfo) -> Self {
        Self {
            gpus: Vec::new(),
            cpu,
            disk: Vec::new(),
            page_size: None,
        }
    }

    pub fn with_page_size(self, page_size: u64) -> Self {
        Self {
            page_size: Some(page_size),
            ..self
        }
    }

    pub fn with_cpu(self, cpu: MemoryInfo) -> Self {
        HardwareInfo { cpu, ..self }
    }

    pub fn with_disk(mut self, info: DiskMemoryInfo) -> Self {
        self.disk.push(info);
        self
    }

    pub fn with_gpu(mut self, gpu: MemoryInfo) -> Self {
        self.gpus.push(gpu);
        self
    }

    pub fn page_size(&self) -> u64 {
        self.page_size.expect("page size not specified")
    }

    pub fn disks_available(&self) -> usize {
        self.disk.len()
    }

    pub fn cpu_allocator(&self, memory_check: bool) -> CpuStaticAllocator {
        CpuStaticAllocator::new(self.cpu().memory_limit() as usize, memory_check)
    }

    pub fn gpu_allocators(&self, memory_check: bool) -> HashMap<i32, CudaAllocator> {
        use zkpoly_cuda_api::mem;
        self.gpus()
            .enumerate()
            .map(|(id, gpu)| {
                (
                    id as i32,
                    CudaAllocator {
                        statik: mem::StaticAllocator::new(
                            0,
                            gpu.smithereen_space() as usize,
                            memory_check,
                        ),
                        page: mem::PageAllocator::new(
                            zkpoly_common::devices::DeviceType::GPU { device_id: 0 },
                            self.page_size() as usize,
                            gpu.page_number(self.page_size()) as usize,
                        ),
                    },
                )
            })
            .collect()
    }

    pub fn disk_allocator(&self, max_block: usize) -> DiskMemoryPool {
        self.disks()
            .map(|disk_info| {
                let disk_path = disk_info.disk_path().cloned();
                BuddyDiskPool::new(max_block, disk_path).expect("cannot create disk pool")
            })
            .collect()
    }

    pub fn smaller_cpus<'a>(&'a self) -> impl Iterator<Item = MemoryInfo> + 'a {
        (1..=3).map(|p| {
            MemoryInfo::new(
                self.cpu.memory_limit / 2u64.pow(p),
                self.cpu.smithereen_space,
            )
        })
    }
}

struct VersionsHelper<'a> {
    mem_info: &'a MemoryInfo,
}

impl<'a> VersionsHelper<'a> {
    fn human_readable_name(&self) -> String {
        human_readable_size(self.mem_info.memory_limit())
    }

    fn log_prologue<S>(&self, msg: S) -> String
    where
        S: AsRef<str>,
    {
        format!(
            "{} for {}",
            msg.as_ref(),
            human_readable_size(self.mem_info.memory_limit())
        )
    }

    fn debug_filename(&self, f: &'static str) -> String {
        let pb = PathBuf::from(f);
        let stem = pb.file_stem().unwrap().to_str().unwrap();
        let ext = pb.extension().unwrap().to_str().unwrap();
        pb.with_file_name(format!(
            "{}_{}.{}",
            stem,
            human_readable_size(self.mem_info.memory_limit()),
            ext
        ))
        .to_str()
        .unwrap()
        .to_string()
    }
}

#[derive(Clone)]
struct Versions<T>(Vec<(MemoryInfo, T)>);

impl<T> Versions<T> {
    pub fn build<'s, Rt: RuntimeType>(
        hardware_info: &HardwareInfo,
        mut f: impl for<'a> FnMut(&'a MemoryInfo, VersionsHelper<'a>) -> Result<T, Error<'s, Rt>>,
    ) -> Result<Self, Error<'s, Rt>> {
        Ok(Self(
            hardware_info
                .smaller_cpus()
                .map(|cpu| {
                    let t = f(&cpu, VersionsHelper { mem_info: &cpu })?;
                    Ok((cpu, t))
                })
                .collect::<Result<_, _>>()?,
        ))
    }

    pub fn map<'s, T2, Rt: RuntimeType>(
        self,
        mut f: impl for<'a> FnMut(&'a MemoryInfo, T, VersionsHelper<'a>) -> Result<T2, Error<'s, Rt>>,
    ) -> Result<Versions<T2>, Error<'s, Rt>> {
        Ok(Versions(
            self.0
                .into_iter()
                .map(|(cpu, t)| {
                    let t2 = f(&cpu, t, VersionsHelper { mem_info: &cpu })?;
                    Ok((cpu, t2))
                })
                .collect::<Result<_, _>>()?,
        ))
    }

    pub fn dump(
        &self,
        dir: impl AsRef<std::path::Path>,
        mut write: impl FnMut(std::fs::File, &MemoryInfo, &T) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        for (cpu, t) in self.0.iter() {
            let f = std::fs::File::create(dir.as_ref().join(format!(
                "version_{}.json",
                human_readable_size(cpu.memory_limit)
            )))?;

            write(f, cpu, t)?;
        }
        Ok(())
    }

    pub fn load(
        dir: impl AsRef<std::path::Path>,
        mut read: impl FnMut(std::fs::File) -> std::io::Result<(MemoryInfo, T)>,
    ) -> std::io::Result<Self> {
        let mut r = Vec::new();

        for entry in dir.as_ref().read_dir()? {
            let path = entry?.path();
            if path
                .file_stem()
                .is_some_and(|f| f.to_string_lossy().into_owned().starts_with("versions_"))
                && path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.to_string_lossy().into_owned() == "json")
            {
                let f = std::fs::File::open(path)?;

                let x = read(f)?;
                r.push(x);
            }
        }
        Ok(Self(r))
    }

    pub fn load_buffered<'b>(
        dir: impl AsRef<std::path::Path>,
        buffers: &'b mut Vec<String>,
        mut read: impl FnMut(&'b str) -> std::io::Result<(MemoryInfo, T)>,
    ) -> std::io::Result<Self>
    where
        T: 'b,
    {
        let mut r = Vec::new();

        for entry in dir.as_ref().read_dir()? {
            let path = entry?.path();
            if path
                .file_stem()
                .is_some_and(|f| f.to_string_lossy().into_owned().starts_with("versions_"))
                && path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.to_string_lossy().into_owned() == "json")
            {
                let mut f = std::fs::File::open(path)?;

                // SAFETY: All existing immutable reference to `buffers` are to elements before the one we are pushing here
                unsafe {
                    let mut buffer = String::new();
                    f.read_to_string(&mut buffer);
                    let buffers: *mut Vec<String> = buffers as *const _ as _;
                    buffers.as_mut().unwrap().push(buffer);
                }

                let x = read(buffers.last().unwrap())?;
                r.push(x);
            }
        }
        Ok(Self(r))
    }

    pub fn iter(&self) -> impl Iterator<Item = &MemoryInfo> {
        self.0.iter().map(|(m, _)| m)
    }

    pub fn ref_of(&self, mem: &MemoryInfo) -> Option<&T> {
        self.0.iter().find(|(m, _)| m == mem).map(|(_, t)| t)
    }
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
    obj_def_use: &type2::object_analysis::cg_def_use::DefUse,
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
            type2::Device::Gpu(_) => Some("#A5D6A7"),
        },
        |vid, _| Some(format!("{:?}", &obj_def_use.value(vid))),
        |vid, v| {
            Some(
                v.uses()
                    .zip(obj_def_use.input_of(vid).map(|vv| format!("{:?}", vv)))
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

pub mod artifect;
pub mod fresh_type2;
pub mod fresh_type3;
pub mod processed_type2;
pub mod processed_type3;
pub mod unfused_type2;

pub use artifect::Artifect;
pub use fresh_type2::FreshType2;
pub use fresh_type3::FreshType3;
pub use processed_type2::ProcessedType2;
pub use processed_type3::ProcessedType3;
