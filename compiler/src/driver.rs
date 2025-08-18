//! This module provides methods to drive the compiler through various intermediate representations,
//! eventually producing the [`Artifect`] which can be run by the dispatcher.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
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

/// Controls whether and where debugging files are written to.
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
    /// Enable all debugging files, written to directory `debug_dir`.
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

    /// Enable no debugging files, written to directory `debug_dir`.
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

    /// Enable only critical debugging files, written to directory `debug_dir`.
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

    /// Configure whether to debug Type3 intermediate results.
    pub fn with_type3(mut self, switch: bool) -> Self {
        self.debug_fresh_type3 = switch;
        self.debug_extend_rewriting = switch;
        self
    }

    /// Configure whether to debug emitted instructions.
    pub fn with_inst(mut self, switch: bool) -> Self {
        self.debug_multithread_instructions = switch;
        self.debug_instructions = switch;
        self
    }

    /// Configure the Type2 visualizer to use.
    pub fn with_type2_visualizer(mut self, type2_visualizer: Type2DebugVisualizer) -> Self {
        self.type2_visualizer = type2_visualizer;
        self
    }

    /// Configure whether to print the pass being applied during compilation.
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

    pub fn with_debug_dir(self, debug_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&debug_dir).unwrap();
        Self { debug_dir, ..self }
    }
}

/// Configure a device's memory.
#[derive(Debug, PartialOrd, Ord, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct MemoryInfo {
    pub(crate) memory_limit: u64,
    pub(crate) smithereen_space: u64,
}

impl MemoryInfo {
    /// Configure the device to use `memory_limit` bytes of memory,
    /// `smithereen_space` of which are dedicated to small memory objects.
    pub fn new(memory_limit: u64, smithereen_space: u64) -> Self {
        Self {
            memory_limit,
            smithereen_space,
        }
    }

    /// Get memory limit in bytes of the configuration.
    pub fn memory_limit(&self) -> u64 {
        self.memory_limit
    }

    /// Get memory limit in giga bytes of the configuration.
    pub fn memory_limit_gigabytes(&self) -> f64 {
        self.memory_limit as f64 / 2.0_f64.powi(30)
    }

    /// Get memory space dedicated to small memory objects.
    pub fn smithereen_space(&self) -> u64 {
        self.smithereen_space
    }

    /// Get memory space dedicated to large memory objects.
    pub fn integral_space(&self) -> u64 {
        self.memory_limit - self.smithereen_space
    }

    pub(crate) fn page_number(&self, page_size: u64) -> u64 {
        self.integral_space() / page_size
    }
}

/// Configure a disk device to provide memory for computation.
#[derive(Debug, Clone)]
pub struct DiskMemoryInfo {
    disk_path: Option<PathBuf>,
}

impl DiskMemoryInfo {
    /// Configure to use the disk device holding `path` in file system.
    /// If [`None`] is provided, use system temporary directory.
    pub fn new(path: Option<PathBuf>) -> Self {
        Self { disk_path: path }
    }

    /// Get the path configured in `new`.
    pub fn disk_path(&self) -> Option<&PathBuf> {
        self.disk_path.as_ref()
    }

    pub fn has_path(&self) -> bool {
        self.disk_path.is_none()
    }
}

/// Configure various devices in system for computation.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    gpus: Vec<MemoryInfo>,
    cpu: MemoryInfo,
    disk: Vec<DiskMemoryInfo>,
    page_size: Option<u64>,
}

impl HardwareInfo {
    /// Get configured number of GPU's in the system.
    pub fn n_gpus(&self) -> usize {
        self.gpus.len()
    }

    /// Iterate through each GPU's configuration.
    pub fn gpus(&self) -> impl Iterator<Item = &MemoryInfo> {
        self.gpus.iter()
    }

    /// Iterate through each disk device's configuration.
    pub fn disks(&self) -> impl Iterator<Item = &DiskMemoryInfo> {
        self.disk.iter()
    }

    /// Get CPU memory's configuration.
    pub fn cpu(&self) -> &MemoryInfo {
        &self.cpu
    }

    pub fn smallest_gpu_memory_integral_limit(&self) -> u64 {
        self.gpus()
            .map(|gpu| gpu.memory_limit - gpu.smithereen_space)
            .min()
            .expect("no GPU")
    }

    /// Create a configuation with only one CPU, no GPU nor disk.
    pub fn new(cpu: MemoryInfo) -> Self {
        Self {
            gpus: Vec::new(),
            cpu,
            disk: Vec::new(),
            page_size: None,
        }
    }

    /// Configure page size used in the system.
    pub fn with_page_size(self, page_size: u64) -> Self {
        Self {
            page_size: Some(page_size),
            ..self
        }
    }

    /// Configure CPU.
    pub fn with_cpu(self, cpu: MemoryInfo) -> Self {
        HardwareInfo { cpu, ..self }
    }

    /// Add a disk configuation.
    pub fn with_disk(mut self, info: DiskMemoryInfo) -> Self {
        self.disk.push(info);
        self
    }

    /// Add a GPU configuation.
    pub fn with_gpu(mut self, gpu: MemoryInfo) -> Self {
        self.gpus.push(gpu);
        self
    }

    /// Get page size used in the system.
    ///
    /// # Panics
    /// When system page size is not configured
    pub fn page_size(&self) -> u64 {
        self.page_size.expect("page size not specified")
    }

    /// Get number of disks in the system.
    pub fn disks_available(&self) -> usize {
        self.disk.len()
    }

    /// Build the CPU memory pool as configured.
    pub fn cpu_allocator(&self, memory_check: bool) -> CpuStaticAllocator {
        CpuStaticAllocator::new(self.cpu().memory_limit() as usize, memory_check)
    }

    /// Build the GPU memory pools as configured, indexed by CUDA device ID's.
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

    /// Build the GPU memory pools as configured, for a subset of GPU's identified by
    /// CUDA device ID's.
    pub fn gpu_allocators_for(
        &self,
        memory_check: bool,
        device_ids: impl Iterator<Item = i32>,
    ) -> HashMap<i32, CudaAllocator> {
        use zkpoly_cuda_api::mem;
        device_ids
            .map(|id| {
                let gpu = &self.gpus[id as usize];
                (
                    id as i32,
                    CudaAllocator {
                        statik: mem::StaticAllocator::new(
                            id,
                            gpu.memory_limit() as usize,
                            memory_check,
                        ),
                        page: mem::PageAllocator::new(
                            zkpoly_common::devices::DeviceType::GPU { device_id: id },
                            self.page_size() as usize,
                            0,
                        ),
                    },
                )
            })
            .collect()
    }

    /// Build the Disk memory pools as configured.
    pub fn disk_allocator(&self, max_block: usize) -> DiskMemoryPool {
        self.disks()
            .map(|disk_info| {
                let disk_path = disk_info.disk_path().cloned();
                BuddyDiskPool::new(max_block, disk_path).expect("cannot create disk pool")
            })
            .collect()
    }

    pub(crate) fn smaller_cpus<'a>(
        &'a self,
        divisions: impl Iterator<Item = u32> + 'a,
    ) -> impl Iterator<Item = MemoryInfo> + 'a {
        divisions.map(|p| {
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

    fn dirname(&self, f: &'static str) -> String {
        format!(
            "{}_{}",
            f,
            human_readable_size(self.mem_info.memory_limit())
        )
    }
}

#[derive(Clone)]
struct Versions<T>(Vec<(MemoryInfo, T)>);

impl<T> Versions<T> {
    pub fn build<'s, Rt: RuntimeType>(
        hardware_info: &HardwareInfo,
        divisions: impl Iterator<Item = u32>,
        mut f: impl for<'a> FnMut(&'a MemoryInfo, VersionsHelper<'a>) -> Result<T, Error<'s, Rt>>,
    ) -> Result<Self, Error<'s, Rt>> {
        let infos = hardware_info.smaller_cpus(divisions).collect::<Vec<_>>();
        infos.iter().fold(HashMap::<String, MemoryInfo>::new(), |mut acc, e| {
            let s = human_readable_size(e.memory_limit());
            if let Some(m) = acc.get(&s) {
                panic!("We are distinguishing versions of artifect using human readable CPU memory limits, but your configuration of versions are undistinguishable: One with size {}({}) and the other with {}", e.memory_limit(), s, m.memory_limit())
            } else {
                acc.insert(s, e.clone());
                acc
            }
        });

        Ok(Self(
            infos
                .into_iter()
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

    pub fn map_i<'s, T2, Rt: RuntimeType>(
        self,
        mut f: impl for<'a> FnMut(
            &'a MemoryInfo,
            T,
            VersionsHelper<'a>,
        ) -> Result<(MemoryInfo, T2), Error<'s, Rt>>,
    ) -> Result<Versions<T2>, Error<'s, Rt>> {
        let mut existent_versions = BTreeSet::new();

        Ok(Versions(
            self.0
                .into_iter()
                .filter_map(|(cpu, t)| {
                    let (cpu, t2) = match f(&cpu, t, VersionsHelper { mem_info: &cpu }) {
                        Ok(x) => x,
                        Err(e) => return Some(Err(e)),
                    };
                    if existent_versions.contains(&cpu) {
                        None
                    } else {
                        existent_versions.insert(cpu.clone());
                        Some(Ok((cpu, t2)))
                    }
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
                .is_some_and(|f| f.to_string_lossy().into_owned().starts_with("version_"))
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
                .is_some_and(|f| f.to_string_lossy().into_owned().starts_with("version_"))
                && path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.to_string_lossy().into_owned() == "json")
            {
                let mut f = std::fs::File::open(path)?;

                let mut buffer = String::new();
                f.read_to_string(&mut buffer)?;

                // SAFETY: All existing immutable reference to `buffers` are to elements before the one we are pushing here
                unsafe {
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

    pub fn iter_items(&self) -> impl Iterator<Item = &(MemoryInfo, T)> {
        self.0.iter()
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

fn debug_partial_typed_type2<'s, Ty: std::fmt::Debug, Ts: std::fmt::Debug>(
    fpath: PathBuf,
    g: &Digraph<type2::VertexId, type2::partial_typed::Vertex<'s, Option<Ty>, Ts>>,
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

fn check_type2_unsubgraphed_dag<'s, Rt: RuntimeType>(
    fpath: PathBuf,
    g: &type2::no_subgraph::Cg<'s, Rt>,
    visualizer: Type2DebugVisualizer,
) -> bool {
    if let Some(cycle) = g.g.try_find_cycle() {
        type2::pretty::unsubgraphed_cg::debug_cycle(g, |i| cycle[i], fpath).expect("io error");
        false
    } else {
        true
    }
}

fn check_type2_subgraphed_dag<'s, Rt: RuntimeType>(
    fpath: PathBuf,
    g: &type2::Cg<'s, Rt>,
    visualizer: Type2DebugVisualizer,
) -> bool {
    if let Some(cycle) = g.g.try_find_cycle() {
        type2::pretty::subgraphed_cg::debug_cycle(g, |i| cycle[i], fpath).expect("io error");
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
/// Used for preventing panics to cause subprocesses writing debug files to exit.
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
