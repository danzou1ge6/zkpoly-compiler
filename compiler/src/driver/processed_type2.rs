use crate::transit::type3;

use super::fresh_type3::FreshType3;
use std::io::Write;
use zkpoly_common::load_dynamic::Libs;
use zkpoly_memory_pool::PinnedMemoryPool;
use zkpoly_runtime::args::{self, RuntimeType};

use super::{
    ast, check_type2_dag, debug_partial_typed_type2, debug_type2, debug_type2_def_use,
    debug_type2_with_seq, type2, DebugOptions, Error, HardwareInfo, PanicJoinHandler, SubDigraph,
};

pub struct ProcessedType2<'s, Rt: RuntimeType> {
    pub(super) cg: type2::Cg<'s, Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) libs: Libs,
    pub(super) allocator: PinnedMemoryPool,
}

impl<'s, Rt: RuntimeType> ProcessedType2<'s, Rt> {
    pub fn to_type3(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        ctx: &PanicJoinHandler,
    ) -> Result<FreshType3<'s, Rt>, Error<'static, Rt>> {
        let Self {
            cg: t2cg,
            constant_table: t2const_tab,
            uf_table: t2uf_tab,
            libs,
            allocator,
        } = self;

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

        let g = SubDigraph::new(&t2cg.g, t2cg.g.connected_component(t2cg.output));

        // - Decide Device
        let devices = options.log_suround(
            "Deciding device",
            || Ok(type2::decide_device::decide(&g)),
            "Done.",
        )?;

        // - Object Analysis
        let (mut obj_def, mut obj_id_allocator) = options.log_suround(
            "Analyzing object definitions (before deciding arith inputs)",
            || {
                Ok(type2::object_analysis::analyze_def(&g, &seq, |vid| {
                    devices[&vid]
                }))
            },
            "Done.",
        )?;

        let vertex_inputs = options.log_suround(
            "Planning vertex inputs (before deciding arith inputs)",
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

        let obj_def = obj_def;
        if options.debug_obj_def_use {
            let fpath = options
                .debug_dir
                .join("type2_object_def_use_before_arith_decide_mutable.dot");
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

        let obj_dies_after = options.log_suround(
            "Analyzing object lifetimes (before deciding arith inputs)",
            || {
                let d = type2::object_analysis::analyze_die_after(
                    &g,
                    &seq,
                    &obj_def,
                    &vertex_inputs,
                    &vertex_inputs,
                );
                Ok(d)
            },
            "Done",
        )?;

        if options.debug_obj_liveness {
            let fpath = options
                .debug_dir
                .join("type2_object_liveness_before_arith_decide_mutable.txt");
            let mut f = std::fs::File::create(&fpath).unwrap();
            obj_dies_after.iter().for_each(|(d, m)| {
                write!(f, "{:?}\n{:?}\n", d, m).unwrap();
            });
        }

        // - Decide inplace inputs of arithmetic subgraphs
        let t2cg = options.log_suround(
            "Deciding inplace inputs",
            || {
                Ok(type2::arith_decide_mutable::decide_mutable(
                    t2cg,
                    &obj_def,
                    &obj_dies_after,
                ))
            },
            "Done",
        )?;

        if options.debug_arith_decide_mutable {
            let fpath = options.debug_dir.join("type2_arith_decide_mutable.dot");
            ctx.add(debug_type2(
                fpath,
                &t2cg.g,
                t2cg.output,
                options.type2_visualizer,
            ));
        }

        // - Object Analysis Again (since the mutable decision might have changed the value definition)
        let g = SubDigraph::new(&t2cg.g, t2cg.g.connected_component(t2cg.output));

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

        let obj_def = obj_def;
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

        let (obj_dies_after, obj_dies_after_reversed) = options.log_suround(
            "Analyzing object lifetimes",
            || {
                let d = type2::object_analysis::analyze_die_after(
                    &g,
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

        // - Analysis Object Use Chain for Cache Planning
        let (obj_used_by, obj_gpu_next_use) = options.log_suround(
            "Analyzing next uses of objects on GPU",
            || {
                let obj_used_by = type2::object_analysis::analyze_used_by(&seq, &vertex_inputs);
                let obj_gpu_next_use = type2::object_analysis::analyze_gpu_next_use(
                    &g,
                    &seq,
                    &obj_def,
                    &vertex_inputs,
                    &obj_used_by,
                );

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

        if hardware_info.gpu_smithereen_space > hardware_info.gpu_memory_limit {
            panic!("you cannot have a smithereen space larger than the gpu memory limit");
        }

        // To Type3 through Memory Planning
        let g = SubDigraph::new(&t2cg.g, t2cg.g.connected_component(t2cg.output));
        let t3chunk = options.log_suround(
            "Planning memory",
            || {
                Ok(type2::memory_planning::plan(
                    hardware_info.gpu_memory_limit,
                    hardware_info.gpu_smithereen_space,
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
                    obj_id_allocator,
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

        Ok(FreshType3 {
            chunk: t3chunk,
            uf_table: t2uf_tab,
            constant_table: t2const_tab,
            allocator,
        })
    }

    pub fn dump(&self, dir: impl AsRef<std::path::Path>) -> std::io::Result<()>
    where
        Rt: for<'de> serde::Deserialize<'de> + serde::Serialize,
    {
        std::fs::create_dir_all(dir.as_ref())?;

        let mut cg_f = std::fs::File::create(dir.as_ref().join("cg.json"))?;
        serde_json::to_writer_pretty(&mut cg_f, &self.cg)?;

        let mut ct_header_f = std::fs::File::create(dir.as_ref().join("constants-manifest.json"))?;
        let ct_header = args::serialization::Header::build(&self.constant_table);
        serde_json::to_writer_pretty(&mut ct_header_f, &ct_header)?;

        let mut ct_f = std::fs::File::create(dir.as_ref().join("constants.bin"))?;
        ct_header.dump_entries_data(&self.constant_table, &mut ct_f)?;

        Ok(())
    }
}
