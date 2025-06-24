use crate::{ast, transit::type3};
use std::io::Write;

use super::{fresh_type3::FreshType3, ConstantPool};
use zkpoly_common::load_dynamic::Libs;
use zkpoly_runtime::args::{self, RuntimeType};

use super::{
    debug_type2, debug_type2_def_use, debug_type2_with_seq, type2, DebugOptions, Error,
    HardwareInfo, PanicJoinHandler, SubDigraph,
};

pub struct ProcessedType2<'s, Rt: RuntimeType> {
    pub(super) cg: type2::Cg<'s, Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) libs: Libs,
}

impl<'s, Rt: RuntimeType> ProcessedType2<'s, Rt> {
    pub fn to_type3(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        constant_pool: &mut ConstantPool,
        ctx: &PanicJoinHandler,
    ) -> Result<FreshType3<'s, Rt>, Error<'static, Rt>> {
        let Self {
            cg: t2cg,
            constant_table: t2const_tab,
            uf_table: t2uf_tab,
            libs,
        } = self;

        if hardware_info.disks_available() != constant_pool.disk.as_ref().map_or(0, |x| x.len()) {
            panic!("inconsistent number of disks in hardware info and constant pool");
        }

        // - Decide constants to be on disk or CPU:
        //   If the constants take more than 1/4 CPU space, put big constants on disk.
        let constants_on_disk = ast::lowering::constant_size(&t2const_tab) * 4
            > hardware_info.cpu().memory_limit()
            && hardware_info.disks_available() > 0;
        let constants_device = t2const_tab.map_by_ref(&mut |_, c| {
            if constants_on_disk && c.typ.can_on_disk::<Rt::Field, Rt::PointAffine>() {
                type3::Device::Disk
            } else {
                type3::Device::Cpu
            }
        });

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
        let (obj_def_use, _) = options.log_suround(
            "Analyzing object definitions (before deciding arith inputs)",
            || {
                Ok(type2::object_analysis::cg_def_use::DefUse::analyze(
                    &g,
                    &t2uf_tab,
                    &seq,
                    t2cg.output,
                    |vid| devices[&vid],
                    hardware_info,
                    &constants_device,
                ))
            },
            "Done.",
        )?;

        if options.debug_obj_def_use {
            let fpath = options
                .debug_dir
                .join("type2_object_def_use_before_arith_decide_mutable.dot");
            ctx.add(debug_type2_def_use(
                fpath,
                &t2cg.g,
                &devices,
                &seq,
                &obj_def_use,
                options.type2_visualizer,
            ));
        }

        if options.debug_obj_liveness {
            let fpath = options
                .debug_dir
                .join("type2_object_liveness_before_arith_decide_mutable.txt");
            let mut f = std::fs::File::create(&fpath).unwrap();
            obj_def_use.debug_dies_after(&mut f).unwrap();
        }

        // - Decide inplace inputs of arithmetic subgraphs
        let t2cg = options.log_suround(
            "Deciding inplace inputs",
            || {
                Ok(type2::arith_decide_mutable::decide_mutable(
                    t2cg,
                    &seq,
                    &obj_def_use,
                    |vid| devices[&vid],
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

        let (obj_def_use, obj_id_allocator) = options.log_suround(
            "Analyzing object definitions",
            || {
                Ok(type2::object_analysis::cg_def_use::DefUse::analyze(
                    &g,
                    &t2uf_tab,
                    &seq,
                    t2cg.output,
                    |vid| devices[&vid],
                    hardware_info,
                    &constants_device,
                ))
            },
            "Done.",
        )?;

        if options.debug_obj_def_use {
            let fpath = options.debug_dir.join("type2_object_def_use.dot");
            ctx.add(debug_type2_def_use(
                fpath,
                &t2cg.g,
                &devices,
                &seq,
                &obj_def_use,
                options.type2_visualizer,
            ));
        }

        // To Type3 through Memory Planning
        let g = SubDigraph::new(&t2cg.g, t2cg.g.connected_component(t2cg.output));

        let (t3chunk, statistics) = options.log_suround(
            "Planning memory",
            || {
                Ok(type2::memory_planning::plan(
                    &t2cg,
                    &g,
                    &seq,
                    &obj_def_use,
                    obj_id_allocator,
                    |vid| devices[&vid],
                    hardware_info,
                    libs,
                )
                .expect("memory plan failed"))
            },
            "Done.",
        )?;

        if options.debug_fresh_type3 {
            let mut f = std::fs::File::create(options.debug_dir.join("type3_fresh.html")).unwrap();
            type3::pretty_print::prettify(&t3chunk, |vid| devices[&vid], &mut f).unwrap();
        }

        if options.debug_memory_planning_statistics {
            let mut f =
                std::fs::File::create(options.debug_dir.join("memory_planning_statistics.txt"))
                    .unwrap();
            write!(&mut f, "{:?}", &statistics).unwrap();
            println!("Memory Planning Statistics:\n{:?}", &statistics);
        }

        Ok(FreshType3 {
            chunk: t3chunk,
            uf_table: t2uf_tab,
            constant_table: t2const_tab,
            constants_device,
            execution_devices: devices,
        })
    }

    pub fn dump(
        &mut self,
        dir: impl AsRef<std::path::Path>,
        constant_pool: &mut ConstantPool,
    ) -> std::io::Result<()>
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
        ct_header.dump_entries_data(&self.constant_table, &mut ct_f, &mut constant_pool.cpu)?;

        Ok(())
    }
}
