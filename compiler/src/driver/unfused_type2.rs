use zkpoly_common::load_dynamic::Libs;
use zkpoly_runtime::args::RuntimeType;

use super::{processed_type2::ProcessedType2, Error};
use crate::{
    driver::{
        check_type2_subgraphed_dag, check_type2_unsubgraphed_dag, DebugOptions, HardwareInfo,
        PanicJoinHandler, Versions,
    },
    transit::type2,
};

/// The Type2 IR before kernel fusion.
pub struct UnfusedType2<'s, Rt: RuntimeType> {
    pub(super) cg: type2::no_subgraph::Cg<'s, Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) libs: Libs,
}

impl<'s, Rt: RuntimeType> UnfusedType2<'s, Rt> {
    /// Apply kernel fusion.
    ///
    /// `versions_cpu_memory_divisons` control different versions to compile for different CPU memory usage.
    /// For example, `[0, 1, 2]` with CPU memory size set to 64GB yields versions that run with
    /// 64GB, 32GB and 16GB.
    pub fn fuse(
        self,
        options: &DebugOptions,
        hardware_info: &HardwareInfo,
        versions_cpu_memory_divisions: impl Iterator<Item = u32>,
        _ctx: &PanicJoinHandler,
    ) -> Result<ProcessedType2<'s, Rt>, Error<'s, Rt>> {
        let UnfusedType2 {
            cg: t2cg,
            constant_table,
            uf_table,
            libs,
        } = self;

        // - Arithmetic Kernel Fusion
        let t2cg_versions = Versions::build(
            hardware_info,
            versions_cpu_memory_divisions,
            |cpu_size, helper| {
                let t2cg = options.log_suround(
                    helper.log_prologue("Fusing arithmetic kernels"),
                    || {
                        Ok(type2::kernel_fusion::fuse_arith(
                            t2cg.clone(),
                            &hardware_info.clone().with_cpu(cpu_size.clone()),
                        ))
                    },
                    "Done.",
                )?;

                if !check_type2_unsubgraphed_dag(
                    options
                        .debug_dir
                        .join(helper.debug_filename("type2_kernel_fusion.dot")),
                    &t2cg,
                    options.type2_visualizer,
                ) {
                    panic!("graph is not a DAG after Arithmetic Kernel Fusion");
                }

                if options.debug_kernel_fusion {
                    type2::pretty::unsubgraphed_cg::debug_graph(
                        &t2cg,
                        options.debug_dir.join("type2_kernel_fusion.dot"),
                    );
                }

                Ok(t2cg)
            },
        )?;

        let t2cg_versions = t2cg_versions.map(|_cpu, t2cg, helper| {
            let t2cg = type2::subgraph_slicing::fuse(t2cg, 100);

            if !check_type2_subgraphed_dag(
                options
                    .debug_dir
                    .join(helper.debug_filename("type2_subgraph_partition.dot")),
                &t2cg,
                options.type2_visualizer,
            ) {
                panic!("graph is not a DAG after Arithmetic Kernel Fusion");
            }

            if options.debug_kernel_fusion {
                type2::pretty::subgraphed_cg::debug_graph(
                    &t2cg,
                    options.debug_dir.join("type2_subgraph_partition.dot"),
                );
            }

            Ok(t2cg)
        })?;

        Ok(ProcessedType2 {
            cg: t2cg_versions,
            constant_table,
            uf_table,
            libs,
        })
    }
}
