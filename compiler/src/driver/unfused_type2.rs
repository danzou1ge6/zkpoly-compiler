use zkpoly_common::load_dynamic::Libs;
use zkpoly_runtime::args::RuntimeType;

use super::{processed_type2::ProcessedType2, Error};
use crate::{
    driver::{
        check_type2_dag, debug_type2, DebugOptions, HardwareInfo, PanicJoinHandler, Versions,
    },
    transit::type2,
};

/// The Type2 IR before kernel fusion.
pub struct UnfusedType2<'s, Rt: RuntimeType> {
    pub(super) cg: type2::Cg<'s, Rt>,
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
        ctx: &PanicJoinHandler,
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

                if !check_type2_dag(
                    options
                        .debug_dir
                        .join(helper.debug_filename("type2_kernel_fusion.dot")),
                    &t2cg.g,
                    t2cg.output,
                    options.type2_visualizer,
                ) {
                    panic!("graph is not a DAG after Arithmetic Kernel Fusion");
                }

                if options.debug_kernel_fusion {
                    ctx.add(debug_type2(
                        options.debug_dir.join("type2_kernel_fusion.dot"),
                        &t2cg.g,
                        t2cg.output,
                        options.type2_visualizer,
                    ));
                }

                Ok(t2cg)
            },
        )?;

        Ok(ProcessedType2 {
            cg: t2cg_versions,
            constant_table,
            uf_table,
            libs,
        })
    }
}
