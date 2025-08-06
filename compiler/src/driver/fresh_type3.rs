use super::{processed_type3::ProcessedType3, type3};
use zkpoly_common::{heap::Heap, load_dynamic::Libs};
use zkpoly_runtime::args::{ConstantId, RuntimeType};

use super::{type2, DebugOptions, Error, Versions};

pub struct Version<'s, Rt: RuntimeType> {
    pub chunk: type3::Chunk<'s, Rt>,
    pub memory_statistics: type2::memory_planning::Statistics,
}

pub struct FreshType3<'s, Rt: RuntimeType> {
    pub(super) versions: Versions<Version<'s, Rt>>,
    pub(super) uf_table: type2::user_function::Table<Rt>,
    pub(super) constant_table: type2::ConstantTable<Rt>,
    pub(super) constants_device: Heap<ConstantId, type3::Device>,
    pub(super) libs: Libs,
}

impl<'s, Rt: RuntimeType> FreshType3<'s, Rt> {
    pub fn apply_passes<'a>(
        self,
        options: &'a DebugOptions,
    ) -> Result<ProcessedType3<'s, Rt>, Error<'s, Rt>> {
        let versions = self.versions.map(|_cpu, version, helper| {
            // - Extend Rewritting
            let chunk = options.log_suround(
                helper.log_prologue("Rewritting Extend and NewPoly"),
                || Ok(type3::rewrite_extend::rewrite(version.chunk)),
                "Done.",
            )?;

            if options.debug_extend_rewriting {
                let mut f = std::fs::File::create(
                    options
                        .debug_dir
                        .join(helper.debug_filename("type3_extend_rewriting.html")),
                )
                .unwrap();
                type3::pretty_print::prettify(&chunk, &mut f).unwrap();
            }

            Ok(Version {
                chunk,
                memory_statistics: version.memory_statistics,
            })
        })?;

        Ok(ProcessedType3 {
            versions,
            uf_table: self.uf_table,
            constant_table: self.constant_table,
            constants_device: self.constants_device,
            libs: self.libs,
        })
    }
}
