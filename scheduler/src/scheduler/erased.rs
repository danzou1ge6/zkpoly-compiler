use std::any::Any;

use super::prelude::*;

pub trait Artifects: Any + Send + 'static {
    fn versions<'a>(&'a self) -> Box<dyn Iterator<Item = &'a MemoryInfo> + 'a>;
    fn prepare_dispatcher(
        &self,
        version: &MemoryInfo,
        pools: Pools,
        rng: AsyncRng,
        gpu_mapping: GpuMapping,
    ) -> BoxedRuntime;
}
impl<Rt: RuntimeType> Artifects for Artifect<Rt> {
    fn versions<'a>(&'a self) -> Box<dyn Iterator<Item = &'a MemoryInfo> + 'a> {
        Box::new(self.versions())
    }

    fn prepare_dispatcher(
        &self,
        version: &MemoryInfo,
        pools: Pools,
        rng: AsyncRng,
        gpu_mapping: GpuMapping,
    ) -> BoxedRuntime {
        let rt = self.prepare_dispatcher(version, pools, rng, gpu_mapping);
        Box::new(rt)
    }
}
pub type BoxedArtifect = Box<dyn Artifects>;

pub trait Variables: Any + Send + 'static {}
impl<Rt: RuntimeType> Variables for Variable<Rt> {}
pub type BoxedVariable = Box<dyn Variables>;

/// The caller is responsible for correctness of downcasted type.
pub fn downcast_variable<Rt: RuntimeType>(a: Box<dyn Variables>) -> Variable<Rt> {
    let a: Box<dyn Any + 'static> = a;
    *a.downcast().expect("downcast failed")
}

pub trait VariableTables: Any + Send + 'static {}
impl<Rt: RuntimeType, I: UsizeId + Send + 'static> VariableTables for Heap<I, Variable<Rt>> {}
pub type BoxedVariableTable = Box<dyn VariableTables>;

pub trait Runtimes: Any + Send + 'static {
    fn run(
        &mut self,
        inputs: &dyn VariableTables,
        debug_opt: RuntimeDebug,
    ) -> (Option<BoxedVariable>, Log);
}
impl<Rt: RuntimeType> Runtimes for Runtime<Rt> {
    fn run(
        &mut self,
        inputs: &dyn VariableTables,
        debug_opt: RuntimeDebug,
    ) -> (Option<BoxedVariable>, Log) {
        let inputs = (inputs as &(dyn Any + 'static))
            .downcast_ref()
            .expect("downcast failed");
        let ((r, log, _), _) = self.run(inputs, debug_opt);
        (r.map(|r| Box::new(r) as BoxedVariable), log)
    }
}
pub type BoxedRuntime = Box<dyn Runtimes>;
