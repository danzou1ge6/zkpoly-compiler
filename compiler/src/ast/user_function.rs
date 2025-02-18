use super::*;
use zkpoly_runtime::error::{Result as RuntimeResult, RuntimeError};

pub type ValueMut<Rt: RuntimeType> =
    Box<dyn FnMut(Vec<&Variable<Rt>>) -> RuntimeResult<Variable<Rt>> + Send + Sync + 'static>;

pub type ValueOnce<Rt: RuntimeType> =
    Box<dyn FnOnce(Vec<&Variable<Rt>>) -> RuntimeResult<Variable<Rt>> + Send + Sync + 'static>;

pub type ValueFn<Rt: RuntimeType> =
    Box<dyn Fn(Vec<&Variable<Rt>>) -> RuntimeResult<Variable<Rt>> + Send + Sync + 'static>;

pub enum FunctionValue<Rt: RuntimeType> {
    Mut(ValueMut<Rt>),
    Once(ValueOnce<Rt>),
    Fn(ValueFn<Rt>),
}

pub struct FunctionData<Rt: RuntimeType> {
    node: FunctionValue<Rt>,
    name: String,
}

impl<Rt: RuntimeType> std::fmt::Debug for FunctionData<Rt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Function({})", self.name)
    }
}

pub(super) type FunctionUntyped<Rt: RuntimeType> = Outer<FunctionData<Rt>>;

impl<Rt: RuntimeType> FunctionUntyped<Rt> {
    pub fn new_fn(name: String, f: ValueFn<Rt>, src: SourceInfo) -> Self {
        FunctionUntyped::new(
            FunctionData {
                node: FunctionValue::Fn(f),
                name,
            },
            src,
        )
    }
}

pub type Function2<Rt: RuntimeType, T0, T1, R> = Phantomed<FunctionUntyped<Rt>, (T0, T1, R)>;

impl<Rt: RuntimeType, T0, T1, R> Function2<Rt, T0, T1, R>
where
    T1: TypeEraseable<Rt>,
    T0: TypeEraseable<Rt>,
    R: CommonConstructors<Rt>,
{
    #[track_caller]
    pub fn call(&self, t0: T0, t1: T1) -> R {
        let args = vec![AstVertex::new(t0), AstVertex::new(t1)];
        let src = SourceInfo::new(Location::caller().clone(), None);
        R::from_function_call(self.t.clone(), args, src)
    }
}

impl<Rt: RuntimeType, T0, T1, R> Function2<Rt, T0, T1, R>
where
    T1: RuntimeCorrespondance<Rt>,
    T0: RuntimeCorrespondance<Rt>,
    R: RuntimeCorrespondance<Rt>,
{
    #[track_caller]
    pub fn new_fn(
        name: String,
        f: impl for<'a, 'b> Fn(T0::RtcBorrowed<'a>, T1::RtcBorrowed<'b>) -> RuntimeResult<R::Rtc> + Send + Sync + 'static,
    ) -> Self {
        let f = move |args: Vec<&Variable<Rt>>| -> RuntimeResult<Variable<Rt>> {
            let arg0 = T0::try_borrow_variable(&args[0])
                .ok_or(RuntimeError::VariableTypError)?;
            let arg1 = T1::try_borrow_variable(&args[1])
                .ok_or(RuntimeError::VariableTypError)?;
            let r = f(arg0, arg1)?;
            Ok(R::to_variable(r))
        };
        let src = SourceInfo::new(Location::caller().clone(), Some(name.clone()));
        Function2::wrap(FunctionUntyped::new_fn(name, Box::new(f), src))
    }
}
