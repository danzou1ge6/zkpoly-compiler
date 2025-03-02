use super::*;
use std::cell::Cell;
use zkpoly_runtime::error::{Result as RuntimeResult, RuntimeError};

pub type ValueMut<Rt: RuntimeType> = Box<
    dyn FnMut(&mut Variable<Rt>, Vec<&Variable<Rt>>) -> RuntimeResult<()> + Send + Sync + 'static,
>;

pub type ValueOnce<Rt: RuntimeType> = Box<
    dyn FnOnce(&mut Variable<Rt>, Vec<&Variable<Rt>>) -> RuntimeResult<()> + Send + Sync + 'static,
>;

pub type ValueFn<Rt: RuntimeType> =
    Box<dyn Fn(&mut Variable<Rt>, Vec<&Variable<Rt>>) -> RuntimeResult<()> + Send + Sync + 'static>;

pub enum Value<Rt: RuntimeType> {
    Mut(ValueMut<Rt>),
    Once(ValueOnce<Rt>),
    Fn(ValueFn<Rt>),
}

pub struct FunctionInCell<Rt: RuntimeType> {
    n_args: usize,
    value: Cell<Option<Value<Rt>>>,
    name: String,
    pub(super) ret_typ: type2::Typ<Rt>,
}

impl<Rt: RuntimeType> FunctionInCell<Rt> {
    pub fn take(&self) -> Function<Rt> {
        let value = self.value.take().unwrap();
        Function {
            n_args: self.n_args,
            value,
            name: self.name.clone(),
            ret_typ: self.ret_typ.clone(),
        }
    }
}

impl<Rt: RuntimeType> std::fmt::Debug for FunctionInCell<Rt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Function({})", self.name)
    }
}

pub struct Function<Rt: RuntimeType> {
    pub(crate) n_args: usize,
    pub(crate) value: Value<Rt>,
    pub(crate) name: String,
    pub(crate) ret_typ: type2::Typ<Rt>,
}

pub(super) type FunctionUntyped<Rt: RuntimeType> = Outer<FunctionInCell<Rt>>;

impl<Rt: RuntimeType> FunctionUntyped<Rt> {
    pub fn new_fn(
        name: String,
        f: ValueFn<Rt>,
        n_args: usize,
        ret_typ: type2::Typ<Rt>,
        src: SourceInfo,
    ) -> Self {
        FunctionUntyped::new(
            FunctionInCell {
                n_args,
                value: Cell::new(Some(Value::Fn(f))),
                name,
                ret_typ,
            },
            src,
        )
    }

    pub fn new_mut(
        name: String,
        f: ValueMut<Rt>,
        n_args: usize,
        ret_typ: type2::Typ<Rt>,
        src: SourceInfo,
    ) -> Self {
        FunctionUntyped::new(
            FunctionInCell {
                n_args,
                value: Cell::new(Some(Value::Mut(f))),
                name,
                ret_typ,
            },
            src,
        )
    }

    pub fn new_once(
        name: String,
        f: ValueOnce<Rt>,
        n_args: usize,
        ret_typ: type2::Typ<Rt>,
        src: SourceInfo,
    ) -> Self {
        FunctionUntyped::new(
            FunctionInCell {
                n_args,
                value: Cell::new(Some(Value::Once(f))),
                name,
                ret_typ,
            },
            src,
        )
    }
}

pub struct FnMarker;
pub struct MutMarker;
pub struct OnceMarker;

pub type FunctionFn0<Rt: RuntimeType, R> = Phantomed<FunctionUntyped<Rt>, (R, FnMarker)>;

impl<Rt: RuntimeType, R> FunctionFn0<Rt, R>
where
    R: CommonConstructors<Rt>,
{
    #[track_caller]
    pub fn call(&self) -> R {
        let args = vec![];
        let src = SourceInfo::new(Location::caller().clone(), None);
        R::from_function_call(self.t.clone(), args, src)
    }
}

impl<Rt: RuntimeType, R> FunctionFn0<Rt, R>
where
    R: RuntimeCorrespondance<Rt>,
{
    #[track_caller]
    pub fn new(
        name: String,
        f: impl for<'a> Fn(R::RtcBorrowedMut<'a>) -> RuntimeResult<()> + Send + Sync + 'static,
        ret_typ: type2::Typ<Rt>,
    ) -> Self {
        let f = move |r: &mut Variable<Rt>, _args: Vec<&Variable<Rt>>| -> RuntimeResult<()> {
            let r = R::try_borrow_variable_mut(r).ok_or(RuntimeError::VariableTypError)?;
            f(r)?;
            Ok(())
        };
        let src = SourceInfo::new(Location::caller().clone(), Some(name.clone()));
        Self::wrap(FunctionUntyped::new_fn(name, Box::new(f), 0, ret_typ, src))
    }
}

macro_rules! define_function_fn {
    ($($n:tt $m:tt => ($($i:tt $arg:ident $T:ident),+)),*) => {
        $(
            pub type $n<Rt: RuntimeType, $($T), +, R> = Phantomed<FunctionUntyped<Rt>, ($($T), +, R, FnMarker)>;

            impl<Rt: RuntimeType, $($T: TypeEraseable<Rt>), +, R> $n<Rt, $($T), +, R>
            where
                R: CommonConstructors<Rt>,
            {
                #[track_caller]
                pub fn call(&self, $($arg: $T),+) -> R {
                    let args = vec![$(AstVertex::new($arg)), +,];
                    let src = SourceInfo::new(Location::caller().clone(), None);
                    R::from_function_call(self.t.clone(), args, src)
                }
            }

            impl<Rt: RuntimeType, $($T: RuntimeCorrespondance<Rt>), +, R> $n<Rt, $($T), +, R>
            where
                R: RuntimeCorrespondance<Rt>,
            {
                #[track_caller]
                pub fn new(
                    name: String,
                    f: impl for<'a> Fn(R::RtcBorrowedMut<'a>, $($T::RtcBorrowed<'a>),+) -> RuntimeResult<()>
                        + Send
                        + Sync
                        + 'static,
                    ret_typ: type2::Typ<Rt>,
                ) -> Self {
                    let f = move |r: &mut Variable<Rt>, args: Vec<&Variable<Rt>>| -> RuntimeResult<()> {
                        $(
                            let $arg = $T::try_borrow_variable(&args[$i]).ok_or(RuntimeError::VariableTypError)?;
                        )+
                        let r = R::try_borrow_variable_mut(r)
                            .ok_or(RuntimeError::VariableTypError)?;
                        f(r, $($arg),+)
                    };
                    let src = SourceInfo::new(Location::caller().clone(), Some(name.clone()));
                    Self::wrap(FunctionUntyped::new_fn(name, Box::new(f), $m, ret_typ, src))
                }
            }
        )*
    };
}

define_function_fn! {
    FunctionFn1 1 => (0 arg0 T0),
    FunctionFn2 2 => (0 arg0 T0, 1 arg1 T1),
    FunctionFn3 3 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2),
    FunctionFn4 4 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3),
    FunctionFn5 5 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4),
    FunctionFn6 6 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4, 5 arg5 T5),
    FunctionFn7 7 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4, 5 arg5 T5, 6 arg6 T6)
}

pub struct PhantomedNoClone<T, P> {
    t: T,
    p: PhantomData<P>,
}

impl<T: Debug, P> Debug for PhantomedNoClone<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Phantomed({:?})", self.t)
    }
}

impl<T, P> PhantomedNoClone<T, P> {
    pub fn wrap(t: T) -> Self {
        Self { t, p: PhantomData }
    }
}

macro_rules! define_function_mut {
    ($($n:tt $m:tt => ($($i:tt $arg:ident $T:ident),+)),*) => {
        $(
            pub type $n<Rt: RuntimeType, $($T), +, R> = PhantomedNoClone<FunctionUntyped<Rt>, ($($T), +, R, MutMarker)>;

            impl<Rt: RuntimeType, $($T: TypeEraseable<Rt>), +, R> $n<Rt, $($T), +, R>
            where
                R: CommonConstructors<Rt>,
            {
                #[track_caller]
                pub fn call(&mut self, $($arg: $T),+) -> R {
                    let args = vec![$(AstVertex::new($arg)), +,];
                    let src = SourceInfo::new(Location::caller().clone(), None);
                    R::from_function_call(self.t.clone(), args, src)
                }
            }

            impl<Rt: RuntimeType, $($T: RuntimeCorrespondance<Rt>), +, R> $n<Rt, $($T), +, R>
            where
                R: RuntimeCorrespondance<Rt>,
            {
                #[track_caller]
                pub fn new(
                    name: String,
                    mut f: impl for<'a> FnMut(R::RtcBorrowedMut<'a>, $($T::RtcBorrowed<'a>),+) -> RuntimeResult<()>
                        + Send
                        + Sync
                        + 'static,
                    ret_typ: type2::Typ<Rt>,
                ) -> Self {
                    let f = move |r: &mut Variable<Rt>, args: Vec<&Variable<Rt>>| -> RuntimeResult<()> {
                        $(
                            let $arg = $T::try_borrow_variable(&args[$i]).ok_or(RuntimeError::VariableTypError)?;
                        )+
                        let r = R::try_borrow_variable_mut(r).ok_or(RuntimeError::VariableTypError)?;
                        f(r, $($arg),+)
                    };
                    let src = SourceInfo::new(Location::caller().clone(), Some(name.clone()));
                    Self::wrap(FunctionUntyped::new_mut(name, Box::new(f), $m, ret_typ, src))
                }
            }
        )*
    };
}

define_function_mut! {
    FunctionMut1 1 => (0 arg0 T0),
    FunctionMut2 2 => (0 arg0 T0, 1 arg1 T1),
    FunctionMut3 3 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2),
    FunctionMut4 4 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3),
    FunctionMut5 5 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4),
    FunctionMut6 6 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4, 5 arg5 T5),
    FunctionMut7 7 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4, 5 arg5 T5, 6 arg6 T6)
}

macro_rules! define_function_once {
    ($($n:tt $m:tt => ($($i:tt $arg:ident $T:ident),+)),*) => {
        $(
            pub type $n<Rt: RuntimeType, $($T), +, R> = PhantomedNoClone<FunctionUntyped<Rt>, ($($T), +, R, OnceMarker)>;

            impl<Rt: RuntimeType, $($T: TypeEraseable<Rt>), +, R> $n<Rt, $($T), +, R>
            where
                R: CommonConstructors<Rt>,
            {
                #[track_caller]
                pub fn call(self, $($arg: $T),+) -> R {
                    let args = vec![$(AstVertex::new($arg)), +,];
                    let src = SourceInfo::new(Location::caller().clone(), None);
                    R::from_function_call(self.t.clone(), args, src)
                }
            }

            impl<Rt: RuntimeType, $($T: RuntimeCorrespondance<Rt>), +, R> $n<Rt, $($T), +, R>
            where
                R: RuntimeCorrespondance<Rt>,
            {
                #[track_caller]
                pub fn new(
                    name: String,
                    f: impl for<'a> FnOnce(R::RtcBorrowedMut<'a>, $($T::RtcBorrowed<'a>),+) -> RuntimeResult<()>
                        + Send
                        + Sync
                        + 'static,
                    ret_typ: type2::Typ<Rt>,
                ) -> Self {
                    let f = move |r: &mut Variable<Rt>, args: Vec<&Variable<Rt>>| -> RuntimeResult<()> {
                        $(
                            let $arg = $T::try_borrow_variable(args[$i]).ok_or(RuntimeError::VariableTypError)?;
                        )+
                        let r = R::try_borrow_variable_mut(r).ok_or(RuntimeError::VariableTypError)?;
                        f(r, $($arg),+)
                    };
                    let src = SourceInfo::new(Location::caller().clone(), Some(name.clone()));
                    Self::wrap(FunctionUntyped::new_once(name, Box::new(f), $m, ret_typ, src))
                }
            }
        )*
    };
}

define_function_once! {
    FunctionOnce1 1 => (0 arg0 T0),
    FunctionOnce2 2 => (0 arg0 T0, 1 arg1 T1),
    FunctionOnce3 3 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2),
    FunctionOnce4 4 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3),
    FunctionOnce5 5 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4),
    FunctionOnce6 6 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4, 5 arg5 T5),
    FunctionOnce7 7 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4, 5 arg5 T5, 6 arg6 T6)
}
