use super::*;
use std::cell::Cell;
use zkpoly_runtime::error::{Result as RuntimeResult, RuntimeError};

pub type ValueFn<Rt: RuntimeType> =
    Box<dyn Fn(&mut Variable<Rt>, Vec<&Variable<Rt>>) -> RuntimeResult<()> + Send + Sync + 'static>;

pub struct FunctionInCell<Rt: RuntimeType> {
    n_args: usize,
    value: Cell<Option<ValueFn<Rt>>>,
    name: String,
    pub(super) ret_typ: type2::Typ<Rt>,
    deterministic: bool,
}

impl<Rt: RuntimeType> FunctionInCell<Rt> {
    pub fn take(&self) -> Function<Rt> {
        let value = self.value.take().unwrap();
        Function {
            n_args: self.n_args,
            value,
            name: self.name.clone(),
            ret_typ: self.ret_typ.clone(),
            deterministic: self.deterministic,
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
    pub(crate) value: ValueFn<Rt>,
    pub(crate) name: String,
    pub(crate) ret_typ: type2::Typ<Rt>,
    pub(crate) deterministic: bool,
}

impl<Rt: RuntimeType> std::fmt::Debug for Function<Rt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Function({}), n_args: {} ret_typ: {:?})",
            self.name, self.n_args, self.ret_typ
        )
    }
}

pub(super) type FunctionUntyped<Rt: RuntimeType> = Outer<FunctionInCell<Rt>>;

impl<Rt: RuntimeType> FunctionUntyped<Rt> {
    pub fn new_fn(
        name: String,
        f: ValueFn<Rt>,
        n_args: usize,
        ret_typ: type2::Typ<Rt>,
        src: SourceInfo,
        deterministic: bool,
    ) -> Self {
        FunctionUntyped::new(
            FunctionInCell {
                n_args,
                value: Cell::new(Some(f)),
                name,
                ret_typ,
                deterministic,
            },
            src,
        )
    }
}

pub struct FnMarker;
pub struct UndeterministicMarker;

pub type FunctionFn0<Rt: RuntimeType, R> = Phantomed<FunctionUntyped<Rt>, (R, FnMarker)>;
pub type FunctionUd0<Rt: RuntimeType, R> =
    Phantomed<FunctionUntyped<Rt>, (R, UndeterministicMarker)>;

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
        Self::wrap(FunctionUntyped::new_fn(
            name,
            Box::new(f),
            0,
            ret_typ,
            src,
            true,
        ))
    }
}

impl<Rt: RuntimeType, R> FunctionUd0<Rt, R>
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

impl<Rt: RuntimeType, R> FunctionUd0<Rt, R>
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
        Self::wrap(FunctionUntyped::new_fn(
            name,
            Box::new(f),
            0,
            ret_typ,
            src,
            false,
        ))
    }
}

macro_rules! define_function_fn {
    ($($n:tt $m:tt => ($($i:tt $arg:ident $T:ident),*)),*) => {
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
                    Self::wrap(FunctionUntyped::new_fn(name, Box::new(f), $m, ret_typ, src, true))
                }
            }
        )*
    };
}

macro_rules! define_function_ud {
    ($($n:tt $m:tt => ($($i:tt $arg:ident $T:ident),*)),*) => {
        $(
            pub type $n<Rt: RuntimeType, $($T), +, R> = Phantomed<FunctionUntyped<Rt>, ($($T), +, R, UndeterministicMarker)>;

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
                    Self::wrap(FunctionUntyped::new_fn(name, Box::new(f), $m, ret_typ, src, false))
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

define_function_ud! {
    FunctionUd1 1 => (0 arg0 T0),
    FunctionUd2 2 => (0 arg0 T0, 1 arg1 T1),
    FunctionUd3 3 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2),
    FunctionUd4 4 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3),
    FunctionUd5 5 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4),
    FunctionUd6 6 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4, 5 arg5 T5),
    FunctionUd7 7 => (0 arg0 T0, 1 arg1 T1, 2 arg2 T2, 3 arg3 T3, 4 arg4 T4, 5 arg5 T5, 6 arg6 T6)
}
