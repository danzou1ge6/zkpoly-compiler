use crate::transit::type2::memory_planning::prelude::*;

pub enum Response<'f, M, T, P, R, Rt: RuntimeType> {
    Complete(R),
    Continue(Continuation<'f, M, T, P, R, Rt>),
}

impl<'s, M, T, P, R, Rt: RuntimeType> Response<'s, M, T, P, R, Rt> {
    pub fn unwrap_complete(self) -> R {
        match self {
            Response::Complete(r) => r,
            _ => panic!("called unwrap_complete on a continuation"),
        }
    }
    pub fn commit<'a, 'i>(
        self,
        allocators: &mut AllocatorCollection<'a, 's, T, P, Rt>,
        machine: &mut M,
        aux: &mut AuxiliaryInfo<'i, Rt>,
    ) -> R {
        allocators.run(self, machine, aux)
    }
}

impl<'s, 'f, M, T, P, R, Rt: RuntimeType> Response<'f, M, T, P, Result<R, Error<'s>>, Rt> {
    pub fn ok(r: R) -> Self {
        Response::Complete(Ok(r))
    }

    pub fn bind_result<R1>(
        self,
        f: impl FnOnce(R) -> Continuation<'f, M, T, P, Result<R1, Error<'s>>, Rt> + 'f,
    ) -> Response<'f, M, T, P, Result<R1, Error<'s>>, Rt>
    where
        R1: 'f,
        Self: 'f,
    {
        match self {
            Response::Complete(Err(e)) => Response::Complete(Err(e)),
            Response::Complete(Ok(r)) => Response::Continue(f(r)),
            Response::Continue(c) => Response::Continue(c.bind_result(f)),
        }
    }
}

pub struct Continuation<'f, M, T, P, R, Rt: RuntimeType> {
    pub(super) f: Box<
        dyn for<'a, 'i> FnOnce(
                &mut AllocatorCollection<'a, 'f, T, P, Rt>,
                &mut M,
                &mut AuxiliaryInfo<'i, Rt>,
            ) -> R
            + 'f,
    >,
    _phantom: PhantomData<fn() -> (M, T, P, R)>,
}

impl<'f, M, T, P, R, Rt: RuntimeType> Continuation<'f, M, T, P, R, Rt> {
    pub fn new(
        f: impl for<'a, 'i> FnOnce(
                &mut AllocatorCollection<'a, 'f, T, P, Rt>,
                &mut M,
                &mut AuxiliaryInfo<'i, Rt>,
            ) -> R
            + 'f,
    ) -> Self {
        Self {
            f: Box::new(f),
            _phantom: PhantomData,
        }
    }

    pub fn collect(
        items: impl Iterator<Item = Self> + 'f,
    ) -> Continuation<'f, M, T, P, Vec<R>, Rt> {
        let f = move |allocators: &mut AllocatorCollection<'_, 'f, T, P, Rt>,
                      machine: &mut M,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let mut res = Vec::new();
            for item in items {
                res.push(allocators.apply_continuation(item, machine, aux));
            }
            res
        };
        Continuation::new(f)
    }

    pub fn return_(r: R) -> Self
    where
        R: 'f,
    {
        let f = move |_allocators: &mut AllocatorCollection<T, P, Rt>,
                      _machine: &mut M,
                      _aux: &mut AuxiliaryInfo<Rt>| { r };
        Continuation::new(f)
    }

    pub fn bind<R1>(
        self,
        f: impl FnOnce(R) -> Continuation<'f, M, T, P, R1, Rt> + 'f,
    ) -> Continuation<'f, M, T, P, R1, Rt>
    where
        Self: 'f,
    {
        let f = move |allocators: &mut AllocatorCollection<'_, 'f, T, P, Rt>,
                      machine: &mut M,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let r = allocators.apply_continuation(self, machine, aux);
            let c = f(r);
            allocators.apply_continuation(c, machine, aux)
        };
        Continuation::new(f)
    }
}

impl<'f, 's, M, T, P, R, Rt: RuntimeType> Continuation<'f, M, T, P, Result<R, Error<'s>>, Rt> {
    pub fn collect_result(
        items: impl Iterator<Item = Self> + 'f,
    ) -> Continuation<'f, M, T, P, Result<Vec<R>, Error<'s>>, Rt> {
        let f = move |allocators: &mut AllocatorCollection<'_, 'f, T, P, Rt>,
                      machine: &mut M,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let mut res = Vec::new();
            for item in items {
                res.push(match allocators.apply_continuation(item, machine, aux) {
                    Ok(p) => p,
                    Err(e) => return Err(e),
                });
            }
            Ok(res)
        };
        Continuation::new(f)
    }

    pub fn bind_result<R1>(
        self,
        f: impl FnOnce(R) -> Continuation<'f, M, T, P, Result<R1, Error<'s>>, Rt> + 'f,
    ) -> Continuation<'f, M, T, P, Result<R1, Error<'s>>, Rt>
    where
        R1: 'f,
        Self: 'f,
    {
        let f = move |allocators: &mut AllocatorCollection<'_, 'f, T, P, Rt>,
                      machine: &mut M,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let r = match allocators.apply_continuation(self, machine, aux) {
                Ok(p) => p,
                Err(e) => return Err(e),
            };
            let c = f(r);
            allocators.apply_continuation(c, machine, aux)
        };
        Continuation::new(f)
    }
}

impl<'a, 's, T, P, Rt: RuntimeType> AllocatorCollection<'a, 's, T, P, Rt> {
    pub fn apply_continuation<'i, R, M>(
        &mut self,
        c: Continuation<'s, M, T, P, R, Rt>,
        machine: &mut M,
        aux: &mut AuxiliaryInfo<'i, Rt>,
    ) -> R {
        (c.f)(self, machine, aux)
    }

    pub fn run<'i, R, M>(
        &mut self,
        resp: Response<'s, M, T, P, R, Rt>,
        machine: &mut M,
        aux: &mut AuxiliaryInfo<'i, Rt>,
    ) -> R {
        match resp {
            Response::Complete(r) => r,
            Response::Continue(c) => self.apply_continuation(c, machine, aux),
        }
    }
}
