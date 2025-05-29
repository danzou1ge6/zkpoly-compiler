use crate::transit::type2::memory_planning::prelude::*;

pub enum Response<'f, M, T, P, R, Rt: RuntimeType> {
    Complete(R),
    Continue(Continuation<'f, M, T, P, R, Rt>),
}

impl<'f, M, T, P, R, Rt: RuntimeType> Response<'f, M, T, P, R, Rt> {
    pub fn unwrap_complete(self) -> R {
        match self {
            Response::Complete(r) => r,
            _ => panic!("called unwrap_complete on a continuation"),
        }
    }

    pub fn commit<'a, 's, 'i>(
        self,
        allocators: &mut AllocatorCollection<'a, T, P, Rt>,
        machine: &mut M,
        aux: &mut AuxiliaryInfo<'i, Rt>,
    ) -> R {
        allocators.run(self, machine, aux)
    }
}

impl<'s, 'f, M, T, P, R, Rt: RuntimeType> Response<'f, M, T, P, Result<R, Error<'s>>, Rt> {
    pub fn bind_result<R1>(
        self,
        f: impl Fn(R) -> Continuation<'f, M, T, P, Result<R1, Error<'s>>, Rt> + 'f,
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
                &mut AllocatorCollection<'a, T, P, Rt>,
                &mut M,
                &mut AuxiliaryInfo<'i, Rt>,
            ) -> Response<'f, M, T, P, R, Rt>
            + 'f,
    >,
    _phantom: PhantomData<fn() -> (M, T, P, R)>,
}

impl<'f, M, T, P, R, Rt: RuntimeType> Continuation<'f, M, T, P, R, Rt> {
    pub fn new(
        f: impl for<'a, 'i> FnOnce(
                &mut AllocatorCollection<'a, T, P, Rt>,
                &mut M,
                &mut AuxiliaryInfo<'i, Rt>,
            ) -> Response<'f, M, T, P, R, Rt>
            + 'f,
    ) -> Self {
        Self {
            f: Box::new(f),
            _phantom: PhantomData,
        }
    }

    pub fn collect(items: impl Iterator<Item = Self> + 'f) -> Continuation<'f, M, T, P, Vec<R>, Rt> {
        let f = move |allocators: &mut AllocatorCollection<T, P, Rt>,
                      machine: &mut M,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let mut res = Vec::new();
            for item in items {
                res.push(allocators.apply_continuation(item, machine, aux));
            }
            Response::Complete(res)
        };
        Continuation::new(f)
    }

    pub fn return_(r: R) -> Self
    where
        R: 'f,
    {
        let f = move |_allocators: &mut AllocatorCollection<T, P, Rt>,
                      _machine: &mut M,
                      _aux: &mut AuxiliaryInfo<Rt>| { Response::Complete(r) };
        Continuation::new(f)
    }

    pub fn bind<R1>(
        self,
        f: impl Fn(R) -> Continuation<'f, M, T, P, R1, Rt> + 'f,
    ) -> Continuation<'f, M, T, P, R1, Rt>
    where
        Self: 'f,
    {
        let f = move |allocators: &mut AllocatorCollection<T, P, Rt>,
                      machine: &mut M,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let r = allocators.apply_continuation(self, machine, aux);
            Response::Continue(f(r))
        };
        Continuation::new(f)
    }
}

impl<'f, 's, M, T, P, R, Rt: RuntimeType> Continuation<'f, M, T, P, Result<R, Error<'s>>, Rt> {
    pub fn collect_result(
        items: impl Iterator<Item = Self> + 'f,
    ) -> Continuation<'f, M, T, P, Result<Vec<R>, Error<'s>>, Rt> {
        let f = move |allocators: &mut AllocatorCollection<T, P, Rt>,
                      machine: &mut M,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let mut res = Vec::new();
            for item in items {
                res.push(match allocators.apply_continuation(item, machine, aux) {
                    Ok(p) => p,
                    Err(e) => return Response::Complete(Err(e)),
                });
            }
            Response::Complete(Ok(res))
        };
        Continuation::new(f)
    }

    pub fn bind_result<R1>(
        self,
        f: impl Fn(R) -> Continuation<'f, M, T, P, Result<R1, Error<'s>>, Rt> + 'f,
    ) -> Continuation<'f, M, T, P, Result<R1, Error<'s>>, Rt>
    where
        R1: 'f,
        Self: 'f,
    {
        let f = move |allocators: &mut AllocatorCollection<T, P, Rt>,
                      machine: &mut M,
                      aux: &mut AuxiliaryInfo<Rt>| {
            let r = match allocators.apply_continuation(self, machine, aux) {
                Ok(p) => p,
                Err(e) => return Response::Complete(Err(e)),
            };
            Response::Continue(f(r))
        };
        Continuation::new(f)
    }
}

impl<'a, T, P, Rt: RuntimeType> AllocatorCollection<'a, T, P, Rt> {
    pub fn apply_continuation<'s, 'i, R, M>(
        &mut self,
        c: Continuation<M, T, P, R, Rt>,
        machine: &mut M,
        aux: &mut AuxiliaryInfo<'i, Rt>,
    ) -> R {
        let resp = (c.f)(self, machine, aux);
        self.run(resp, machine, aux)
    }

    pub fn run<'s, 'i, R, M>(
        &mut self,
        resp: Response<M, T, P, R, Rt>,
        machine: &mut M,
        aux: &mut AuxiliaryInfo<'i, Rt>,
    ) -> R {
        match resp {
            Response::Complete(r) => r,
            Response::Continue(c) => self.apply_continuation(c, machine, aux),
        }
    }
}
