use std::{any::{self, type_name}, cell::Cell, sync::Arc};

use super::*;

pub enum WhateverNode<Rt: RuntimeType> {
    Constant(Cell<Arc<dyn any::Any + Send + Sync>>),
    Common(CommonNode<Rt>),
}

impl<Rt: RuntimeType> Debug for WhateverNode<Rt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WhateverNode::Constant(_) => f.debug_tuple("Constant").finish(),
            WhateverNode::Common(cn) => f.debug_tuple("Common").field(cn).finish(),
        }
    }
}

pub(super) type WhateverUntyped<Rt: RuntimeType> = Outer<WhateverNode<Rt>>;

impl<Rt: RuntimeType> TypeEraseable<Rt> for WhateverUntyped<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        cg.lookup_or_insert_with(self.as_ptr(), |cg| {
            use WhateverNode::*;
            let new_vertex = |node, typ| Vertex::new(node, typ, self.src_lowered());

            match &self.inner.t {
                Constant(x) => {
                    let val = x.replace(Arc::new(0));
                    let var = Variable::Any(val);
                    let constant =
                        cg.add_constant(var, None);
                    new_vertex(VertexNode::Constant(constant), Some(Typ::Scalar))
                }
                Common(cn) => cn.vertex(cg, self.src_lowered()),
            }
        })
    }
}

pub type Whatever<Rt: RuntimeType, T> = Phantomed<WhateverUntyped<Rt>, T>;

impl<Rt: RuntimeType, T> From<(CommonNode<Rt>, SourceInfo)> for Whatever<Rt, T> {
    fn from(value: (CommonNode<Rt>, SourceInfo)) -> Self {
        Phantomed::wrap(Outer::new(WhateverNode::Common(value.0), value.1))
    }
}

impl<Rt: RuntimeType, T> TypeEraseable<Rt> for Whatever<Rt, T>
where
    T: 'static,
{
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        self.t.erase(cg)
    }
}

impl<Rt: RuntimeType, T> RuntimeCorrespondance<Rt> for Whatever<Rt, T>
where
    T: 'static + Send + Sync,
{
    type Rtc = T;
    type RtcBorrowed<'a> = &'a Self::Rtc;
    type RtcBorrowedMut<'a> = Box<dyn FnOnce(Self::Rtc) + 'a>;

    fn to_variable(x: Self::Rtc) -> Variable<Rt> {
        Variable::Any(Arc::new(x))
    }

    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
        match var {
            Variable::Any(x) => {
                let down_cast = x.downcast_ref();
                if down_cast.is_some() {
                    Some(down_cast.unwrap())
                } else {
                    panic!("expected {:?} with type_id {:?}, got type with type_id {:?}", type_name::<T>(), any::TypeId::of::<T>(), x.type_id())
                }
            }
            _ => None,
        }
    }
    fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
        match var {
            Variable::Any(x) => Some(Box::new(|t| *x = Arc::new(t))),
            _ => {
                eprintln!("expected Any, got {:?}", var);
                None
            }
        }
    }
}

impl<Rt: RuntimeType, T> Whatever<Rt, T> {
    #[track_caller]
    pub fn constant(data: impl any::Any + Send + Sync, name: String) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), Some(name));
        let untyped = WhateverUntyped::new(WhateverNode::Constant(Cell::new(Arc::new(data))), src);
        Phantomed::wrap(untyped)
    }
}
