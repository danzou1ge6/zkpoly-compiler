use std::{any, sync::Arc};

use super::*;

#[derive(Debug)]
pub enum WhateverNode<Rt: RuntimeType> {
    Entry,
    Constant(Box<dyn any::Any>),
    Common(CommonNode<Rt>),
}

pub(super) type WhateverUntyped<Rt: RuntimeType> = Outer<WhateverNode<Rt>>;

impl<Rt: RuntimeType> TypeEraseable<Rt> for WhateverUntyped<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        cg.lookup_or_insert_with(self.as_ptr(), |cg| {
            use WhateverNode::*;
            let new_vertex = |node, typ| Vertex::new(node, typ, self.src_lowered());

            match &self.inner.t {
                Entry => todo!("track entry ID"),
                Constant(x) => {
                    let constant = unimplemented!();
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

impl<Rt: RuntimeType, T> TypeEraseable<Rt> for Whatever<Rt, T> where T: 'static {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        self.t.erase(cg)
    }
}

impl<Rt: RuntimeType, T> RuntimeCorrespondance<Rt> for Whatever<Rt, T> where T: 'static + Send + Sync {
    type Rtc = T;
    type RtcBorrowed<'a> = &'a Self::Rtc;

    fn to_variable(x: Self::Rtc) -> Variable<Rt> {
        Variable::Any(Arc::new(x))
    }

    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
        match var {
            Variable::Any(x) => Some(x.downcast_ref().unwrap()),
            _ => None,
        }
    }
}

impl<Rt: RuntimeType, T> Whatever<Rt, T> {
    #[track_caller]
    pub fn entry() -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let untyped = WhateverUntyped::new(WhateverNode::Entry, src);
        Phantomed::wrap(untyped)
    }

    #[track_caller]
    pub fn constant(data: impl any::Any) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let untyped = WhateverUntyped::new(WhateverNode::Constant(Box::new(data)), src);
        Phantomed::wrap(untyped)
    }
}
