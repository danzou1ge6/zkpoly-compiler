use std::any;

use super::*;

pub enum WhateverNode<Rt: RuntimeType> {
    Entry,
    Constant(Box<dyn any::Any>),
    Common(CommonNode<Rt>),
}

pub(super) type WhateverUntyped<Rt: RuntimeType> = Outer<WhateverNode<Rt>>;

pub type Whatever<Rt: RuntimeType, T> = Phantomed<WhateverUntyped<Rt>, T>;

impl<Rt: RuntimeType, T> From<(CommonNode<Rt>, SourceInfo)> for Whatever<Rt, T> {
    fn from(value: (CommonNode<Rt>, SourceInfo)) -> Self {
        Phantomed::wrap(Outer::new(WhateverNode::Common(value.0), value.1))
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
