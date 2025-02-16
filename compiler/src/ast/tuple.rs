use super::*;

#[derive(Debug)]
pub enum TupleNode<Rt: RuntimeType> {
    SqueezeScalar(Transcript<Rt>),
    Common(CommonNode<Rt>),
}

pub(super) type TupleUntyped<Rt: RuntimeType> = Outer<TupleNode<Rt>>;

macro_rules! define_tuples {
    ($($n:tt => ($($m:ident $i:tt $T:ident),+)),*) => {
        $(
            pub type $n<$($T),+, Rt: RuntimeType> = Phantomed<TupleUntyped<Rt>, ($($T),+, Rt)>;

            impl<$($T: TypeEraseable<Rt>),+, Rt: RuntimeType> TypeEraseable<Rt> for $n<$($T),+, Rt> {
                fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
                    match &self.t.inner.t {
                        TupleNode::SqueezeScalar(t) => t.erase(cg),
                        TupleNode::Common(cn) => cn.erase(cg),
                    }
                }
            }

            impl<$($T: CommonConstructors<Rt> + Clone + 'static),+, Rt: RuntimeType> $n<$($T),+, Rt> {
                $(
                    #[track_caller]
                    pub fn $m(&self) -> $T {
                        let src = SourceInfo::new(Location::caller().clone(), None);
                        $T::from_tuple_get(self.t.clone(), $i, src)
                    }
                )+
            }
        )*
    };
}

define_tuples! {
    Tuple2 => (get0 0 T0, get1 1 T1),
    Tuple3 => (get0 0 T0, get1 1 T1, get2 2 T2),
    Tuple4 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3),
    Tuple5 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4),
    Tuple6 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5),
    Tuple7 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5, get6 6 T6),
    Tuple8 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5, get6 6 T6, get7 7 T7),
    Tuple9 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5, get6 6 T6, get7 7 T7, get8 8 T8),
    Tuple10 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5, get6 6 T6, get7 7 T7, get8 8 T8, get9 9 T9)
}

impl<Rt: RuntimeType> Tuple2<Transcript<Rt>, Scalar<Rt>, Rt> {
    pub(super) fn from_squeeze_scalar(transcript: Transcript<Rt>, src: SourceInfo) -> Self {
        Self::wrap(TupleUntyped::new(TupleNode::SqueezeScalar(transcript), src))
    }
}
