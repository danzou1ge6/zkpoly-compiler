use super::*;

#[derive(Debug)]
pub enum TupleNode<Rt: RuntimeType> {
    SqueezeScalar(Transcript<Rt>),
    PlonkPermute(PolyLagrange<Rt>, PolyLagrange<Rt>, usize),
    Common(CommonNode<Rt>),
}

impl<Rt: RuntimeType> TypeEraseable<Rt> for TupleUntyped<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        cg.lookup_or_insert_with(self.as_ptr(), |cg| match &self.inner.t {
            TupleNode::SqueezeScalar(t) => {
                let operand = t.erase(cg);
                Vertex::new(
                    VertexNode::SqueezeScalar(operand),
                    Some(Typ::Tuple(vec![Typ::Transcript, Typ::Scalar])),
                    self.src_lowered(),
                )
            }
            TupleNode::PlonkPermute(input, table, usable_rows) => {
                let input = input.erase(cg);
                let table = table.erase(cg);
                Vertex::new(
                    VertexNode::PolyPermute(input, table, *usable_rows),
                    Some(Typ::Tuple(vec![
                        Typ::lagrange_with_deg(*usable_rows as u64),
                        Typ::lagrange_with_deg(*usable_rows as u64),
                    ])),
                    self.src_lowered(),
                )
            }
            TupleNode::Common(cn) => cn.vertex(cg, self.src_lowered()),
        })
    }
}

pub(super) type TupleUntyped<Rt: RuntimeType> = Outer<TupleNode<Rt>>;

macro_rules! define_tuples {
    ($($n:tt $ni:tt => ($($m:ident $i:tt $T:ident),+)),*) => {
        $(
            pub type $n<$($T),+, Rt: RuntimeType> = Phantomed<TupleUntyped<Rt>, ($($T),+, Rt)>;

            impl<$($T: TypeEraseable<Rt>),+, Rt: RuntimeType> TypeEraseable<Rt> for $n<$($T),+, Rt> {
                fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
                    self.t.erase(cg)
                }
            }

            impl<$($T: RuntimeCorrespondance<Rt>),+, Rt: RuntimeType> RuntimeCorrespondance<Rt> for $n<$($T),+, Rt> {
                type Rtc = ($($T::Rtc),+,);
                type RtcBorrowed<'a> = ($($T::RtcBorrowed<'a>),+,);
                type RtcBorrowedMut<'a> = ($($T::RtcBorrowedMut<'a>),+,);

                fn to_variable(x: Self::Rtc) -> Variable<Rt> {
                    Variable::Tuple(vec![$($T::to_variable(x.$i)),+,])
                }

                fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
                    match var {
                        Variable::Tuple(t) => Some(($($T::try_borrow_variable(&t[$i])?),+,)),
                        _ => {
                            eprintln!("expected Tuple, got {:?}", var);
                            None
                        }
                    }
                }

                fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
                    match var {
                        Variable::Tuple(t) => {
                            assert!(t.len() == $ni);
                            let ptr = t.as_mut_ptr();
                            unsafe {
                                Some(($($T::try_borrow_variable_mut(&mut *ptr.add($i))?),+,))
                            }
                        }
                        _ => {
                            eprintln!("expected Tuple, got {:?}", var);
                            None
                        }
                    }
                }
            }

            impl<$($T: CommonConstructors<Rt> + Clone +'static),+, Rt: RuntimeType> From<(CommonNode<Rt>, SourceInfo)> for $n<$($T),+, Rt> {
                fn from((cn, src): (CommonNode<Rt>, SourceInfo)) -> Self {
                    Phantomed::wrap(TupleUntyped::new(TupleNode::Common(cn), src))
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

                #[track_caller]
                pub fn unpack(&self) -> ($($T),+,) {
                    let src = SourceInfo::new(Location::caller().clone(), None);
                    (
                        ($($T::from_tuple_get(self.t.clone(), $i, src.clone())),+,)
                    )
                }
            }
        )*
    };
}

define_tuples! {
    Tuple2 2 => (get0 0 T0, get1 1 T1),
    Tuple3 3 => (get0 0 T0, get1 1 T1, get2 2 T2),
    Tuple4 4 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3),
    Tuple5 5 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4),
    Tuple6 6 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5),
    Tuple7 7 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5, get6 6 T6),
    Tuple8 8 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5, get6 6 T6, get7 7 T7),
    Tuple9 9 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5, get6 6 T6, get7 7 T7, get8 8 T8),
    Tuple10 10 => (get0 0 T0, get1 1 T1, get2 2 T2, get3 3 T3, get4 4 T4, get5 5 T5, get6 6 T6, get7 7 T7, get8 8 T8, get9 9 T9)
}

impl<Rt: RuntimeType> Tuple2<Transcript<Rt>, Scalar<Rt>, Rt> {
    pub(super) fn from_squeeze_scalar(transcript: Transcript<Rt>, src: SourceInfo) -> Self {
        Self::wrap(TupleUntyped::new(TupleNode::SqueezeScalar(transcript), src))
    }
}

impl<Rt: RuntimeType> Tuple2<PolyLagrange<Rt>, PolyLagrange<Rt>, Rt> {
    #[track_caller]
    pub fn plonk_permute(
        input: &PolyLagrange<Rt>,
        table: &PolyLagrange<Rt>,
        usable_rows: usize,
    ) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Self::wrap(TupleUntyped::new(
            TupleNode::PlonkPermute(input.clone(), table.clone(), usable_rows),
            src,
        ))
    }
}
