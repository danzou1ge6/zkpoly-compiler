use super::{transit::HashTyp, *};

#[derive(Debug)]
pub enum TranscriptNode<Rt: RuntimeType> {
    HashScalar(Transcript<Rt>, Scalar<Rt>, HashTyp),
    HashLagrange(Transcript<Rt>, PolyLagrange<Rt>, HashTyp),
    HashCoef(Transcript<Rt>, PolyCoef<Rt>, HashTyp),
    HashPoint(Transcript<Rt>, Point<Rt>, HashTyp),
    Common(CommonNode<Rt>),
}

pub type Transcript<Rt: RuntimeType> = Outer<TranscriptNode<Rt>>;

impl<Rt: RuntimeType> From<(CommonNode<Rt>, SourceInfo)> for Transcript<Rt> {
    fn from((node, src): (CommonNode<Rt>, SourceInfo)) -> Self {
        Self::new(TranscriptNode::Common(node), src)
    }
}

impl<Rt: RuntimeType> TypeEraseable<Rt> for Transcript<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        cg.lookup_or_insert_with(self.as_ptr(), |cg| {
            fn hash_transcript<'s, Rt: RuntimeType>(
                t: &impl TypeEraseable<Rt>,
                s: &impl TypeEraseable<Rt>,
                typ: &HashTyp,
                cg: &mut Cg<'s, Rt>,
                src: transit::SourceInfo<'s>,
            ) -> Vertex<'s, Rt> {
                let t = t.erase(cg);
                let s = s.erase(cg);
                Vertex::new(
                    VertexNode::HashTranscript {
                        transcript: t,
                        value: s,
                        typ: *typ,
                    },
                    Some(Typ::Transcript),
                    src,
                )
            }

            use TranscriptNode::*;
            match &self.inner.t {
                HashScalar(t, s, hash_typ) => {
                    hash_transcript(t, s, hash_typ, cg, self.src_lowered())
                }
                HashLagrange(t, s, hash_typ) => {
                    hash_transcript(t, s, hash_typ, cg, self.src_lowered())
                }
                HashCoef(t, s, hash_typ) => hash_transcript(t, s, hash_typ, cg, self.src_lowered()),
                HashPoint(t, s, hash_typ) => {
                    hash_transcript(t, s, hash_typ, cg, self.src_lowered())
                }
                Common(node) => node.vertex(cg, self.src_lowered()),
            }
        })
    }
}

impl<Rt: RuntimeType> RuntimeCorrespondance<Rt> for Transcript<Rt> {
    type Rtc = Rt::Trans;
    type RtcBorrowed<'a> = &'a Self::Rtc;
    type RtcBorrowedMut<'a> = &'a mut Self::Rtc;

    fn to_variable(x: Self::Rtc) -> Variable<Rt> {
        Variable::Transcript(zkpoly_runtime::transcript::TranscriptObject::new(x))
    }
    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
        match var {
            Variable::Transcript(x) => Some(x.as_ref()),
            _ => {
                eprintln!("expected transcript, got {:?}", var);
                None
            }
        }
    }
    fn try_borrow_variable_mut(var: &mut Variable<Rt>) -> Option<Self::RtcBorrowedMut<'_>> {
        match var {
            Variable::Transcript(x) => Some(x.as_mut()),
            _ => {
                eprintln!("expected transcript, got {:?}", var);
                None
            }
        }
    }
}

impl<Rt: RuntimeType> Transcript<Rt> {
    #[track_caller]
    pub fn hash_scalar(&mut self, data: &Scalar<Rt>, hash_typ: HashTyp) {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let t = Self::new(
            TranscriptNode::HashScalar(self.clone(), data.clone(), hash_typ),
            src,
        );
        *self = t;
    }

    #[track_caller]
    pub fn hash_lagrange(&mut self, data: &PolyLagrange<Rt>, hash_typ: HashTyp) {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let t = Self::new(
            TranscriptNode::HashLagrange(self.clone(), data.clone(), hash_typ),
            src,
        );
        *self = t;
    }

    #[track_caller]
    pub fn hash_coef(&mut self, data: &PolyCoef<Rt>, hash_typ: HashTyp) {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let t = Self::new(
            TranscriptNode::HashCoef(self.clone(), data.clone(), hash_typ),
            src,
        );
        *self = t;
    }

    #[track_caller]
    pub fn hash_point(&mut self, data: &Point<Rt>, hash_typ: HashTyp) {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let t = Self::new(
            TranscriptNode::HashPoint(self.clone(), data.clone(), hash_typ),
            src,
        );
        *self = t;
    }

    #[track_caller]
    pub fn squeeze_challenge_scalar(&mut self) -> Scalar<Rt> {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let (t, s) = Tuple2::from_squeeze_scalar(self.clone(), src).unpack();
        *self = t;
        s
    }
}
