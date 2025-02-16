use super::*;

#[derive(Debug)]
pub enum TranscriptNode<Rt: RuntimeType> {
    Entry,
    HashScalar(Transcript<Rt>, Scalar<Rt>),
    HashLagrange(Transcript<Rt>, PolyLagrange<Rt>),
    HashCoef(Transcript<Rt>, PolyCoef<Rt>),
    HashPoint(Transcript<Rt>, Point<Rt>),
    Common(CommonNode<Rt>)
}

pub type Transcript<Rt: RuntimeType> = Outer<TranscriptNode<Rt>>;

impl<Rt: RuntimeType> From<(CommonNode<Rt>, SourceInfo)> for Transcript<Rt> {
    fn from((node, src): (CommonNode<Rt>, SourceInfo)) -> Self {
        Self::new(TranscriptNode::Common(node), src)
    }
}

impl<Rt: RuntimeType> TypeEraseable<Rt> for Transcript<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        unimplemented!()
    }
}

impl<Rt: RuntimeType> Transcript<Rt> {
    #[track_caller]
    pub fn entry(name: String) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), Some(name));
        Self::new(TranscriptNode::Entry, src)
    }

    #[track_caller]
    pub fn hash_scalar(&self, data: &Scalar<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Self::new(TranscriptNode::HashScalar(self.clone(), data.clone()), src)
    }

    #[track_caller]
    pub fn hash_lagrange(&self, data: &PolyLagrange<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Self::new(TranscriptNode::HashLagrange(self.clone(), data.clone()), src)
    }

    #[track_caller]
    pub fn hash_coef(&self, data: &PolyCoef<Rt>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Self::new(TranscriptNode::HashCoef(self.clone(), data.clone()), src)
    }

    #[track_caller]
    pub fn squeeze_challenge_scalar(&self) -> Tuple2<Transcript<Rt>, Scalar<Rt>, Rt> {
        let src = SourceInfo::new(Location::caller().clone(), None);
        Tuple2::from_squeeze_scalar(self.clone(), src)
    }
}

