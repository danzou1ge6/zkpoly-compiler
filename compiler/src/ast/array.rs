use super::*;

#[derive(Debug)]
pub enum ArrayNode<Rt: RuntimeType> {
    MsmCoef(Vec<PolyCoef<Rt>>, Vec<PrecomputedPoints<Rt>>),
    MsmLagrange(Vec<PolyLagrange<Rt>>, Vec<PrecomputedPoints<Rt>>),
    Construct(Vec<AstVertex<Rt>>),
    Common(CommonNode<Rt>),
}

pub(super) type ArrayUntyped<Rt: RuntimeType> = Outer<ArrayNode<Rt>>;

pub type Array<Rt: RuntimeType, T> = Phantomed<ArrayUntyped<Rt>, T>;

impl<Rt: RuntimeType, T> Array<Rt, T>
where
    T: TypeEraseable<Rt>,
{
    #[track_caller]
    pub fn construct(elements: impl Iterator<Item = T>) -> Self {
        let src = SourceInfo::new(Location::caller().clone(), None);
        let elems: Vec<_> = elements.map(|e| AstVertex::new(e)).collect();
        let untyped = ArrayUntyped::new(ArrayNode::Construct(elems), src);
        Phantomed::wrap(untyped)
    }
}

impl<Rt: RuntimeType, T> Array<Rt, T>
where
    T: CommonConstructors<Rt>,
{
    #[track_caller]
    fn index(&self, index: usize) -> T {
        let src = SourceInfo::new(Location::caller().clone(), None);
        T::from_array_get(self.t.clone(), index, src)
    }
}

impl<Rt: RuntimeType, T> From<(CommonNode<Rt>, SourceInfo)> for Array<Rt, T> {
    fn from(value: (CommonNode<Rt>, SourceInfo)) -> Self {
        Phantomed::wrap(Outer::new(ArrayNode::Common(value.0), value.1))
    }
}
