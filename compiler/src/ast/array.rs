use zkpoly_common::msm_config::MsmConfig;

use super::*;

#[derive(Debug)]
pub enum ArrayNode<Rt: RuntimeType> {
    MsmCoef(Vec<PolyCoef<Rt>>, PrecomputedPoints<Rt>),
    MsmLagrange(Vec<PolyLagrange<Rt>>, PrecomputedPoints<Rt>),
    Construct(Vec<AstVertex<Rt>>),
    Common(CommonNode<Rt>),
}

pub(super) type ArrayUntyped<Rt: RuntimeType> = Outer<ArrayNode<Rt>>;

impl<Rt: RuntimeType> TypeEraseable<Rt> for ArrayUntyped<Rt> {
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        fn msm<'s, Rt: RuntimeType>(
            polys: Vec<VertexId>,
            points: &PrecomputedPoints<Rt>,
            cg: &mut Cg<'s, Rt>,
            src: transit::SourceInfo<'static>,
        ) -> Vertex<'s, Rt> {
            let points = points.erase(cg);
            let len = polys.len();
            Vertex::new(
                VertexNode::Msm {
                    polys,
                    points: vec![points],
                    alg: MsmConfig::default(),
                },
                Some(Typ::Array(Box::new(Typ::Point), len)),
                src,
            )
        }

        cg.lookup_or_insert_with(self.as_ptr(), |cg| {
            use ArrayNode::*;
            let new_vertex = |node, typ| Vertex::new(node, typ, self.src_lowered());
            match &self.inner.t {
                MsmCoef(polys, points) => {
                    let polys = polys.iter().map(|p| p.erase(cg)).collect();
                    msm(polys, points, cg, self.src_lowered())
                }
                MsmLagrange(polys, points) => {
                    let polys = polys.iter().map(|p| p.erase(cg)).collect();
                    msm(polys, points, cg, self.src_lowered())
                }
                Construct(elems) => {
                    let elems = elems.iter().map(|e| e.erase(cg)).collect();
                    new_vertex(VertexNode::Array(elems), None)
                }
                Common(cn) => cn.vertex(cg, self.src_lowered()),
            }
        })
    }
}

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

impl<Rt: RuntimeType, T> TypeEraseable<Rt> for Array<Rt, T>
where
    T: 'static,
{
    fn erase<'s>(&self, cg: &mut Cg<'s, Rt>) -> VertexId {
        self.t.erase(cg)
    }
}

impl<Rt: RuntimeType, T> RuntimeCorrespondance<Rt> for Array<Rt, T>
where
    T: RuntimeCorrespondance<Rt>,
{
    type Rtc = Vec<T::Rtc>;
    type RtcBorrowed<'a> = Vec<T::RtcBorrowed<'a>>;

    fn to_variable(x: Self::Rtc) -> Variable<Rt> {
        Variable::Tuple(x.into_iter().map(T::to_variable).collect())
    }

    fn try_borrow_variable(var: &Variable<Rt>) -> Option<Self::RtcBorrowed<'_>> {
        match var {
            Variable::Tuple(t) => t
                .iter()
                .map(|v| T::try_borrow_variable(v))
                .collect::<Option<_>>(),
            _ => None,
        }
    }
}
