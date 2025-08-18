use crate as vis;

pub use vis::Id;

#[derive(Clone)]
pub struct Context<'a> {
    prefixes: Vec<&'a str>,
}

impl<'a> Context<'a> {
    pub fn pop(&self) -> Self {
        let mut r = self.clone();
        r.prefixes.pop().expect("empty context");
        r
    }

    pub fn push(&self, s: &'a str) -> Self {
        let mut r = self.clone();
        r.prefixes.push(s);
        r
    }

    pub fn new() -> Self {
        Self {
            prefixes: Vec::new(),
        }
    }
}

impl<'a> std::fmt::Display for Context<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.prefixes.last().unwrap_or(&""))
    }
}

pub trait PrettyGraph<V, I> {
    fn vertices(&self) -> impl Iterator<Item = I>;
    fn vertex(&self, vid: I) -> &V;
    fn indirect_from<'a>(&self, vid: I, ctx: &Context<'a>) -> Id;
}

impl<I, V> PrettyGraph<V, I> for () {
    fn vertices(&self) -> impl Iterator<Item = I> {
        std::iter::empty()
    }
    fn vertex(&self, _vid: I) -> &V {
        panic!("this method is not expected to be called because unit PrettyGraph is used in case when there is no outside graph of a graph")
    }
    fn indirect_from<'a>(&self, _vid: I, _ctx: &Context<'a>) -> Id {
        panic!("this method is not expected to be called because unit PrettyGraph is used in case when there is no outside graph of a graph")
    }
}

pub enum PredecessorId<I, Ie> {
    External(Ie),
    Internal(I),
}

pub trait IdentifyVertex<I> {
    fn identifier<'a>(vid: I, ctx: &Context<'a>) -> vis::Id
    where
        I: std::fmt::Display,
    {
        format!("{}_{}", ctx, vid)
    }
}

pub trait PrettyVertex<I, Ie, A>: IdentifyVertex<I> {
    fn pretty_vertex<'a>(&self, vid: I, ctx: Context<'a>, aux: &A, builder: &mut vis::Builder);
    fn labeled_predecessors(
        &self,
        vid: I,
        aux: &A,
    ) -> impl Iterator<Item = (PredecessorId<I, Ie>, vis::EdgeStyle)>;
    fn pretty_vertex_internal_edges<'a, G>(
        &self,
        vid: I,
        ctx: Context<'a>,
        aux: &A,
        g: &G,
        builder: &mut vis::Builder,
    ) where
        G: PrettyGraph<Self, I>,
        Self: Sized;
}

pub fn pretty_vertices<'a, G, I, Ie, V, A>(
    g: &G,
    aux: &A,
    ctx: Context<'a>,
    builder: &mut vis::Builder,
) where
    V: PrettyVertex<I, Ie, A>,
    I: std::fmt::Display + Copy,
    G: PrettyGraph<V, I>,
{
    for vid in g.vertices() {
        let vertex = g.vertex(vid);
        let id = V::identifier(vid, &ctx);
        vertex.pretty_vertex(vid, ctx.push(&id), aux, builder);
    }
}

pub fn pretty_edges<'a, G, I, Ie, V, A, Ge, Ve>(
    g: &G,
    aux: &A,
    ctx: Context<'a>,
    eg: &Ge,
    builder: &mut vis::Builder,
) where
    V: PrettyVertex<I, Ie, A>,
    I: std::fmt::Display + Copy,
    G: PrettyGraph<V, I>,
    Ge: PrettyGraph<Ve, Ie>,
{
    for vid in g.vertices() {
        let vertex = g.vertex(vid);
        let id = V::identifier(vid, &ctx);
        vertex.pretty_vertex_internal_edges(vid, ctx.push(&id), aux, g, builder);
    }

    for vid in g.vertices() {
        let vertex = g.vertex(vid);
        for (pred, es) in vertex.labeled_predecessors(vid, aux) {
            match pred {
                PredecessorId::Internal(pred_vid) => {
                    let pred_id = g.indirect_from(pred_vid, &ctx);
                    builder.edge(vis::Edge::new(pred_id, V::identifier(vid, &ctx)).with_style(es));
                }
                PredecessorId::External(external_vid) => {
                    let pred_id = eg.indirect_from(external_vid, &ctx.pop());
                    builder.edge(vis::Edge::new(pred_id, V::identifier(vid, &ctx)).with_style(es));
                }
            }
        }
    }
}

pub fn pretty<'a, G, I, V, A>(g: &G, aux: &A, builder: &mut vis::Builder)
where
    V: PrettyVertex<I, (), A>,
    I: std::fmt::Display + Copy,
    G: PrettyGraph<V, I>,
{
    let ctx = Context::new();
    pretty_vertices(g, aux, ctx.clone(), builder);
    pretty_edges::<G, I, (), V, A, (), ()>(g, aux, ctx, &(), builder);
}
