use crate::typ::PolyRepr;

pub struct Polynomial<F> {
    values: *mut F,
    typ: PolyRepr,
    rotate: u64,
}