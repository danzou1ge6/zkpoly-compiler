use crate::poly::Polynomial;
use group::ff::Field;
use std::fmt::Debug;
use std::{any::Any, sync::RwLock};
use zkpoly_common::heap;

zkpoly_common::define_usize_id!(VariableId);
zkpoly_common::define_usize_id!(ConstantId);

pub type VariableTable<T> = heap::Heap<VariableId, RwLock<Option<Variable<T>>>>;
pub type ConstantTable<T> = heap::Heap<ConstantId, Constant<T>>;

pub trait RuntimeType: 'static {
    type Field: Field;
}

#[derive(Debug)]
pub enum Variable<T: RuntimeType> {
    Poly(Polynomial<T::Field>),
    PointBase,
    Scalar(T::Field),
    Transcript,
    Point,
    Tuple(Vec<Variable<T>>),
    Array(Box<[Variable<T>]>),
    Any(Box<dyn Any + Send + Sync>),
}

#[derive(Debug)]
pub struct Constant<T: RuntimeType> {
    name: String,
    value: Variable<T>,
}

impl<T: RuntimeType> Constant<T> {
    pub fn new(name: String, value: Variable<T>) -> Self {
        Self { name, value }
    }
}
