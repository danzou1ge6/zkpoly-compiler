use crate::gpu_buffer::GpuBuffer;
use crate::point_base::PointBase;
use crate::poly::Polynomial;
use crate::scaler::Scalar;
use group::ff::Field;
use pasta_curves::arithmetic::CurveAffine;
use std::fmt::Debug;
use std::{any::Any, sync::RwLock};
use zkpoly_common::heap;
use zkpoly_cuda_api::stream::CudaStream;

zkpoly_common::define_usize_id!(VariableId);
zkpoly_common::define_usize_id!(ConstantId);

pub type VariableTable<T> = heap::Heap<VariableId, RwLock<Option<Variable<T>>>>;
pub type ConstantTable<T> = heap::Heap<ConstantId, Constant<T>>;

pub trait RuntimeType: 'static + Debug + Clone + Send + Sync {
    type Field: Field;
    type Point: CurveAffine;
}

#[derive(Debug)]
pub enum Variable<T: RuntimeType> {
    Poly(Polynomial<T::Field>),
    PointBase(PointBase<T::Point>),
    Scalar(Scalar<T::Field>),
    Transcript,
    Point(T::Point),
    Tuple(Vec<Variable<T>>),
    Array(Box<[Variable<T>]>),
    Stream(CudaStream),
    Any(Box<dyn Any + Send + Sync>),
    GpuBuffer(GpuBuffer),
}

impl<T: RuntimeType> Variable<T> {
    pub fn unwrap_poly(&self) -> &Polynomial<T::Field> {
        match self {
            Variable::Poly(poly) => poly,
            _ => panic!("unwrap_poly: not a polynomial"),
        }
    }

    pub fn unwrap_poly_mut(&mut self) -> &mut Polynomial<T::Field> {
        match self {
            Variable::Poly(poly) => poly,
            _ => panic!("unwrap_poly_mut: not a polynomial"),
        }
    }

    pub fn unwrap_point_base(&self) -> &PointBase<T::Point> {
        match self {
            Variable::PointBase(point_base) => point_base,
            _ => panic!("unwrap_point_base: not a point base"),
        }
    }

    pub fn unwrap_point_base_mut(&mut self) -> &mut PointBase<T::Point> {
        match self {
            Variable::PointBase(point_base) => point_base,
            _ => panic!("unwrap_point_base_mut: not a point base"),
        }
    }

    pub fn unwrap_scalar(&self) -> &Scalar<T::Field> {
        match self {
            Variable::Scalar(scalar) => scalar,
            _ => panic!("unwrap_scalar: not a scalar"),
        }
    }

    pub fn unwrap_scalar_mut(&mut self) -> &mut Scalar<T::Field> {
        match self {
            Variable::Scalar(scalar) => scalar,
            _ => panic!("unwrap_scalar_mut: not a scalar"),
        }
    }

    pub fn unwrap_transcript(&self) {
        match self {
            Variable::Transcript => (),
            _ => panic!("unwrap_transcript: not a transcript"),
        }
    }

    pub fn unwrap_point(&self) -> &T::Point {
        match self {
            Variable::Point(point) => point,
            _ => panic!("unwrap_point: not a point"),
        }
    }

    pub fn unwrap_point_mut(&mut self) -> &mut T::Point {
        match self {
            Variable::Point(point) => point,
            _ => panic!("unwrap_point_mut: not a point"),
        }
    }

    pub fn unwrap_tuple(&self) -> &[Variable<T>] {
        match self {
            Variable::Tuple(tuple) => tuple,
            _ => panic!("unwrap_tuple: not a tuple"),
        }
    }

    pub fn unwrap_tuple_mut(&mut self) -> &mut [Variable<T>] {
        match self {
            Variable::Tuple(tuple) => tuple,
            _ => panic!("unwrap_tuple_mut: not a tuple"),
        }
    }

    pub fn unwrap_array(&self) -> &[Variable<T>] {
        match self {
            Variable::Array(array) => array,
            _ => panic!("unwrap_array: not an array"),
        }
    }

    pub fn unwrap_array_mut(&mut self) -> &mut [Variable<T>] {
        match self {
            Variable::Array(array) => array,
            _ => panic!("unwrap_array_mut: not an array"),
        }
    }

    pub fn unwrap_stream(&self) -> &CudaStream {
        match self {
            Variable::Stream(stream) => stream,
            _ => panic!("unwrap_stream: not a stream"),
        }
    }

    pub fn unwrap_any(&self) -> &dyn Any {
        match self {
            Variable::Any(any) => any.as_ref(),
            _ => panic!("unwrap_any: not an any"),
        }
    }

    pub fn unwrap_any_mut(&mut self) -> &mut dyn Any {
        match self {
            Variable::Any(any) => any.as_mut(),
            _ => panic!("unwrap_any_mut: not an any"),
        }
    }
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
