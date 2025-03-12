use crate::gpu_buffer::GpuBuffer;
use crate::point::{Point, PointArray};
use crate::scalar::{Scalar, ScalarArray};
use crate::transcript::{EncodedChallenge, TranscriptWrite};
use group::ff::PrimeField;
use pasta_curves::arithmetic::CurveAffine;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::{any::Any, sync::RwLock};
use zkpoly_common::heap;
use zkpoly_cuda_api::stream::CudaStream;

zkpoly_common::define_usize_id!(VariableId);
zkpoly_common::define_usize_id!(ConstantId);
zkpoly_common::define_usize_id!(EntryId);

pub type VariableTable<T> = heap::Heap<VariableId, RwLock<Option<Variable<T>>>>;
pub type ConstantTable<T> = heap::Heap<ConstantId, Mutex<Option<Constant<T>>>>;
pub type EntryTable<T> = heap::Heap<EntryId, Mutex<Option<Variable<T>>>>;

pub fn new_variable_table<T: RuntimeType>(len: usize) -> VariableTable<T> {
    heap::Heap::repeat_with(|| RwLock::new(None), len)
}

pub fn add_entry<T: RuntimeType>(t: &mut EntryTable<T>, var: Variable<T>) {
    t.push(Mutex::new(Some(var)));
}

pub trait RuntimeType: 'static + Clone + Send + Sync + Debug {
    type PointAffine: CurveAffine<ScalarExt = Self::Field>;
    type Field: PrimeField
        + Into<<Self::PointAffine as CurveAffine>::ScalarExt>
        + From<<Self::PointAffine as CurveAffine>::ScalarExt>;
    type Challenge: EncodedChallenge<Self::PointAffine>;
    type Trans: TranscriptWrite<Self::PointAffine, Self::Challenge>;
}

#[derive(Debug, Clone)]
pub enum Variable<T: RuntimeType> {
    ScalarArray(ScalarArray<T::Field>),
    PointArray(PointArray<T::PointAffine>),
    Scalar(Scalar<T::Field>),
    Transcript(T::Trans),            // cpu only
    Point(Point<T::PointAffine>),    // cpu only
    Tuple(Vec<Variable<T>>),         // cpu only
    Stream(CudaStream),              // cpu only
    Any(Arc<dyn Any + Send + Sync>), // cpu only
    GpuBuffer(GpuBuffer),            // gpu only
}

impl<T: RuntimeType> Variable<T> {
    pub fn unwrap_scalar_array(&self) -> &ScalarArray<T::Field> {
        match self {
            Variable::ScalarArray(poly) => poly,
            _ => panic!("unwrap_scalar_array: not a polynomial"),
        }
    }

    pub fn unwrap_scalar_array_mut(&mut self) -> &mut ScalarArray<T::Field> {
        match self {
            Variable::ScalarArray(poly) => poly,
            _ => panic!("unwrap_scalar_array_mut: not a polynomial"),
        }
    }

    pub fn unwrap_point_array(&self) -> &PointArray<T::PointAffine> {
        match self {
            Variable::PointArray(point_array) => point_array,
            _ => panic!("unwrap_point_array: not a point base"),
        }
    }

    pub fn unwrap_point_array_mut(&mut self) -> &mut PointArray<T::PointAffine> {
        match self {
            Variable::PointArray(point_array) => point_array,
            _ => panic!("unwrap_point_array_mut: not a point base"),
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

    pub fn unwrap_transcript(&self) -> &T::Trans {
        match self {
            Variable::Transcript(transcript) => transcript,
            _ => panic!("unwrap_transcript: not a transcript"),
        }
    }

    pub fn unwrap_transcript_mut(&mut self) -> &mut T::Trans {
        match self {
            Variable::Transcript(transcript) => transcript,
            _ => panic!("unwrap_transcript_mut: not a transcript"),
        }
    }

    pub fn unwrap_point(&self) -> &Point<T::PointAffine> {
        match self {
            Variable::Point(point) => point,
            _ => panic!("unwrap_point: not a point"),
        }
    }

    pub fn unwrap_point_mut(&mut self) -> &mut Point<T::PointAffine> {
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

    pub fn unwrap_gpu_buffer(&self) -> &GpuBuffer {
        match self {
            Variable::GpuBuffer(buffer) => buffer,
            _ => panic!("unwrap_gpu_buffer: not a gpu buffer"),
        }
    }

    pub fn unwrap_gpu_buffer_mut(&mut self) -> &mut GpuBuffer {
        match self {
            Variable::GpuBuffer(buffer) => buffer,
            _ => panic!("unwrap_gpu_buffer_mut: not a gpu buffer"),
        }
    }
}

pub struct Constant<T: RuntimeType> {
    pub name: String,
    pub value: Variable<T>,
}

impl<T: RuntimeType> Constant<T> {
    pub fn new(name: String, value: Variable<T>) -> Self {
        Self { name, value }
    }
}
