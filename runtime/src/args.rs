use crate::any::AnyWrapper;
use crate::gpu_buffer::GpuBuffer;
use crate::point::{Point, PointArray};
use crate::runtime::transfer::Transfer;
use crate::scalar::{Scalar, ScalarArray};
use crate::transcript::{EncodedChallenge, TranscriptObject, TranscriptWrite};
use group::ff::PrimeField;
use halo2curves::CurveAffine;
use std::fmt::Debug;
use zkpoly_common::devices::DeviceType;
use zkpoly_common::heap;
use zkpoly_common::typ::Typ;
use zkpoly_cuda_api::stream::CudaStream;
use zkpoly_memory_pool::buddy_disk_pool::DiskMemoryPool;
use zkpoly_memory_pool::CpuMemoryPool;

zkpoly_common::define_usize_id!(VariableId);
zkpoly_common::define_usize_id!(ConstantId);
zkpoly_common::define_usize_id!(EntryId);

pub type VariableTable<T> = heap::Heap<VariableId, Option<Variable<T>>>;
pub type ConstantTable<T> = heap::Heap<ConstantId, Constant<T>>;
pub type EntryTable<T> = heap::Heap<EntryId, Variable<T>>;

pub fn new_variable_table<T: RuntimeType>(len: usize) -> VariableTable<T> {
    heap::Heap::repeat_with(|| (None), len)
}

pub fn move_constant<T: RuntimeType>(
    c: Constant<T>,
    on_device: DeviceType,
    cpu_allocator: &mut CpuMemoryPool,
    disk_allocator: Option<&mut DiskMemoryPool>,
) -> Constant<T> {
    move_constant_(c, on_device, cpu_allocator, disk_allocator, true)
}

pub fn copy_constant<T: RuntimeType>(
    c: Constant<T>,
    on_device: DeviceType,
    cpu_allocator: &mut CpuMemoryPool,
    disk_allocator: Option<&mut DiskMemoryPool>,
) -> Constant<T> {
    move_constant_(c, on_device, cpu_allocator, disk_allocator, false)
}

pub fn move_constant_<T: RuntimeType>(
    c: Constant<T>,
    on_device: DeviceType,
    cpu_allocator: &mut CpuMemoryPool,
    disk_allocator: Option<&mut DiskMemoryPool>,
    free: bool,
) -> Constant<T> {
    match (c.device.clone(), on_device.clone()) {
        (a, b) if a == b => c,
        (DeviceType::CPU, DeviceType::Disk) => {
            let new_var: Variable<T> = match c.value {
                Variable::ScalarArray(poly) => {
                    let mut new_poly = ScalarArray::<T::Field>::alloc_disk(
                        poly.len(),
                        disk_allocator.expect(
                            "expect to have disk pool if you are moving polynomials to disk",
                        ),
                    );
                    poly.cpu2disk(&mut new_poly);
                    if free {
                        cpu_allocator.free(poly.values);
                    }
                    Variable::ScalarArray(new_poly)
                }
                Variable::PointArray(points) => {
                    let mut new_points = PointArray::<T::PointAffine>::alloc_disk(
                        points.len,
                        disk_allocator
                            .expect("expect to have disk pool if you are moving points to disk"),
                    );
                    points.cpu2disk(&mut new_points);
                    if free {
                        cpu_allocator.free(points.values);
                    }
                    Variable::PointArray(new_points)
                }
                _ => unreachable!("small items don't need to be swaped out"),
            };
            Constant {
                device: on_device.clone(),
                name: c.name,
                value: new_var,
                typ: c.typ,
            }
        }
        (DeviceType::Disk, DeviceType::CPU) => {
            let new_var: Variable<T> = match c.value {
                Variable::ScalarArray(mut poly) => {
                    let mut new_poly =
                        ScalarArray::<T::Field>::alloc_cpu(poly.len(), cpu_allocator);
                    poly.disk2cpu(&mut new_poly);
                    if free {
                        poly.free_disk(disk_allocator.expect(
                            "expect to have disk pool if you are moving polynomials out of disk",
                        ));
                    }
                    Variable::ScalarArray(new_poly)
                }
                Variable::PointArray(mut points) => {
                    let mut new_points =
                        PointArray::<T::PointAffine>::alloc_cpu(points.len, cpu_allocator);
                    points.disk2cpu(&mut new_points);
                    if free {
                        points.free_disk(disk_allocator.expect(
                            "expect to have disk pool if you are moving points out of disk",
                        ));
                    }
                    Variable::PointArray(new_points)
                }
                _ => unreachable!("small items don't need to be swaped out"),
            };
            Constant {
                device: on_device.clone(),
                name: c.name,
                value: new_var,
                typ: c.typ,
            }
        }
        (a, b) => panic!("cannot move constant from {:?} to {:?}", a, b),
    }
}

/// Since constructed during AST generation, all constants are on CPU,
/// now we need move them to where they are assumed to be on during memory planning.
pub fn move_constant_table<T: RuntimeType>(
    table: ConstantTable<T>,
    on_device: &heap::Heap<ConstantId, DeviceType>,
    cpu_allocator: &mut CpuMemoryPool,
    mut disk_allocator: Option<&mut DiskMemoryPool>,
) -> ConstantTable<T> {
    let new_table = table.map(&mut |i, c| {
        move_constant(
            c,
            on_device[i].clone(),
            cpu_allocator,
            disk_allocator.as_deref_mut(),
        )
    });
    cpu_allocator.shrink();
    new_table
}

pub fn add_entry<T: RuntimeType>(t: &mut EntryTable<T>, var: Variable<T>) {
    t.push(var);
}

pub trait RuntimeType: 'static + Clone + Send + Sync + Debug {
    type PointAffine: CurveAffine<ScalarExt = Self::Field>;
    type Field: PrimeField
        + Into<<Self::PointAffine as CurveAffine>::ScalarExt>
        + From<<Self::PointAffine as CurveAffine>::ScalarExt>;
    type Challenge: EncodedChallenge<Self::PointAffine>;
    type Trans: TranscriptWrite<Self::PointAffine, Self::Challenge> + std::fmt::Debug;
}

#[derive(Debug, Clone)]
pub enum Variable<T: RuntimeType> {
    ScalarArray(ScalarArray<T::Field>),
    PointArray(PointArray<T::PointAffine>),
    Scalar(Scalar<T::Field>),
    Transcript(TranscriptObject<T>), // cpu only
    Point(Point<T::PointAffine>),    // cpu only
    Tuple(Vec<Variable<T>>),         // cpu only
    Stream(CudaStream),              // cpu only
    Any(AnyWrapper),                 // cpu only
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

    pub fn unwrap_transcript_move(self) -> TranscriptObject<T> {
        match self {
            Variable::Transcript(transcript) => transcript,
            _ => panic!("unwrap_transcript: not a transcript"),
        }
    }

    pub fn unwrap_transcript(&self) -> &TranscriptObject<T> {
        match self {
            Variable::Transcript(transcript) => transcript,
            _ => panic!("unwrap_transcript: not a transcript"),
        }
    }

    pub fn unwrap_transcript_mut(&mut self) -> &mut TranscriptObject<T> {
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

#[derive(Debug, Clone)]
pub struct Constant<T: RuntimeType> {
    pub device: DeviceType,
    pub name: Option<String>,
    pub value: Variable<T>,
    pub typ: Typ,
}

impl<Rt: RuntimeType> Constant<Rt> {
    pub fn on_cpu(value: Variable<Rt>, name: Option<String>, typ: zkpoly_common::typ::Typ) -> Self {
        Self {
            name,
            value,
            typ,
            device: DeviceType::CPU,
        }
    }

    pub fn new(
        value: Variable<Rt>,
        name: Option<String>,
        typ: zkpoly_common::typ::Typ,
        device: DeviceType,
    ) -> Self {
        Self {
            name,
            value,
            typ,
            device,
        }
    }
}

pub mod serialization;
