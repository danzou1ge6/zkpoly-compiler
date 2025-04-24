use core::slice;
use std::io::{self, Read, Write};

use crate::{
    point::{Point, PointArray},
    scalar::{Scalar, ScalarArray},
};

use super::{RuntimeType, Variable};
use zkpoly_common::typ::Typ;
use zkpoly_memory_pool::PinnedMemoryPool;

impl<Rt: RuntimeType> Variable<Rt> {
    pub fn dump_binary(&self, writer: &mut impl Write) -> io::Result<()> {
        match self {
            Variable::ScalarArray(poly) => {
                if !poly.device.is_cpu() {
                    panic!("polynomail must be on CPU");
                }
                if poly.slice_info.is_some() || poly.rotate != 0 {
                    panic!("polynomial must be unrotated and unsliced");
                }
                unsafe {
                    let v = slice::from_raw_parts(
                        poly.values.cast::<u8>(),
                        poly.len * std::mem::size_of::<Rt::Field>(),
                    );
                    writer.write_all(v)?;
                }
            }
            Variable::Scalar(s) => {
                if !s.device.is_cpu() {
                    panic!("scalar must be on CPU");
                }
                unsafe {
                    let v = slice::from_raw_parts(
                        s.value.cast::<u8>(),
                        std::mem::size_of::<Rt::Field>(),
                    );
                    writer.write_all(v)?;
                }
            }
            Variable::Point(p) => unsafe {
                let v = slice::from_raw_parts(
                    p.value.cast::<u8>(),
                    std::mem::size_of::<Rt::PointAffine>(),
                );
                writer.write_all(v)?;
            },
            Variable::PointArray(ps) => {
                if !ps.device.is_cpu() {
                    panic!("point must be on CPU");
                }
                unsafe {
                    let v = slice::from_raw_parts(
                        ps.values.cast::<u8>(),
                        ps.len * std::mem::size_of::<Rt::PointAffine>(),
                    );
                    writer.write_all(v)?;
                }
            }
            _ => panic!("unsupported variable type"),
        }
        Ok(())
    }

    pub fn load_binary(
        &self,
        typ: &Typ,
        reader: &mut impl Read,
        allocator: &mut PinnedMemoryPool,
    ) -> io::Result<Self> {
        match typ {
            Typ::ScalarArray { len, .. } => {
                let p: ScalarArray<Rt::Field> = ScalarArray::alloc_cpu(*len, allocator);
                unsafe {
                    let v = slice::from_raw_parts_mut(
                        p.values.cast::<u8>(),
                        p.len * std::mem::size_of::<Rt::Field>(),
                    );
                    reader.read_exact(v)?;
                }
                Ok(Variable::ScalarArray(p))
            }
            Typ::Scalar => {
                let p: Scalar<Rt::Field> = Scalar::new_cpu();
                unsafe {
                    let v = slice::from_raw_parts_mut(
                        p.value.cast::<u8>(),
                        std::mem::size_of::<Rt::Field>(),
                    );
                    reader.read_exact(v)?;
                }
                Ok(Variable::Scalar(p))
            }
            Typ::Point => {
                let p = unsafe {
                    let p = std::mem::MaybeUninit::<Rt::PointAffine>::uninit();
                    let v = slice::from_raw_parts_mut(
                        p.as_ptr().cast::<u8>().cast_mut(),
                        std::mem::size_of::<Rt::PointAffine>(),
                    );
                    reader.read_exact(v)?;
                    p.assume_init()
                };
                Ok(Variable::Point(Point::new(p)))
            }
            Typ::PointBase { len } => {
                let p: PointArray<Rt::PointAffine> = PointArray::alloc_cpu(*len, allocator);
                unsafe {
                    let v = slice::from_raw_parts_mut(
                        p.values.cast::<u8>(),
                        p.len * std::mem::size_of::<Rt::PointAffine>(),
                    );
                    reader.read_exact(v)?;
                }
                Ok(Variable::PointArray(p))
            }
            _ => panic!("unsupported variable type"),
        }
    }
}
