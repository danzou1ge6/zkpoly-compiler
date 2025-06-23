use core::slice;
use scopeguard::defer;
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Seek, Write};

use crate::{
    point::{Point, PointArray},
    runtime::transfer::Transfer,
    scalar::{Scalar, ScalarArray},
};

use super::{move_constant, Constant, ConstantId, ConstantTable, RuntimeType, Variable};
use zkpoly_common::{devices::DeviceType, typ::Typ};
use zkpoly_memory_pool::{buddy_disk_pool::DiskMemoryPool, CpuMemoryPool};

impl<Rt: RuntimeType> Variable<Rt> {
    pub fn dump_binary(
        &self,
        writer: &mut impl Write,
        allocator: &mut CpuMemoryPool,
    ) -> io::Result<()> {
        match self {
            Variable::ScalarArray(poly) => {
                let on_disk = poly.device.is_disk();
                let poly = if on_disk {
                    let mut cpu_poly = ScalarArray::alloc_cpu(poly.len, allocator);
                    poly.disk2cpu(&mut cpu_poly);
                    cpu_poly
                } else {
                    poly.clone()
                };
                defer!(if on_disk {
                    allocator.free(poly.values);
                });

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
                let on_disk = ps.device.is_disk();
                let ps = if on_disk {
                    let mut cpu_ps = PointArray::alloc_cpu(ps.len, allocator);
                    ps.disk2cpu(&mut cpu_ps);
                    cpu_ps
                } else {
                    ps.clone()
                };

                defer!(if on_disk {
                    allocator.free(ps.values);
                });

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

    pub fn dump_size(&self) -> Option<usize> {
        match self {
            Variable::ScalarArray(poly) => Some(poly.len * std::mem::size_of::<Rt::Field>()),
            Variable::Scalar(_) => Some(std::mem::size_of::<Rt::Field>()),
            Variable::Point(_) => Some(std::mem::size_of::<Rt::PointAffine>()),
            Variable::PointArray(ps) => Some(ps.len * std::mem::size_of::<Rt::PointAffine>()),
            _ => None,
        }
    }

    pub fn load_binary(
        typ: &Typ,
        reader: &mut impl Read,
        allocator: &mut CpuMemoryPool,
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

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Entry {
    /// (offset, size) if the data
    position: Option<(usize, usize)>,
    name: Option<String>,
    typ: Typ,
    device: DeviceType,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct Header {
    entries: Vec<Entry>,
    data_size: usize,
}

impl Header {
    pub fn build<Rt: RuntimeType>(ct: &ConstantTable<Rt>) -> Self {
        let mut data_offset = 0;

        let entries: Vec<_> = ct
            .iter()
            .map(|v| {
                let r = if let Some(size) = v.value.dump_size() {
                    Entry {
                        position: Some((data_offset, size)),
                        name: v.name.clone(),
                        typ: v.typ.clone(),
                        device: v.device.clone(),
                    }
                } else {
                    Entry {
                        position: None,
                        name: v.name.clone(),
                        typ: v.typ.clone(),
                        device: v.device.clone(),
                    }
                };

                data_offset += v.value.dump_size().unwrap_or(0);
                r
            })
            .collect();

        Self {
            entries,
            data_size: data_offset,
        }
    }

    pub fn dump_entries_data<Rt: RuntimeType>(
        &self,
        ct: &ConstantTable<Rt>,
        writer: &mut impl Write,
        allocator: &mut CpuMemoryPool,
    ) -> io::Result<()> {
        for (entry, c) in self.entries.iter().zip(ct.iter()) {
            if let Some(_) = entry.position {
                c.value.dump_binary(writer, allocator)?;
            }
        }
        Ok(())
    }

    pub fn load_constant_table<Rt: RuntimeType>(
        &self,
        ct: &mut ConstantTable<Rt>,
        reader: &mut (impl Read + Seek),
        allocator: &mut CpuMemoryPool,
        disk_allocator: &mut DiskMemoryPool,
    ) -> io::Result<()> {
        for (i, entry) in self.entries.iter().enumerate() {
            if let Some((offset, _size)) = entry.position {
                reader.seek(io::SeekFrom::Start(offset as u64))?;
                let val = Variable::load_binary(&entry.typ, reader, allocator)?;
                let constant = Constant::on_cpu(val, entry.name.clone(), entry.typ.clone());
                let constant =
                    move_constant(constant, entry.device.clone(), allocator, disk_allocator);

                while ct.len() <= i {
                    // Tuple(vec![]) is placeholder
                    ct.push(Constant::on_cpu(
                        Variable::Tuple(vec![]),
                        entry.name.clone(),
                        entry.typ.clone(),
                    ));
                }

                ct[ConstantId::from(i)] = constant;
            } else {
                if ct.len() <= i {
                    panic!("expect some constant table already exists at index {}", i);
                }
                let constant = &ct[ConstantId::from(i)];
                if constant.name != entry.name {
                    panic!(
                        "constant name mismatch at index {}, expect {:?}, got {:?}",
                        i, &entry.name, &constant.name
                    );
                }
                if constant.typ != entry.typ {
                    panic!(
                        "constant type mismatch at index {}, expect {:?}, got {:?}",
                        i, &entry.typ, &constant.typ
                    );
                }
            };
        }

        Ok(())
    }
}
