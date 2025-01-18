use std::any::type_name;

use zkpoly_cuda_api::stream::CudaStream;

use crate::{
    args::{RuntimeType, Variable, VariableId},
    devices::DeviceType,
    runtime::RuntimeInfo,
};

pub trait Transfer {
    fn cpu2gpu(&self, _: &mut Self, _: &CudaStream) {
        unimplemented!("cpu2gpu not implemented for {:?}", type_name::<Self>());
    }
    fn gpu2cpu(&self, _: &mut Self, _: &CudaStream) {
        unimplemented!("gpu2cpu not implemented for {:?}", type_name::<Self>());
    }
    fn gpu2gpu(&self, _: &mut Self, _: &CudaStream) {
        unimplemented!("gpu2gpu not implemented for {:?}", type_name::<Self>());
    }
    fn cpu2cpu(&self, _: &mut Self) {
        unimplemented!("cpu2cpu not implemented for {:?}", type_name::<Self>());
    }
    fn cpu2disk(&self, _: &mut Self) {
        unimplemented!("cpu2disk not implemented for {:?}", type_name::<Self>());
    }
    fn disk2cpu(&self, _: &mut Self) {
        unimplemented!("disk2cpu not implemented for {:?}", type_name::<Self>());
    }
}

macro_rules! match_transfer {
    ($dst:expr, $src:expr, $method:ident, $($variant:ident => $unwrap_fn:ident),+) => {
        match $src {
            $(Variable::$variant(src) => {
                let dst = $dst.$unwrap_fn();
                src.$method(dst);
            })+
            _ => unreachable!(),
        }
    };
}

macro_rules! match_transfer_stream {
    ($self:expr, $dst:expr, $src:expr, $stream:expr, $method:ident, $($variant:ident => $unwrap_fn:ident),+) => {
        match $src {
            $(Variable::$variant(src) => {
                let dst = $dst.$unwrap_fn();
                let stream_guard = $self.variable[$stream.unwrap()].read().unwrap();
                src.$method(dst, stream_guard.as_ref().unwrap().unwrap_stream());
            })+
            _ => unreachable!(),
        }
    };
}

impl<T: RuntimeType> RuntimeInfo<T> {
    pub(super) fn transfer(
        &self,
        src: &Variable<T>,
        dst: &mut Variable<T>,
        src_device: DeviceType,
        dst_device: DeviceType,
        stream: Option<VariableId>,
    ) {
        match src_device {
            DeviceType::CPU => match dst_device {
                DeviceType::CPU => {
                    match_transfer!(
                        dst,
                        src,
                        cpu2cpu,
                        Poly => unwrap_poly_mut,
                        PointBase => unwrap_point_base_mut,
                        Scalar => unwrap_scalar_mut,
                        Point => unwrap_point_mut
                    );
                }
                DeviceType::GPU { .. } => {
                    match_transfer_stream!(
                        self,
                        dst,
                        src,
                        stream,
                        cpu2gpu,
                        Poly => unwrap_poly_mut,
                        PointBase => unwrap_point_base_mut,
                        Scalar => unwrap_scalar_mut
                    );
                }
                DeviceType::Disk => todo!(),
            },
            DeviceType::GPU { .. } => match dst_device {
                DeviceType::CPU => {
                    match_transfer_stream!(
                        self,
                        dst,
                        src,
                        stream,
                        gpu2cpu,
                        Poly => unwrap_poly_mut,
                        PointBase => unwrap_point_base_mut,
                        Scalar => unwrap_scalar_mut
                    );
                }
                DeviceType::GPU { .. } => {
                    match_transfer_stream!(
                        self,
                        dst,
                        src,
                        stream,
                        gpu2gpu,
                        Poly => unwrap_poly_mut,
                        PointBase => unwrap_point_base_mut,
                        Scalar => unwrap_scalar_mut
                    );
                }
                DeviceType::Disk => todo!(),
            },
            DeviceType::Disk => todo!(),
        }
    }
}
