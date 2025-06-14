use std::any::type_name;

use zkpoly_common::devices::DeviceType;
use zkpoly_cuda_api::stream::CudaStream;

use crate::{
    args::{RuntimeType, Variable, VariableId},
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
    fn disk2gpu(&self, _: &mut Self) {
        unimplemented!("disk2gpu not implemented for {:?}", type_name::<Self>());
    }
    fn gpu2disk(&self, _: &mut Self) {
        unimplemented!("gpu2disk not implemented for {:?}", type_name::<Self>());
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
                let stream_guard = &(*$self.variable)[$stream.unwrap()];
                src.$method(dst, stream_guard.as_ref().unwrap().unwrap_stream());
            })+
            _ => unreachable!(),
        }
    };
}

impl<T: RuntimeType> RuntimeInfo<T> {
    /// Here we don't need the gpu_mapping, because the transfer is done according to the stream device.
    #[allow(dangerous_implicit_autorefs)]
    pub(super) unsafe fn transfer(
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
                        ScalarArray => unwrap_scalar_array_mut,
                        PointArray => unwrap_point_array_mut,
                        Scalar => unwrap_scalar_mut,
                        Point => unwrap_point_mut,
                        Transcript => unwrap_transcript_mut
                    );
                }
                DeviceType::GPU { .. } => {
                    match_transfer_stream!(
                        self,
                        dst,
                        src,
                        stream,
                        cpu2gpu,
                        ScalarArray => unwrap_scalar_array_mut,
                        PointArray => unwrap_point_array_mut,
                        Scalar => unwrap_scalar_mut
                    );
                }
                DeviceType::Disk => {
                    match src {
                        Variable::ScalarArray(poly) => {
                            let dst = dst.unwrap_scalar_array_mut();
                            poly.cpu2disk(dst);
                        }
                        Variable::PointArray(points) => {
                            let dst = dst.unwrap_point_array_mut();
                            points.cpu2disk(dst);
                        }
                        _ => unimplemented!()
                    }
                }
            },
            DeviceType::GPU { .. } => match dst_device {
                DeviceType::CPU => {
                    match_transfer_stream!(
                        self,
                        dst,
                        src,
                        stream,
                        gpu2cpu,
                        ScalarArray => unwrap_scalar_array_mut,
                        PointArray => unwrap_point_array_mut,
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
                        ScalarArray => unwrap_scalar_array_mut,
                        PointArray => unwrap_point_array_mut,
                        Scalar => unwrap_scalar_mut
                    );
                }
                DeviceType::Disk => {
                    match src {
                        Variable::ScalarArray(poly) => {
                            let dst = dst.unwrap_scalar_array_mut();
                            poly.gpu2disk(dst);
                        }
                        _ => unimplemented!()
                    }
                }
            },
            DeviceType::Disk => {
                match dst_device {
                    DeviceType::CPU => {
                        match src {
                            Variable::ScalarArray(poly) => {
                                let dst = dst.unwrap_scalar_array_mut();
                                poly.disk2cpu(dst);
                            }
                            Variable::PointArray(points) => {
                                let dst = dst.unwrap_point_array_mut();
                                points.disk2cpu(dst);
                            }
                            _ => unimplemented!()
                        }
                    }
                    DeviceType::GPU { .. } => {
                        match src {
                            Variable::ScalarArray(poly) => {
                                let dst = dst.unwrap_scalar_array_mut();
                                poly.disk2gpu(dst);
                            }
                            _ => unimplemented!()
                        }
                    }
                    DeviceType::Disk => unimplemented!(),
                }
            }
        }
    }
}
