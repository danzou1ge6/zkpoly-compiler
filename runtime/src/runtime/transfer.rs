use zkpoly_cuda_api::stream::CudaStream;

use crate::{
    args::{RuntimeType, Variable, VariableId},
    devices::DeviceType,
    runtime::RuntimeInfo,
};

pub trait Transfer {
    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream);
    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream);
    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream);
    fn cpu2cpu(&self, target: &mut Self);
    fn cpu2disk(&self, target: &mut Self);
    fn disk2cpu(&self, target: &mut Self);
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
                DeviceType::CPU => match src {
                    Variable::Poly(src) => {
                        let dst = dst.unwrap_poly_mut();
                        src.cpu2cpu(dst);
                    }
                    Variable::PointBase(src) => {
                        let dst = dst.unwrap_point_base_mut();
                        src.cpu2cpu(dst);
                    }
                    Variable::Scalar(src) => {
                        let dst = dst.unwrap_scalar_mut();
                        src.cpu2cpu(dst);
                    }
                    Variable::Transcript => unreachable!(),
                    Variable::Point(src) => {
                        let dst = dst.unwrap_point_mut();
                        *dst = src.clone();
                    }
                    Variable::Tuple(src) => {
                        let dst = dst.unwrap_tuple_mut();
                        assert_eq!(src.len(), dst.len());
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            self.transfer(
                                src,
                                dst,
                                src_device.clone(),
                                dst_device.clone(),
                                stream.clone(),
                            );
                        }
                    }
                    Variable::Array(src) => {
                        let dst = dst.unwrap_array_mut();
                        assert_eq!(src.len(), dst.len());
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            self.transfer(
                                src,
                                dst,
                                src_device.clone(),
                                dst_device.clone(),
                                stream.clone(),
                            );
                        }
                    }
                    _ => unreachable!(),
                },
                DeviceType::GPU { .. } => match src {
                    Variable::Poly(src) => {
                        let dst = dst.unwrap_poly_mut();
                        let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                        src.cpu2gpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                    }
                    Variable::PointBase(src) => {
                        let dst = dst.unwrap_point_base_mut();
                        let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                        src.cpu2gpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                    }
                    Variable::Scalar(src) => {
                        let dst = dst.unwrap_scalar_mut();
                        let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                        src.cpu2gpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                    }
                    Variable::Array(src) => {
                        let dst = dst.unwrap_array_mut();
                        assert_eq!(src.len(), dst.len());
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            self.transfer(
                                src,
                                dst,
                                src_device.clone(),
                                dst_device.clone(),
                                stream.clone(),
                            );
                        }
                    }
                    Variable::Tuple(src) => {
                        let dst = dst.unwrap_tuple_mut();
                        assert_eq!(src.len(), dst.len());
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            self.transfer(
                                src,
                                dst,
                                src_device.clone(),
                                dst_device.clone(),
                                stream.clone(),
                            );
                        }
                    }
                    _ => unreachable!(),
                },
                DeviceType::Disk => todo!(),
            },
            DeviceType::GPU { .. } => match dst_device {
                DeviceType::CPU => match src {
                    Variable::Poly(src) => {
                        let dst = dst.unwrap_poly_mut();
                        let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                        src.gpu2cpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                    }
                    Variable::PointBase(src) => {
                        let dst = dst.unwrap_point_base_mut();
                        let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                        src.gpu2cpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                    }
                    Variable::Scalar(src) => {
                        let dst = dst.unwrap_scalar_mut();
                        let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                        src.gpu2cpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                    }
                    Variable::Array(src) => {
                        let dst = dst.unwrap_array_mut();
                        assert_eq!(src.len(), dst.len());
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            self.transfer(
                                src,
                                dst,
                                src_device.clone(),
                                dst_device.clone(),
                                stream.clone(),
                            );
                        }
                    }
                    Variable::Tuple(src) => {
                        let dst = dst.unwrap_tuple_mut();
                        assert_eq!(src.len(), dst.len());
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            self.transfer(
                                src,
                                dst,
                                src_device.clone(),
                                dst_device.clone(),
                                stream.clone(),
                            );
                        }
                    }
                    _ => unreachable!(),
                },
                DeviceType::GPU { .. } => match src {
                    Variable::Poly(src) => {
                        let dst = dst.unwrap_poly_mut();
                        let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                        src.gpu2gpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                    }
                    Variable::PointBase(src) => {
                        let dst = dst.unwrap_point_base_mut();
                        let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                        src.gpu2gpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                    }
                    Variable::Scalar(src) => {
                        let dst = dst.unwrap_scalar_mut();
                        let stream_guard = self.variable[stream.unwrap()].read().unwrap();
                        src.gpu2gpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                    }
                    Variable::Array(src) => {
                        let dst = dst.unwrap_array_mut();
                        assert_eq!(src.len(), dst.len());
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            self.transfer(
                                src,
                                dst,
                                src_device.clone(),
                                dst_device.clone(),
                                stream.clone(),
                            );
                        }
                    }
                    Variable::Tuple(src) => {
                        let dst = dst.unwrap_tuple_mut();
                        assert_eq!(src.len(), dst.len());
                        for (src, dst) in src.iter().zip(dst.iter_mut()) {
                            self.transfer(
                                src,
                                dst,
                                src_device.clone(),
                                dst_device.clone(),
                                stream.clone(),
                            );
                        }
                    }
                    _ => unreachable!(),
                },
                DeviceType::Disk => todo!(),
            },
            DeviceType::Disk => todo!(),
        }
    }
}
