use crate::{
    args::{RuntimeType, Variable, VariableId},
    devices::DeviceType,
    run::RuntimeInfo,
    transport::Transport,
};

pub fn transfer<T: RuntimeType>(
    src: &Variable<T>,
    dst: &mut Variable<T>,
    src_device: DeviceType,
    dst_device: DeviceType,
    stream: Option<VariableId>,
    info: &RuntimeInfo<T>,
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
                    *dst = src.clone();
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
                        transfer(
                            src,
                            dst,
                            src_device.clone(),
                            dst_device.clone(),
                            stream.clone(),
                            info,
                        );
                    }
                }
                Variable::Array(src) => {
                    let dst = dst.unwrap_array_mut();
                    assert_eq!(src.len(), dst.len());
                    for (src, dst) in src.iter().zip(dst.iter_mut()) {
                        transfer(
                            src,
                            dst,
                            src_device.clone(),
                            dst_device.clone(),
                            stream.clone(),
                            info,
                        );
                    }
                }
                Variable::Stream(_) => unreachable!(),
                Variable::Any(_) => unreachable!(),
            },
            DeviceType::GPU { .. } => match src {
                Variable::Poly(src) => {
                    let dst = dst.unwrap_poly_mut();
                    let stream_guard = info.variable[stream.unwrap()].read().unwrap();
                    src.cpu2gpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                }
                _ => todo!(),
            },
            DeviceType::Disk => todo!(),
        },
        DeviceType::GPU { .. } => match dst_device {
            DeviceType::CPU => match src {
                Variable::Poly(src) => {
                    let dst = dst.unwrap_poly_mut();
                    let stream_guard = info.variable[stream.unwrap()].read().unwrap();
                    src.gpu2cpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                }
                Variable::PointBase(src) => {
                    let dst = dst.unwrap_point_base_mut();
                    let stream_guard = info.variable[stream.unwrap()].read().unwrap();
                    src.gpu2cpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                }
                Variable::Array(src) => {
                    let dst = dst.unwrap_array_mut();
                    assert_eq!(src.len(), dst.len());
                    for (src, dst) in src.iter().zip(dst.iter_mut()) {
                        transfer(
                            src,
                            dst,
                            src_device.clone(),
                            dst_device.clone(),
                            stream.clone(),
                            info,
                        );
                    }
                }
                Variable::Tuple(src) => {
                    let dst = dst.unwrap_tuple_mut();
                    assert_eq!(src.len(), dst.len());
                    for (src, dst) in src.iter().zip(dst.iter_mut()) {
                        transfer(
                            src,
                            dst,
                            src_device.clone(),
                            dst_device.clone(),
                            stream.clone(),
                            info,
                        );
                    }
                }
                _ => unreachable!(),
            },
            DeviceType::GPU { .. } => match src {
                Variable::Poly(src) => {
                    let dst = dst.unwrap_poly_mut();
                    let stream_guard = info.variable[stream.unwrap()].read().unwrap();
                    src.gpu2gpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                }
                Variable::PointBase(src) => {
                    let dst = dst.unwrap_point_base_mut();
                    let stream_guard = info.variable[stream.unwrap()].read().unwrap();
                    src.gpu2gpu(dst, stream_guard.as_ref().unwrap().unwrap_stream());
                }
                Variable::Array(src) => {
                    let dst = dst.unwrap_array_mut();
                    assert_eq!(src.len(), dst.len());
                    for (src, dst) in src.iter().zip(dst.iter_mut()) {
                        transfer(
                            src,
                            dst,
                            src_device.clone(),
                            dst_device.clone(),
                            stream.clone(),
                            info,
                        );
                    }
                }
                Variable::Tuple(src) => {
                    let dst = dst.unwrap_tuple_mut();
                    assert_eq!(src.len(), dst.len());
                    for (src, dst) in src.iter().zip(dst.iter_mut()) {
                        transfer(
                            src,
                            dst,
                            src_device.clone(),
                            dst_device.clone(),
                            stream.clone(),
                            info,
                        );
                    }
                }
                _ => unreachable!(),
            },
            DeviceType::Disk => {
                unreachable!("Currently, we do not support GPU direct transfer to disk")
            }
        },
        DeviceType::Disk => todo!(),
    }
}
