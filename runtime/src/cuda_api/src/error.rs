#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(unused)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[macro_export]
macro_rules! cuda_check {
    ($x:expr) => {{
        let err = $x; // 执行 CUDA 操作并获取错误码
        if err !=  cudaError_cudaSuccess {
            eprintln!(
                "CUDA Error [{}:{}]: {:?}",
                file!(),
                line!(),
                cudaGetErrorString(err)
            );
            panic!("CUDA Error");
        }
    }};
}