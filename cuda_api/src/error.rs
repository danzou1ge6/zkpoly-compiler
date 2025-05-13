#[macro_export]
macro_rules! cuda_check {
    ($x:expr) => {{
        let err = $x; // 执行 CUDA 操作并获取错误码
        if err != $crate::bindings::cudaError_cudaSuccess {
            eprintln!(
                "CUDA Error [{}:{}]: {}",
                file!(),
                line!(),
                std::ffi::CStr::from_ptr($crate::bindings::cudaGetErrorString(err))
                    .to_str()
                    .unwrap()
            );
            panic!("CUDA Error");
        }
    }};
}
