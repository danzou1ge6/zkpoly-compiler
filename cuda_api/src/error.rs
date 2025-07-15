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

#[macro_export]
macro_rules! cuda_driver_check {
    ($x:expr) => {{
        let err = $x; // 执行 CUDA 操作并获取错误码
        if err != $crate::bindings::cudaError_enum_CUDA_SUCCESS {
            let mut err_code_ptr: *const i8 = std::ptr::null_mut();
            $crate::bindings::cuGetErrorString(err, &mut err_code_ptr);
            eprintln!(
                "CUDA driver Error [{}:{}]: {}",
                file!(),
                line!(),
                std::ffi::CStr::from_ptr(err_code_ptr).to_str().unwrap()
            );
            panic!("CUDA Error");
        }
    }};
}
