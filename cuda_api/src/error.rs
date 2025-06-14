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

#[macro_export]
macro_rules! cufile_check {
    ($x:expr) => {{
        fn cufile_op_error_name(err: CUfileOpError) -> &'static str {
            match err {
                0 => "CU_FILE_SUCCESS",
                5001 => "CU_FILE_DRIVER_NOT_INITIALIZED",
                5002 => "CU_FILE_DRIVER_INVALID_PROPS",
                5003 => "CU_FILE_DRIVER_UNSUPPORTED_LIMIT",
                5004 => "CU_FILE_DRIVER_VERSION_MISMATCH",
                5005 => "CU_FILE_DRIVER_VERSION_READ_ERROR",
                5006 => "CU_FILE_DRIVER_CLOSING",
                5007 => "CU_FILE_PLATFORM_NOT_SUPPORTED",
                5008 => "CU_FILE_IO_NOT_SUPPORTED",
                5009 => "CU_FILE_DEVICE_NOT_SUPPORTED",
                5010 => "CU_FILE_NVFS_DRIVER_ERROR",
                5011 => "CU_FILE_CUDA_DRIVER_ERROR",
                5012 => "CU_FILE_CUDA_POINTER_INVALID",
                5013 => "CU_FILE_CUDA_MEMORY_TYPE_INVALID",
                5014 => "CU_FILE_CUDA_POINTER_RANGE_ERROR",
                5015 => "CU_FILE_CUDA_CONTEXT_MISMATCH",
                5016 => "CU_FILE_INVALID_MAPPING_SIZE",
                5017 => "CU_FILE_INVALID_MAPPING_RANGE",
                5018 => "CU_FILE_INVALID_FILE_TYPE",
                5019 => "CU_FILE_INVALID_FILE_OPEN_FLAG",
                5020 => "CU_FILE_DIO_NOT_SET",
                5022 => "CU_FILE_INVALID_VALUE",
                5023 => "CU_FILE_MEMORY_ALREADY_REGISTERED",
                5024 => "CU_FILE_MEMORY_NOT_REGISTERED",
                5025 => "CU_FILE_PERMISSION_DENIED",
                5026 => "CU_FILE_DRIVER_ALREADY_OPEN",
                5027 => "CU_FILE_HANDLE_NOT_REGISTERED",
                5028 => "CU_FILE_HANDLE_ALREADY_REGISTERED",
                5029 => "CU_FILE_DEVICE_NOT_FOUND",
                5030 => "CU_FILE_INTERNAL_ERROR",
                5031 => "CU_FILE_GETNEWFD_FAILED",
                5033 => "CU_FILE_NVFS_SETUP_ERROR",
                5034 => "CU_FILE_IO_DISABLED",
                5035 => "CU_FILE_BATCH_SUBMIT_FAILED",
                5036 => "CU_FILE_GPU_MEMORY_PINNING_FAILED",
                5037 => "CU_FILE_BATCH_FULL",
                5038 => "CU_FILE_ASYNC_NOT_SUPPORTED",
                5039 => "CU_FILE_IO_MAX_ERROR",
                _ => "UNKNOWN",
            }
        }
        let err: $crate::bindings::CUfileError = $x; // 执行 cufile 操作并获取错误码
        if err.err != $crate::bindings::CUfileOpError_CU_FILE_SUCCESS {
            // cuda driver error
            if err.err == $crate::bindings::CUfileOpError_CU_FILE_CUDA_DRIVER_ERROR {
                let mut err_code_ptr: *const i8 = std::ptr::null_mut();
                $crate::bindings::cuGetErrorString(err.cu_err, &mut err_code_ptr);
                eprintln!(
                    "CUFile CUDA driver Error [{}:{}]: {}",
                    file!(),
                    line!(),
                    std::ffi::CStr::from_ptr(err_code_ptr).to_str().unwrap()
                );
                panic!("CUFile CUDA driver Error");
            }
            eprintln!(
                "CUFile Error [{}:{}]: {}",
                file!(),
                line!(),
                cufile_op_error_name(err.err) // 获取错误, 因为那个获取string的函数好像没bind进来
            );
            panic!("CUFile Error");
        }
    }};
}

#[test]
#[should_panic(expected = "CUFile Error")]
fn test_cufile_check() {
    use crate::bindings::*;
    use std::ptr::null;
    unsafe {
        cufile_check!(cuFileBufRegister(null(), 0, 0));
    }
}
