use std::{fs::File, os::fd::AsRawFd};

use crate::{
    bindings::{
        cuFileHandleRegister, CUfileDescr_t, CUfileFileHandleType_CU_FILE_HANDLE_TYPE_OPAQUE_FD,
        CUfileHandle_t, CUfileOpError
    },
    cufile_check,
};

pub fn register_cufile(file: &File) -> (CUfileDescr_t, CUfileHandle_t) {
    let fd = file.as_raw_fd();
    let mut cu_file_descr: CUfileDescr_t = unsafe { std::mem::zeroed() };
    let mut cu_file_handle: CUfileHandle_t = unsafe { std::mem::zeroed() };

    // Linux only
    cu_file_descr.handle.fd = fd;
    cu_file_descr.type_ = CUfileFileHandleType_CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    unsafe {
        cufile_check!(cuFileHandleRegister(
            &mut cu_file_handle as *mut _,
            &mut cu_file_descr as *mut _,
        ));
    }

    (
        cu_file_descr,
        cu_file_handle,
    )
}
