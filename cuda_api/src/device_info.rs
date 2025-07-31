use crate::{
    bindings::{cudaDeviceProp, cudaGetDeviceProperties_v2},
    cuda_check,
};

pub fn get_num_sms(device: i32) -> i32 {
    unsafe {
        let mut device_prop: cudaDeviceProp = std::mem::zeroed();
        cuda_check!(cudaGetDeviceProperties_v2(
            &mut device_prop as *mut cudaDeviceProp,
            device
        ));
        device_prop.multiProcessorCount
    }
}

pub fn sm_version(device: i32) -> String {
    unsafe {
        let mut device_prop: cudaDeviceProp = std::mem::zeroed();
        cuda_check!(cudaGetDeviceProperties_v2(
            &mut device_prop as *mut cudaDeviceProp,
            device
        ));
        format!("{}{}", device_prop.major, device_prop.minor)
    }
}
