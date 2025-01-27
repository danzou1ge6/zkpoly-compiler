#pragma once
#include "common.cuh"

namespace detail {

template <typename Field>
cudaError_t scan_mul(void * temp_buffer, usize *buffer_size, const u32 *poly, u32 *target, const u32 *x0, u64 len, cudaStream_t stream) {
    auto mul_op = [] __device__ __host__(const Field &a, const Field &b) { return a * b; };

    if (temp_buffer == nullptr) {
        usize temp_scan_size = 0;
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr, temp_scan_size, reinterpret_cast<Field*>(target), mul_op, len));
        *buffer_size = temp_scan_size;
        return cudaSuccess;
    }
    CUDA_CHECK(cudaMemcpyAsync(target + Field::LIMBS, poly, sizeof(Field) * (len - 1), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(target, x0, sizeof(Field), cudaMemcpyDeviceToDevice, stream));

    void *d_temp_scan = nullptr;
    usize temp_scan_size = 0;

    // calculate x0, x0 * p0, x0 * p0 * p1, ...
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(d_temp_scan, temp_scan_size, reinterpret_cast<Field*>(target), mul_op, len, stream));
    d_temp_scan = temp_buffer;
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(d_temp_scan, temp_scan_size, reinterpret_cast<Field*>(target), mul_op, len, stream));
    return cudaSuccess;
}

} // namespace detail