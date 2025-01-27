#pragma once
#include "../../common/mont/src/field_impls.cuh"
#include "../../common/error/src/check.cuh"

namespace detail {
using mont::u32;
using mont::u64;
using mont::usize;

template <typename Field>
__global__ void init_pow_series(u32 *temp_buf, const Field *x, u64 len) {
    u64 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    if (index == 0) {
        Field::one().store(temp_buf);
    } else {
        x->store(temp_buf + index * Field::LIMBS);
    }
}

template <typename Field>
cudaError_t get_pow_series(void *temp_buf, usize *temp_buf_size, u32 *pow_series, const Field *x, u64 len, cudaStream_t stream) {
    auto mul_op = [] __device__ __host__(const Field &a, const Field &b) { return a * b; };

    if (temp_buf == nullptr) {
        usize temp_scan_size = 0;
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr, temp_scan_size, reinterpret_cast<Field*>(pow_series), mul_op, len));
        *temp_buf_size = temp_scan_size;
    } else {
        u32 threads = 256;
        u32 blocks = (len + threads - 1) / threads;

        // 1, x, x, ..., x
        init_pow_series<Field><<<blocks, threads, 0, stream>>>(reinterpret_cast<u32*>(pow_series), x, len);
        CUDA_CHECK(cudaGetLastError());

        void *d_temp_scan = nullptr;
        usize temp_scan_size = 0;

        // calculate x^0, x^1, x^2, ..., x^(n-1)
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(d_temp_scan, temp_scan_size, reinterpret_cast<Field*>(pow_series), mul_op, len, stream));
        d_temp_scan = temp_buf;
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(d_temp_scan, temp_scan_size, reinterpret_cast<Field*>(pow_series), mul_op, len, stream));
    }
    return cudaSuccess;
}

} // namespace detail