#pragma once
#include "common.cuh"
#include "poly_basic.cuh"

namespace detail {

template <typename Field>
__global__ void init_eval_buf(u32 *temp_buf, const Field *x, u64 len) {
    u64 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    if (index == 0) {
        Field::one().store(temp_buf);
    } else {
        x->store(temp_buf + index * Field::LIMBS);
    }
}

template <typename Field>
cudaError_t poly_eval(void* temp_buf, usize *temp_buf_size, const u32 *poly,  u32* res, const Field *x, u64 len, cudaStream_t stream) {
    usize x_size = len * Field::LIMBS * sizeof(u32);
    auto mul_op = [] __device__ __host__(const Field &a, const Field &b) { return a * b; };
    auto add_op = [] __device__ __host__(const Field &a, const Field &b) { return a + b; };

    if (temp_buf == nullptr) {
        usize temp_scan_size = 0;
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr, temp_scan_size, reinterpret_cast<Field*>(temp_buf), mul_op, len));
        usize temp_storage_bytes_reduce = 0;
        CUDA_CHECK(
            cub::DeviceReduce::Reduce(
                nullptr, 
                temp_storage_bytes_reduce, 
                reinterpret_cast<const Field*>(temp_buf), 
                reinterpret_cast<Field*>(res),
                len,
                add_op,
                Field::zero())
        );
        usize max_temp_buf = std::max(temp_scan_size, temp_storage_bytes_reduce);
        *temp_buf_size = x_size + max_temp_buf;
    } else {
        u32 threads = 256;
        u32 blocks = (len + threads - 1) / threads;

        // 1, x, x, ..., x
        init_eval_buf<Field><<<blocks, threads, 0, stream>>>(reinterpret_cast<u32*>(temp_buf), x, len);
        CUDA_CHECK(cudaGetLastError());

        void *d_temp_scan = nullptr;
        usize temp_scan_size = 0;

        // calculate x^0, x^1, x^2, ..., x^(n-1)
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(d_temp_scan, temp_scan_size, reinterpret_cast<Field*>(temp_buf), mul_op, len, stream));
        d_temp_scan = reinterpret_cast<char*>(temp_buf) + x_size;
        CUDA_CHECK(cub::DeviceScan::InclusiveScan(d_temp_scan, temp_scan_size, reinterpret_cast<Field*>(temp_buf), mul_op, len, stream));

        // a_i * x^i
        poly_mul<Field><<<blocks, threads, 0, stream>>>(poly, reinterpret_cast<u32*>(temp_buf), reinterpret_cast<u32*>(temp_buf), len);
        CUDA_CHECK(cudaGetLastError());

        // ruduce
        void *d_temp_storage_reduce = nullptr;
        usize temp_storage_bytes_reduce = 0;

        CUDA_CHECK(
            cub::DeviceReduce::Reduce(
                d_temp_storage_reduce, 
                temp_storage_bytes_reduce, 
                reinterpret_cast<const Field*>(temp_buf), 
                reinterpret_cast<Field*>(res),
                len,
                add_op,
                Field::zero(),
                stream
            )
        );
        d_temp_storage_reduce = reinterpret_cast<char*>(temp_buf) + x_size;
        CUDA_CHECK(
            cub::DeviceReduce::Reduce(
                d_temp_storage_reduce, 
                temp_storage_bytes_reduce, 
                reinterpret_cast<const Field*>(temp_buf), 
                reinterpret_cast<Field*>(res),
                len, 
                add_op,
                Field::zero(),
                stream
            )
        );
    }
    
    return cudaSuccess;
}
} // namespace detail