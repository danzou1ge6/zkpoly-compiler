#pragma once
#include "common.cuh"
#include "poly_basic.cuh"

namespace detail {

template <typename Field>
cudaError_t poly_eval(void* temp_buf, usize *temp_buf_size, ConstPolyPtr poly,  u32* res, const Field *x, cudaStream_t stream) {
    u64 len = poly.len;
    usize x_size = len * Field::LIMBS * sizeof(u32);
    auto add_op = [] __device__ __host__(const Field &a, const Field &b) { return a + b; };
    auto poly_iter = make_slice_iter<Field>(poly);
    auto x_iter = SliceIterator<Field>(reinterpret_cast<Field*>(temp_buf), len);

    if (temp_buf == nullptr) {
        usize temp_scan_size = 0;
        CUDA_CHECK(get_pow_series(nullptr, &temp_scan_size, reinterpret_cast<u32*>(temp_buf), x, len, 0));
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

        // calculate x^0, x^1, x^2, ..., x^(n-1)
        CUDA_CHECK(get_pow_series(reinterpret_cast<char*>(temp_buf) + x_size, nullptr, reinterpret_cast<u32*>(temp_buf), x, len, stream));

        // a_i * x^i
        poly_mul<Field><<<blocks, threads, 0, stream>>>(poly_iter, x_iter, x_iter, len);
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