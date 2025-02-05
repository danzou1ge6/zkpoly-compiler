#pragma once
#include "common.cuh"

namespace detail {

template <typename Field>
cudaError_t rotate(const Field *src, Field *dst, u64 len, i64 shift, cudaStream_t stream) {
    if (shift == 0) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, sizeof(Field) * len, cudaMemcpyDeviceToDevice, stream));
        return cudaSuccess;
    } else if (shift > 0) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src + shift, sizeof(Field) * (len - shift), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dst + len - shift, src, sizeof(Field) * shift, cudaMemcpyDeviceToDevice, stream));
    } else {
        shift = -shift;
        CUDA_CHECK(cudaMemcpyAsync(dst, src + len - shift, sizeof(Field) * shift, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dst + shift, src, sizeof(Field) * (len - shift), cudaMemcpyDeviceToDevice, stream));
    }

    return cudaSuccess;
}

} // namespace detail