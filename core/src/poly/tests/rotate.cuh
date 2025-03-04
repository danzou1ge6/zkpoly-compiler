#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include "../../common/error/src/check.cuh"

template <typename Field>
cudaError_t rotating(const Field *src, Field *dst, unsigned long long len, long long shift, cudaStream_t stream) {
    assert(src != dst);
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
