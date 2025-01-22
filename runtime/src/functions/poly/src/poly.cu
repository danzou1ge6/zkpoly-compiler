#include "poly.h"
#include "poly_basic.cuh"
#include "../../common/error/src/check.cuh"

cudaError_t poly_add(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream) {
    unsigned int block = 1024;
    unsigned int grid = (len - 1) / block + 1;
    detail::NaiveAdd<POLY_FIELD><<<grid, block, 0, stream>>>(a, b, result, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_sub(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream) {
    unsigned int block = 1024;
    unsigned int grid = (len - 1) / block + 1;
    detail::NaiveSub<POLY_FIELD><<<grid, block, 0, stream>>>(a, b, result, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_mul(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream) {
    unsigned int block = 1024;
    unsigned int grid = (len - 1) / block + 1;
    detail::NaiveMul<POLY_FIELD><<<grid, block, 0, stream>>>(a, b, result, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}
