#include "poly.h"
#include "poly_basic.cuh"
#include "poly_eval.cuh"
#include "kate_division.cuh"
#include "scan_mul.cuh"

cudaError_t poly_add(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream) {
    unsigned int block = 256;
    unsigned int grid = (len - 1) / block + 1;
    detail::poly_add<POLY_FIELD><<<grid, block, 0, stream>>>(a, b, result, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_sub(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream) {
    unsigned int block = 256;
    unsigned int grid = (len - 1) / block + 1;
    detail::poly_sub<POLY_FIELD><<<grid, block, 0, stream>>>(a, b, result, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_mul(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream) {
    unsigned int block = 256;
    unsigned int grid = (len - 1) / block + 1;
    detail::poly_mul<POLY_FIELD><<<grid, block, 0, stream>>>(a, b, result, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_one(unsigned int * target, unsigned long long len, cudaStream_t stream) {
    return detail::poly_one<POLY_FIELD>(target, len, stream);
}

cudaError_t poly_zero(unsigned int * target, unsigned long long len, cudaStream_t stream) {
    return detail::poly_zero<POLY_FIELD>(target, len, stream);
}

cudaError_t poly_eval(void* temp_buf, unsigned long *temp_buf_size, const unsigned int *poly,  unsigned int* res, const unsigned int*x, unsigned long long len, cudaStream_t stream) {
    return detail::poly_eval<POLY_FIELD>(temp_buf, temp_buf_size, poly, res, reinterpret_cast<const POLY_FIELD*>(x), len, stream);
}

cudaError_t kate_division(void* temp_buf, unsigned long *temp_buf_size, unsigned int log_p, const unsigned int *p, const unsigned int *b, unsigned int *q, cudaStream_t stream) {
    return detail::kate_division<POLY_FIELD>(temp_buf, temp_buf_size, log_p, reinterpret_cast<const POLY_FIELD*>(p), reinterpret_cast<const POLY_FIELD*>(b), reinterpret_cast<POLY_FIELD*>(q), stream);
}

cudaError_t scan_mul(void * temp_buffer, unsigned long *buffer_size, const unsigned int *poly, unsigned int *target, const unsigned int *x0, unsigned long long len, cudaStream_t stream) {
    return detail::scan_mul<POLY_FIELD>(temp_buffer, buffer_size, poly, target, x0, len, stream);
}
