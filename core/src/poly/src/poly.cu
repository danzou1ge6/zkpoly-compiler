#include "poly.h"
#include "poly_basic.cuh"
#include "poly_eval.cuh"
#include "kate_division.cuh"
#include "scan_mul.cuh"
#include "batched_invert.cuh"
#include "rotate.cuh"

cudaError_t poly_add(unsigned int *result, long long r_rotate, const unsigned int *a, long long a_rotate, const unsigned int *b, long long b_rotate, unsigned long long len, cudaStream_t stream) {
    unsigned int block = 256;
    unsigned int grid = (len - 1) / block + 1;
    auto a_iter = mont::make_rotating_iter(reinterpret_cast<const POLY_FIELD*>(a), a_rotate, len);
    auto b_iter = mont::make_rotating_iter(reinterpret_cast<const POLY_FIELD*>(b), b_rotate, len);
    auto r_iter = mont::make_rotating_iter(reinterpret_cast<POLY_FIELD*>(result), r_rotate, len);
    detail::poly_add<POLY_FIELD><<<grid, block, 0, stream>>>(a_iter, b_iter, r_iter, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_sub(unsigned int *result, long long r_rotate, const unsigned int *a, long long a_rotate, const unsigned int *b, long long b_rotate, unsigned long long len, cudaStream_t stream) {
    unsigned int block = 256;
    unsigned int grid = (len - 1) / block + 1;
    auto a_iter = mont::make_rotating_iter(reinterpret_cast<const POLY_FIELD*>(a), a_rotate, len);
    auto b_iter = mont::make_rotating_iter(reinterpret_cast<const POLY_FIELD*>(b), b_rotate, len);
    auto r_iter = mont::make_rotating_iter(reinterpret_cast<POLY_FIELD*>(result), r_rotate, len);
    detail::poly_sub<POLY_FIELD><<<grid, block, 0, stream>>>(a_iter, b_iter, r_iter, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_mul(unsigned int *result, long long r_rotate, const unsigned int *a, long long a_rotate, const unsigned int *b, long long b_rotate, unsigned long long len, cudaStream_t stream) {
    unsigned int block = 256;
    unsigned int grid = (len - 1) / block + 1;
    auto a_iter = mont::make_rotating_iter(reinterpret_cast<const POLY_FIELD*>(a), a_rotate, len);
    auto b_iter = mont::make_rotating_iter(reinterpret_cast<const POLY_FIELD*>(b), b_rotate, len);
    auto r_iter = mont::make_rotating_iter(reinterpret_cast<POLY_FIELD*>(result), r_rotate, len);
    detail::poly_mul<POLY_FIELD><<<grid, block, 0, stream>>>(a_iter, b_iter, r_iter, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_one_lagrange(unsigned int * target, unsigned long long len, cudaStream_t stream) {
    return detail::poly_one_lagrange<POLY_FIELD>(target, len, stream);
}

cudaError_t poly_one_coef(unsigned int * target, long long rotate, unsigned long long len, cudaStream_t stream) {
    return detail::poly_one_coef<POLY_FIELD>(target, rotate, len, stream);
}

cudaError_t poly_zero(unsigned int * target, unsigned long long len, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(target, 0, len * sizeof(POLY_FIELD), stream));
    return cudaSuccess;
}

cudaError_t poly_eval(void* temp_buf, unsigned long *temp_buf_size, const unsigned int *poly,  unsigned int* res, const unsigned int*x, unsigned long long len, long long rotate, cudaStream_t stream) {
    return detail::poly_eval<POLY_FIELD>(temp_buf, temp_buf_size, reinterpret_cast<const POLY_FIELD*>(poly), res, reinterpret_cast<const POLY_FIELD*>(x), len, rotate, stream);
}

cudaError_t kate_division(void* temp_buf, unsigned long *temp_buf_size, unsigned int log_p, const unsigned int *p, long long p_rotate, const unsigned int *b, unsigned int *q, long long q_rotate, cudaStream_t stream) {
    return detail::kate_division<POLY_FIELD>(temp_buf, temp_buf_size, log_p, reinterpret_cast<const POLY_FIELD*>(p), p_rotate, reinterpret_cast<const POLY_FIELD*>(b), reinterpret_cast<POLY_FIELD*>(q), q_rotate, stream);
}

cudaError_t scan_mul(void * temp_buffer, unsigned long *buffer_size, const unsigned int *poly, long long p_rotate, unsigned int *target, long long t_rotate, const unsigned int *x0, unsigned long long len, cudaStream_t stream) {
    return detail::scan_mul<POLY_FIELD>(temp_buffer, buffer_size, poly, p_rotate, target, t_rotate, x0, len, stream);
}

cudaError_t batched_invert(void *temp_buffer, unsigned long *buffer_size, unsigned int *poly, unsigned int *inv, unsigned long long len, cudaStream_t stream) {
    return detail::batched_invert<POLY_FIELD>(temp_buffer, buffer_size, poly, inv, len, stream);
}
