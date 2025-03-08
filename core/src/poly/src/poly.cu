#include "poly.h"
#include "poly_basic.cuh"
#include "poly_eval.cuh"
#include "kate_division.cuh"
#include "scan_mul.cuh"
#include "batched_invert.cuh"

cudaError_t poly_add(PolyPtr r, ConstPolyPtr a, ConstPolyPtr b, cudaStream_t stream) {
    auto len = std::min(a.len, b.len);
    unsigned int block = 256;
    unsigned int grid = (len - 1) / block + 1;
    auto a_iter = iter::make_slice_iter<POLY_FIELD>(a);
    auto b_iter = iter::make_slice_iter<POLY_FIELD>(b);
    auto r_iter = iter::make_slice_iter<POLY_FIELD>(r);
    detail::poly_add<POLY_FIELD><<<grid, block, 0, stream>>>(a_iter, b_iter, r_iter, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_sub(PolyPtr r, ConstPolyPtr a, ConstPolyPtr b, cudaStream_t stream) {
    auto len = std::min(a.len, b.len);
    unsigned int block = 256;
    unsigned int grid = (len - 1) / block + 1;
    auto a_iter = iter::make_slice_iter<POLY_FIELD>(a);
    auto b_iter = iter::make_slice_iter<POLY_FIELD>(b);
    auto r_iter = iter::make_slice_iter<POLY_FIELD>(r);
    detail::poly_sub<POLY_FIELD><<<grid, block, 0, stream>>>(a_iter, b_iter, r_iter, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_mul(PolyPtr r, ConstPolyPtr a, ConstPolyPtr b, cudaStream_t stream) {
    auto len = std::min(a.len, b.len);
    unsigned int block = 256;
    unsigned int grid = (len - 1) / block + 1;
    auto a_iter = iter::make_slice_iter<POLY_FIELD>(a);
    auto b_iter = iter::make_slice_iter<POLY_FIELD>(b);
    auto r_iter = iter::make_slice_iter<POLY_FIELD>(r);
    detail::poly_mul<POLY_FIELD><<<grid, block, 0, stream>>>(a_iter, b_iter, r_iter, len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_one_lagrange(PolyPtr target, cudaStream_t stream) {
    return detail::poly_one_lagrange<POLY_FIELD>(target, stream);
}

cudaError_t poly_one_coef(PolyPtr target, cudaStream_t stream) {
    return detail::poly_one_coef<POLY_FIELD>(target, stream);
}

cudaError_t poly_zero(PolyPtr target, cudaStream_t stream) {
    return detail::poly_zero<POLY_FIELD>(target, stream);
}

cudaError_t poly_eval(void* temp_buf, unsigned long *temp_buf_size, ConstPolyPtr poly,  unsigned int* res, const unsigned int*x, cudaStream_t stream) {
    return detail::poly_eval<POLY_FIELD>(temp_buf, temp_buf_size, poly, res, reinterpret_cast<const POLY_FIELD*>(x), stream);
}

cudaError_t kate_division(void* temp_buf, unsigned long *temp_buf_size, unsigned int log_p, ConstPolyPtr p, const unsigned int *b, PolyPtr q, cudaStream_t stream) {
    return detail::kate_division<POLY_FIELD>(temp_buf, temp_buf_size, log_p, p, reinterpret_cast<const POLY_FIELD*>(b), q, stream);
}

cudaError_t scan_mul(void * temp_buffer, unsigned long *buffer_size, PolyPtr target, cudaStream_t stream) {
    return detail::scan_mul<POLY_FIELD>(temp_buffer, buffer_size, target, stream);
}

cudaError_t batched_invert(void *temp_buffer, unsigned long *buffer_size, PolyPtr poly, unsigned int *inv, cudaStream_t stream) {
    return detail::batched_invert<POLY_FIELD>(temp_buffer, buffer_size, poly, inv, stream);
}

cudaError_t inv_scalar(unsigned int* target, cudaStream_t stream) {
    detail::inverse_scalar<POLY_FIELD><<< 1, 1, 0, stream >>>(reinterpret_cast<POLY_FIELD*>(target));
    return cudaGetLastError();
}

cudaError_t scalar_pow(unsigned int* target, unsigned long long exp, cudaStream_t stream) {
    detail::scalar_pow<POLY_FIELD><<< 1, 1, 0, stream >>>(reinterpret_cast<POLY_FIELD*>(target), exp);
    return cudaGetLastError();
}