#include "poly.h"
#include "poly_basic.cuh"

cudaError_t poly_add(PolyPtr r, ConstPolyPtr a, ConstPolyPtr b, cudaStream_t stream) {
    auto len = std::min(a.len, b.len);
    auto upper_len = std::max(a.len, b.len);
    if (a.len < b.len) {
        std::swap(a, b);
    }
    unsigned int block = 256;
    unsigned int grid = (upper_len - 1) / block + 1;
    auto a_iter = iter::make_slice_iter<POLY_FIELD>(a);
    auto b_iter = iter::make_slice_iter<POLY_FIELD>(b);
    auto r_iter = iter::make_slice_iter<POLY_FIELD>(r);
    detail::poly_add<POLY_FIELD><<<grid, block, 0, stream>>>(a_iter, b_iter, r_iter, len, upper_len);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_sub(PolyPtr r, ConstPolyPtr a, ConstPolyPtr b, cudaStream_t stream) {
    auto len = std::min(a.len, b.len);
    auto upper_len = std::max(a.len, b.len);
    auto neg = false;
    if (a.len < b.len) {
        std::swap(a, b);
        neg = true;
    }
    unsigned int block = 256;
    unsigned int grid = (upper_len - 1) / block + 1;
    auto a_iter = iter::make_slice_iter<POLY_FIELD>(a);
    auto b_iter = iter::make_slice_iter<POLY_FIELD>(b);
    auto r_iter = iter::make_slice_iter<POLY_FIELD>(r);
    detail::poly_sub<POLY_FIELD><<<grid, block, 0, stream>>>(a_iter, b_iter, r_iter, len, upper_len, neg);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

cudaError_t poly_mul(PolyPtr r, ConstPolyPtr a, ConstPolyPtr b, cudaStream_t stream) {
    auto len = std::min(a.len, b.len);
    auto upper_len = std::max(a.len, b.len);
    if (a.len < b.len) {
        std::swap(a, b);
    }
    unsigned int block = 256;
    unsigned int grid = (upper_len - 1) / block + 1;
    auto a_iter = iter::make_slice_iter<POLY_FIELD>(a);
    auto b_iter = iter::make_slice_iter<POLY_FIELD>(b);
    auto r_iter = iter::make_slice_iter<POLY_FIELD>(r);
    detail::poly_mul<POLY_FIELD><<<grid, block, 0, stream>>>(a_iter, b_iter, r_iter, len, upper_len);
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

cudaError_t inv_scalar(unsigned int* target, cudaStream_t stream) {
    detail::inverse_scalar<POLY_FIELD><<< 1, 1, 0, stream >>>(reinterpret_cast<POLY_FIELD*>(target));
    return cudaGetLastError();
}

cudaError_t scalar_pow(unsigned int* target, unsigned long long exp, cudaStream_t stream) {
    detail::scalar_pow<POLY_FIELD><<< 1, 1, 0, stream >>>(reinterpret_cast<POLY_FIELD*>(target), exp);
    return cudaGetLastError();
}