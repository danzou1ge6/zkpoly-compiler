#pragma once
#include <cuda_runtime.h>
#include "../../common/iter/src/iter.cuh"
extern "C" cudaError_t poly_add(PolyPtr r, ConstPolyPtr a, ConstPolyPtr b, cudaStream_t stream);

extern "C" cudaError_t poly_sub(PolyPtr r, ConstPolyPtr a, ConstPolyPtr b, cudaStream_t stream);

extern "C" cudaError_t poly_mul(PolyPtr r, ConstPolyPtr a, ConstPolyPtr b, cudaStream_t stream);

extern "C" cudaError_t poly_one_lagrange(PolyPtr target, cudaStream_t stream);

extern "C" cudaError_t poly_one_coef(PolyPtr target, cudaStream_t stream);

extern "C" cudaError_t poly_zero(PolyPtr target, cudaStream_t stream);

extern "C" cudaError_t poly_eval(void* temp_buf, unsigned long *temp_buf_size, ConstPolyPtr poly,  unsigned int* res, const unsigned int*x, cudaStream_t stream);

extern "C" cudaError_t kate_division(void* temp_buf, unsigned long *temp_buf_size, unsigned int log_p, ConstPolyPtr p, const unsigned int *b, PolyPtr q, cudaStream_t stream);

extern "C" cudaError_t scan_mul(void * temp_buffer, unsigned long *buffer_size, PolyPtr target, cudaStream_t stream);

extern "C" cudaError_t batched_invert(void *temp_buffer, unsigned long *buffer_size, PolyPtr poly, unsigned int *inv, cudaStream_t stream);

extern "C" cudaError_t inv_scalar(unsigned int* target, cudaStream_t stream);

extern "C" cudaError_t scalar_pow(unsigned int* target, unsigned long long exp, cudaStream_t stream);