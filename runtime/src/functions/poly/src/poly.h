#pragma once
#include <cuda_runtime.h>
extern "C" cudaError_t poly_add(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_sub(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_mul(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_one(unsigned int * target, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_zero(unsigned int *target, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_eval(void* temp_buf, unsigned long *temp_buf_size, const unsigned int *poly,  unsigned int* res, const unsigned int*x, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t kate_division(void* temp_buf, unsigned long *temp_buf_size, unsigned int log_p, const unsigned int *p, const unsigned int *b, unsigned int *q, cudaStream_t stream);

extern "C" cudaError_t scan_mul(void * temp_buffer, unsigned long *buffer_size, const unsigned int *poly, unsigned int *target, const unsigned int *x0, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t batched_invert(void *temp_buffer, unsigned long *buffer_size, unsigned int *poly, unsigned int *inv, unsigned long long len, cudaStream_t stream);