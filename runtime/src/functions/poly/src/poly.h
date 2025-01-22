#pragma once
#include <cuda_runtime.h>
extern "C" cudaError_t poly_add(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_sub(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_mul(unsigned int *result, const unsigned int *a, const unsigned int *b, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_eval(void* temp_buf, unsigned long *temp_buf_size, const unsigned int *poly,  unsigned int* res, const unsigned int*x, unsigned long long len, cudaStream_t stream);
