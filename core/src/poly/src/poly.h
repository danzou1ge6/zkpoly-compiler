#pragma once
#include <cuda_runtime.h>
extern "C" cudaError_t poly_add(unsigned int *result, long long r_rotate, const unsigned int *a, long long a_rotate, const unsigned int *b, long long b_rotate, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_sub(unsigned int *result, long long r_rotate, const unsigned int *a, long long a_rotate, const unsigned int *b, long long b_rotate, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_mul(unsigned int *result, long long r_rotate, const unsigned int *a, long long a_rotate, const unsigned int *b, long long b_rotate, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_one_lagrange(unsigned int * target, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_one_coef(unsigned int * target, long long rotate, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_zero(unsigned int *target, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t poly_eval(void* temp_buf, unsigned long *temp_buf_size, const unsigned int *poly,  unsigned int* res, const unsigned int*x, unsigned long long len, long long rotate, cudaStream_t stream);

extern "C" cudaError_t kate_division(void* temp_buf, unsigned long *temp_buf_size, unsigned int log_p, const unsigned int *p, long long p_rotate, const unsigned int *b, unsigned int *q, long long q_rotate, cudaStream_t stream);

extern "C" cudaError_t scan_mul(void * temp_buffer, unsigned long *buffer_size, const unsigned int *poly, long long p_rotate, unsigned int *target, long long t_rotate, const unsigned int *x0, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t batched_invert(void *temp_buffer, unsigned long *buffer_size, unsigned int *poly, unsigned int *inv, unsigned long long len, cudaStream_t stream);

extern "C" cudaError_t inv_scalar(unsigned int* target, cudaStream_t stream);