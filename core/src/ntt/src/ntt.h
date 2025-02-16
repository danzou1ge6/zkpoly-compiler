#pragma once
#include <cuda_runtime.h>

// fastest NTT for data on device
extern "C" cudaError_t ssip_ntt(unsigned int *x, long long x_rotate, const unsigned int *twiddle, unsigned int log_len, cudaStream_t stream, const unsigned int max_threads_stage1_log, const unsigned int max_threads_stage2_log);

// precompute twiddle factors at compile time
extern "C" cudaError_t ssip_precompute(unsigned int *twiddle, unsigned int log_len, const unsigned int *unit);

// recompute NTT for lower memory usage
extern "C" cudaError_t recompute_ntt(unsigned int *x, long long x_rotate, const unsigned int *pq_d, unsigned int pq_deg, const unsigned int *omegas_d, unsigned int log_len, cudaStream_t stream, const unsigned int max_threads_stage1_log, const unsigned int max_threads_stage2_log);

// precompute pq and omegas at compile time for recompute NTT
extern "C" void gen_pq_omegas(unsigned int *pq, unsigned int *omegas, unsigned int pq_deg, unsigned int len, unsigned int *unit);

extern "C" cudaError_t distribute_pow_zeta(unsigned int *poly, long long rotate, const unsigned int *zeta, unsigned long long len, cudaStream_t stream);