#pragma once
#include <cuda_runtime.h>
#include "common.cuh"

// fastest NTT for data on device
extern "C" cudaError_t ssip_ntt(PolyPtr x, const unsigned int *twiddle, unsigned int log_len, cudaStream_t stream, const unsigned int max_threads_stage1_log, const unsigned int max_threads_stage2_log);

// precompute twiddle factors at compile time
extern "C" cudaError_t ssip_precompute(unsigned int *twiddle, unsigned int log_len, const unsigned int *unit);

// recompute NTT for lower memory usage
extern "C" cudaError_t recompute_ntt(PolyPtr x, const unsigned int *pq_d, unsigned int pq_deg, const unsigned int *omegas_d, unsigned int log_len, cudaStream_t stream, const unsigned int max_threads_stage1_log, const unsigned int max_threads_stage2_log);

// precompute pq and omegas at compile time for recompute NTT
extern "C" void gen_pq_omegas(unsigned int *pq, unsigned int *omegas, unsigned int pq_deg, unsigned int len, unsigned int *unit);

extern "C" cudaError_t distribute_powers(PolyPtr poly, const unsigned int *powers, unsigned long long power_num, cudaStream_t stream);