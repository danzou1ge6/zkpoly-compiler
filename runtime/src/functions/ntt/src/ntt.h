#pragma once
#include <cuda_runtime.h>

// fastest NTT for data on device
extern "C" cudaError_t ssip_ntt(unsigned int *x, const unsigned int *twiddle, unsigned int log_len, cudaStream_t stream, const unsigned int max_threads_stage1_log, const unsigned int max_threads_stage2_log);

// precompute twiddle factors at compile time
extern "C" cudaError_t ssip_precompute(unsigned int *twiddle, unsigned int log_len, const unsigned int *unit);