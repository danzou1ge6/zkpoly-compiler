#pragma once
#include <cuda_runtime.h>

extern "C" cudaError_t msm(
    void * const* buffers, unsigned long* buffer_sizes, unsigned long long len,
    unsigned int batch_per_run, unsigned int parts, unsigned int stage_scalers,
    unsigned int stage_points, unsigned int num_cards, const unsigned int* cards,
    unsigned int const * const* h_points, unsigned int batches, unsigned int const* const* h_scaler_batch, unsigned int*const* h_result
);

extern "C" cudaError_t msm_precompute(unsigned long long len, unsigned int*const* h_points, unsigned int max_cards);