#include "msm_impl.cuh"
#include "msm.h"
#include "../../common/curve/src/curve_impls.cuh"
#include <array>
#include <vector>

using Config = detail::MsmConfig<MSM_BITS, MSM_WINDOW_SIZE, MSM_TARGET_WINDOWS, MSM_DEBUG>;

cudaError_t msm(
    void * const* buffers, unsigned long* buffer_sizes, unsigned long long len,
    unsigned int batch_per_run, unsigned int parts, unsigned int stage_scalers,
    unsigned int stage_points, unsigned int num_cards, const unsigned int* cards,
    unsigned int const * const* h_points, unsigned int batches, unsigned int const* const* h_scaler_batch, unsigned int * const* h_result
) {
    std::vector<unsigned int> cards_vec;
    for (int i = 0; i < num_cards; i++) {
        cards_vec.push_back(cards[i]);
    }
    detail::MultiGPUMSM<Config, MSM_CURVE::Field, MSM_CURVE::Point, MSM_CURVE::PointAffine> msm(len, batch_per_run, parts, stage_scalers, stage_points, cards_vec);
    if (buffers == nullptr) {
        return msm.alloc_gpu(nullptr, buffer_sizes);
    }
    std::array<const unsigned int*, Config::n_precompute> h_points_array;
    for (int i = 0; i < Config::n_precompute; i++) {
        h_points_array[i] = h_points[i];
    }
    msm.set_points(h_points_array);
    msm.alloc_gpu(buffers, nullptr);
    std::vector<const unsigned int*> h_scaler_batches;
    h_scaler_batches.resize(batches);
    for (int i = 0; i < batches; i++) {
        h_scaler_batches[i] = h_scaler_batch[i];
    }
    std::vector<MSM_CURVE::Point> h_result_vec;
    h_result_vec.resize(batches);
    CUDA_CHECK(msm.msm(h_scaler_batches, h_result_vec));
    for (int i = 0; i < batches; i++) {
        *reinterpret_cast<MSM_CURVE::PointAffine*>(h_result[i]) = h_result_vec[i].to_affine();
    }
    return cudaSuccess;
}

cudaError_t msm_precompute(unsigned long long len, unsigned int*const* h_points, unsigned int max_cards) {
    std::array<unsigned int*, Config::n_precompute> h_points_array;
    for (int i = 0; i < Config::n_precompute; i++) {
        h_points_array[i] = h_points[i];
    }
    return detail::MSMPrecompute<Config, MSM_CURVE::Point, MSM_CURVE::PointAffine>::precompute(len, h_points_array, max_cards);
}