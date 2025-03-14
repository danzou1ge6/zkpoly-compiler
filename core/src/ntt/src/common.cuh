#pragma once
#include "../../common/mont/src/field_impls.cuh"
#include "../../common/error/src/check.cuh"
#include "../../common/iter/src/iter.cuh"
#include <cuda/barrier>
#include <cub/cub.cuh>
#include <algorithm>

namespace detail {
using mont::u32;
using mont::u64;
using mont::usize;
using mont::i64;
using iter::make_slice_iter;
using iter::SliceIterator;

static __host__ __device__ __forceinline__ constexpr u32 log2_int(u32 x) {
    u32 ans = 0;
    while (x >>= 1) ans++;
    return ans;
}

// plan partition for NTT stages
u32 get_deg (u32 deg_stage, u32 max_deg_stage) {
    u32 deg_per_round;
    for (u32 rounds = 1; ; rounds++) {
        deg_per_round = rounds == 1 ? deg_stage : (deg_stage - 1) / rounds + 1;
        if (deg_per_round <= max_deg_stage) break;
    }
    return deg_per_round;
}
} // namespace detail