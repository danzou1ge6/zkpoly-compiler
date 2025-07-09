#pragma once
#include "common.cuh"
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

namespace detail {
template <typename Field>
class gen_roots_cub {
    public:
    __host__ __forceinline__ cudaError_t operator() (u32 * roots, u32 len, const Field &unit) {
        if (len == 0) return cudaSuccess;

        Field::one().store(roots);
        // compute one by one
        for (u32 i = 1; i < len; i++) {
            reinterpret_cast<Field*>(roots)[i] = reinterpret_cast<Field*>(roots)[i - 1] * unit;
        }
        
        return cudaSuccess;
    }
};

template <typename Field>
void gen_pq_omegas(u32* pq, u32* omegas, u32 max_deg, u32 len, const Field& unit) {
    // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
    static const u32 WORDS = Field::LIMBS;
    memset(pq, 0, (1 << max_deg >> 1) * sizeof(u32) * WORDS);
    Field::one().store(pq);
    auto twiddle = unit.pow(len >> max_deg);
    if (max_deg > 1) {
        twiddle.store(pq + WORDS);
        auto last = twiddle;
        for (u64 i = 2; i < (1 << max_deg >> 1); i++) {
            last = last * twiddle;
            last.store(pq + i * WORDS);
        }
    }

    // omegas: [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]

    unit.store(omegas);
    auto last = unit;
    for (u32 i = 1; i < 32; i++) {
        last = last.square();
        last.store(omegas + i * WORDS);
    }
}
} // namespace detail