#pragma once
#include "common.cuh"
namespace detail {
    template <typename Field>
    __global__ void poly_add(const u32 * a, const u32 * b, u32 * dst, u64 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        auto a_val = Field::load(a + index * Field::LIMBS);
        auto b_val = Field::load(b + index * Field::LIMBS);
        (a_val + b_val).store(dst + index * Field::LIMBS);
    }

    template <typename Field>
    __global__ void poly_mul(const u32 * a, const u32 * b, u32 * dst, u64 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        auto a_val = Field::load(a + index * Field::LIMBS);
        auto b_val = Field::load(b + index * Field::LIMBS);
        (a_val * b_val).store(dst + index * Field::LIMBS);
    }

    template <typename Field>
    __global__ void poly_sub(const u32 * a, const u32 * b, u32 * dst, u64 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        auto a_val = Field::load(a + index * Field::LIMBS);
        auto b_val = Field::load(b + index * Field::LIMBS);
        (a_val - b_val).store(dst + index * Field::LIMBS);
    }
} // namespace detail