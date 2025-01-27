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

    template <typename Field>
    __global__ void poly_set(Field * dst, Field value, u64 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        dst[index] = value;
    }

    template<typename Field>
    cudaError_t poly_zero(u32 * target, u64 len, cudaStream_t stream) {
        u32 block = 256;
        u32 grid = (len - 1) / block + 1;
        poly_set<<<grid, block, 0, stream>>>(reinterpret_cast<Field*>(target), Field::zero(), len);
        CUDA_CHECK(cudaGetLastError());
        return cudaSuccess;
    }

    template<typename Field>
    cudaError_t poly_one(u32 * target, u64 len, cudaStream_t stream) {
        u32 block = 256;
        u32 grid = (len - 1) / block + 1;
        poly_set<<<grid, block, 0, stream>>>(reinterpret_cast<Field*>(target), Field::one(), len);
        CUDA_CHECK(cudaGetLastError());
        return cudaSuccess;
    }
} // namespace detail