#pragma once
#include "common.cuh"
namespace detail {
    template <typename Field>
    __global__ void poly_add(RotatingIterator<const Field> ita, RotatingIterator<const Field> itb, RotatingIterator<Field> dst, usize len) {
        usize index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        auto a = ita[index];
        auto b = itb[index];
        dst[index] = a + b;
    }

    template <typename Field>
    __global__ void poly_mul(RotatingIterator<const Field> ita, RotatingIterator<const Field> itb, RotatingIterator<Field> dst, usize len) {
        usize index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        auto a = ita[index];
        auto b = itb[index];
        dst[index] = a * b;
    }

    template <typename Field>
    __global__ void poly_sub(RotatingIterator<const Field> ita, RotatingIterator<const Field> itb, RotatingIterator<Field> dst, usize len) {
        usize index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        auto a = ita[index];
        auto b = itb[index];
        dst[index] = a - b;
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