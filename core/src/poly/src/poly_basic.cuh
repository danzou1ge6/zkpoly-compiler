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
    __global__ void poly_set_lagrange(Field * dst, Field value, u64 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        dst[index] = value;
    }

    template <typename Field>
    __global__ void poly_set_coef(RotatingIterator<Field> iter, Field value, u64 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        if (index == 0) iter[index] = value;
        else iter[index] = Field::zero();
    }

    template<typename Field>
    cudaError_t poly_one_lagrange(u32 * target, u64 len, cudaStream_t stream) {
        u32 block = 256;
        u32 grid = (len - 1) / block + 1;
        poly_set_lagrange<<<grid, block, 0, stream>>>(reinterpret_cast<Field*>(target), Field::one(), len);
        CUDA_CHECK(cudaGetLastError());
        return cudaSuccess;
    }

    template<typename Field>
    cudaError_t poly_one_coef(u32 * target, i64 rotate, u64 len, cudaStream_t stream) {
        u32 block = 256;
        u32 grid = (len - 1) / block + 1;
        auto iter = make_rotating_iter(reinterpret_cast<Field*>(target), rotate, len);
        poly_set_coef<<<grid, block, 0, stream>>>(iter, Field::one(), len);
        CUDA_CHECK(cudaGetLastError());
        return cudaSuccess;
    }
} // namespace detail