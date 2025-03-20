#pragma once
#include "common.cuh"
namespace detail {
    template <typename Field>
    __global__ void poly_add(
        SliceIterator<const Field> ita, SliceIterator<const Field> itb, SliceIterator<Field> dst, usize len, usize upper_len) {
        usize index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len && index < upper_len) {
            dst[index] = ita[index];
        } else {
            auto a = ita[index];
            auto b = itb[index];
            dst[index] = a + b;
        }
    }

    template <typename Field>
    __global__ void poly_mul(
        SliceIterator<const Field> ita, SliceIterator<const Field> itb, SliceIterator<Field> dst, usize len, usize upper_len) {
        usize index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len && index < upper_len) {
            dst[index] = ita[index];
        } else {
            auto a = ita[index];
            auto b = itb[index];
            dst[index] = a * b;
        }
    }

    template <typename Field>
    __global__ void poly_sub(
        SliceIterator<const Field> ita, SliceIterator<const Field> itb, SliceIterator<Field> dst, usize len,
        usize upper_len, bool neg) {
        usize index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len && index < upper_len) {
            if (neg) {
                dst[index] = ita[index].neg();
            } else {
                dst[index] = ita[index];
            }
        } else {
            auto a = ita[index];
            auto b = itb[index];
            if (neg) {
                dst[index] = b - a;
            } else {
                dst[index] = a - b;
            }
        }
    }

    template <typename Field>
    __global__ void poly_set_lagrange(SliceIterator<Field> dst, Field value, u64 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        dst[index] = value;
    }

    template <typename Field>
    __global__ void poly_set_coef(SliceIterator<Field> iter, Field value, u64 len) {
        u64 index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= len) return;
        if (index == 0) iter[index] = value;
        else iter[index] = Field::zero();
    }

    template <typename Field>
    __global__ void inverse_scalar(Field * target) {
        *target = (*target).invert();
    }

    template <typename Field>
    __global__ void scalar_pow(Field * target, u64 exp) {
        *target = (*target).pow(exp);
    }

    template <typename Field>
    cudaError_t poly_zero(PolyPtr target, cudaStream_t stream) {
        u64 len = target.len;
        u32 block = 256;
        u32 grid = (len - 1) / block + 1;
        auto iter = make_slice_iter<Field>(target);
        poly_set_coef<<<grid, block, 0, stream>>>(iter, Field::zero(), len);
        CUDA_CHECK(cudaGetLastError());
        return cudaSuccess;
    }

    template<typename Field>
    cudaError_t poly_one_lagrange(PolyPtr target, cudaStream_t stream) {
        u64 len = target.len;
        u32 block = 256;
        u32 grid = (len - 1) / block + 1;
        auto iter = make_slice_iter<Field>(target);
        poly_set_lagrange<<<grid, block, 0, stream>>>(iter, Field::one(), len);
        CUDA_CHECK(cudaGetLastError());
        return cudaSuccess;
    }

    template<typename Field>
    cudaError_t poly_one_coef(PolyPtr target, cudaStream_t stream) {
        u64 len = target.len;
        u32 block = 256;
        u32 grid = (len - 1) / block + 1;
        auto iter = make_slice_iter<Field>(target);
        poly_set_coef<<<grid, block, 0, stream>>>(iter, Field::one(), len);
        CUDA_CHECK(cudaGetLastError());
        return cudaSuccess;
    }
} // namespace detail