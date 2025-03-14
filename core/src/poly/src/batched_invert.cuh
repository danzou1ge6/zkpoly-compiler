#include "common.cuh"

namespace detail {

// too slow
// template <typename Field>
// __global__ void naive_invert(Field *p, u64 len) {
//     u64 index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index >= len) return;
//     p[index] = p[index].invert();
// }

template <typename Field>
__global__ void copy_and_check_zero(SliceIterator<Field> p, Field *copy, u32* bitmask, u64 len) {
    u64 index = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
    u32 local_mask = 0;
    for (u64 i = index; i < index + 32 && i < len; i++) {
        auto num = p[i];
        bool is_zero = num.is_zero();
        if (is_zero) {
            local_mask |= 1 << (i - index);
            p[i] = Field::one();
            copy[len - i - 1] = Field::one();
        } else {
            copy[len - i - 1] = num;
        }
    }
    bitmask[index / 32] = local_mask;
}

template <typename Field>
__global__ void recover_zero(SliceIterator<Field> p, u32* bitmask, u64 len) {
    u64 index = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
    u32 local_mask = bitmask[index / 32];
    for (u64 i = index; i < index + 32 && i < len; i++) {
        if (local_mask & (1 << (i - index))) {
            p[i] = Field::zero();
        }
    }
}

template <typename Field>
__global__ void invert_one(SliceIterator<Field> p, const Field *copy, u64 len, Field *invert) {
    assert(blockDim.x == 1 && blockIdx.x == 0); // one thread
    invert[0] = (copy[0] * p[len - 1]).invert();
}

template <typename Field>
__global__ void invert(SliceIterator<Field> p, const Field *copy, const Field *invert, u64 len) {
    u64 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    p[index] = p[index] * invert[0] * copy[len - 1 - index];
}

template <typename Field>
cudaError_t batched_invert(void* temp_buffer, usize *buffer_size, PolyPtr target, cudaStream_t stream) {
    u64 len = target.len;
    auto target_iter = make_slice_iter<Field>(target);
    usize mask_sz = (len + 31) / 32 * sizeof(u32);
    usize copy_sz = len * sizeof(Field);
    usize inv_sz = sizeof(Field);
    auto mul_op = [] __device__ __host__(const Field &a, const Field &b) { return a * b; };
    usize temp_scan_size = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveScan(nullptr, temp_scan_size, target_iter, mul_op, Field::one(), len));
    if (temp_buffer == nullptr) {
        *buffer_size = temp_scan_size + mask_sz + copy_sz + inv_sz;
        return cudaSuccess;
    }

    Field *copy = reinterpret_cast<Field*>(temp_buffer);
    u32 *bitmask = reinterpret_cast<u32*>(reinterpret_cast<char*>(temp_buffer) + copy_sz);
    Field *inv = reinterpret_cast<Field*>(reinterpret_cast<char*>(temp_buffer) + mask_sz + copy_sz);
    void *d_temp_scan = reinterpret_cast<char*>(temp_buffer) + mask_sz + copy_sz + inv_sz;

    u32 block = 256;
    u32 grid = div_ceil(len, block * 32);

    copy_and_check_zero<Field><<< grid, block, 0, stream>>>(target_iter, copy, bitmask, len);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cub::DeviceScan::ExclusiveScan(d_temp_scan, temp_scan_size, target_iter, mul_op, Field::one(), len, stream));
    
    invert_one<Field><<<1, 1, 0, stream>>>(target_iter, copy, len, inv);
    CUDA_CHECK(cub::DeviceScan::ExclusiveScan(d_temp_scan, temp_scan_size, copy, mul_op, Field::one(), len, stream));
    invert<Field><<<div_ceil(len, block), block, 0, stream>>>(target_iter, copy, inv, len);
    CUDA_CHECK(cudaGetLastError());

    recover_zero<Field><<<grid, block, 0, stream>>>(target_iter, bitmask, len);
    CUDA_CHECK(cudaGetLastError());

    return cudaSuccess;
}

} // namespace detail