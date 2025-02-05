#include "common.cuh"

namespace detail {

template <typename Field>
__global__ void kate_kernel(const Field *p, Field *q, const Field *pow_b, const Field *b, u32 len_p, u32 deg) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len_p / 2) return;
    u32 seg_len = 1 << deg;
    u32 inter_seg_id = index / seg_len;
    u32 intra_seg_id = index % seg_len;
    u32 seg_start = inter_seg_id * seg_len * 2;
    u32 sum_id = seg_start + seg_len;
    q[seg_start + intra_seg_id] = q[seg_start + intra_seg_id] + (q[sum_id] * (*b) + p[sum_id]) * pow_b[seg_len - 1 - intra_seg_id];
}

template <typename Field>
__global__ void local_kate_kernel(const Field *p, Field *q, const Field *pow_b, const Field *b, u32 len_p, u32 max_deg) {
    extern __shared__ Field shared[];
    u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len_p / 2) return;
    u32 seg_len = 1 << (max_deg - 1);
    assert(seg_len >= threadIdx.x);
    u32 inter_seg_id = index / seg_len;
    u32 seg_start = inter_seg_id * seg_len * 2;
    p += seg_start;
    q += seg_start;
    for (u32 i = threadIdx.x; i < seg_len * 2 * Field::LIMBS / 4; i += seg_len) {
        reinterpret_cast<uint4*>(shared)[i] = uint4{0,0,0,0};
    }
    __syncthreads();
    index = threadIdx.x;
    for (u32 deg = 0; deg < max_deg; deg++) {
        u32 seg_len = 1 << deg;
        u32 inter_seg_id = index / seg_len;
        u32 intra_seg_id = index % seg_len;
        u32 seg_start = inter_seg_id * seg_len * 2;
        u32 sum_id = seg_start + seg_len;
        shared[seg_start + intra_seg_id] = shared[seg_start + intra_seg_id] + (shared[sum_id] * (*b) + p[sum_id]) * pow_b[seg_len - 1 - intra_seg_id];
        __syncthreads();
    }
    for (u32 i = threadIdx.x; i < seg_len * 2 * Field::LIMBS / 4; i += seg_len) {
        reinterpret_cast<uint4*>(q)[i] = reinterpret_cast<uint4*>(shared)[i];
    }
}

// assume len_p = 2^k
template <typename Field>
cudaError_t kate_division(void *temp_buffer, usize *buffer_size, u32 log_p, const Field *p, const Field *b, Field *q, cudaStream_t stream) {
    u64 len_p = 1 << log_p;
    usize pow_sz = len_p / 2 * Field::LIMBS * sizeof(u32);
    if (temp_buffer == nullptr) {
        usize scan_sz = 0;
        CUDA_CHECK(get_pow_series<Field>(nullptr, &scan_sz, nullptr, b, len_p / 2, 0));
        *buffer_size = scan_sz + pow_sz;
        return cudaSuccess;
    }
    u32 *pow_b = reinterpret_cast<u32*>(temp_buffer);
    CUDA_CHECK(get_pow_series<Field>(reinterpret_cast<char*>(temp_buffer) + pow_sz, nullptr, pow_b, b, len_p / 2, stream));

    u32 log_threads = 8;
    u32 threads = 1 << log_threads;
    u32 blocks = (len_p / 2 + threads - 1) / threads;

    CUDA_CHECK(cudaFuncSetAttribute(local_kate_kernel<Field>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * threads * sizeof(Field)));

    local_kate_kernel<Field> <<<blocks, threads, threads * 2 * sizeof(Field), stream >>> (reinterpret_cast<const Field *>(p), reinterpret_cast<Field *>(q), reinterpret_cast<Field *>(pow_b), b, len_p, std::min(log_threads + 1, log_p));

    CUDA_CHECK(cudaGetLastError());

    for (int deg = log_threads + 1; deg < log_p; deg++) {        
        kate_kernel<Field> <<<blocks, threads>>> (p, q, reinterpret_cast<Field *>(pow_b), b, len_p, deg);
    }
    CUDA_CHECK(cudaGetLastError());

    return cudaSuccess;
}

} // namespace detail