#include "common.cuh"

namespace detail {
    template<typename Field>
    __global__ void distribute_powers_kernel(SliceIterator<Field> target, const Field *powers, u64 power_num, unsigned long long len) {
        unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < len) {
            target[idx] = target[idx] * powers[idx % power_num];
        }
    }

    template<typename Field>
    cudaError_t distribute_powers(PolyPtr poly, const Field *powers, u64 power_num, cudaStream_t stream) {
        u64 len = poly.len;
        int block = 256;
        int grid = (len + block - 1) / block;
        auto iter = make_slice_iter<Field>(poly);
        distribute_powers_kernel<Field><<<grid, block, 0, stream>>>(iter, powers, power_num, len);
        return cudaGetLastError();
    }
}