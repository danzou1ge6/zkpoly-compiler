#include "common.cuh"

namespace detail {
    template<typename Field>
    __global__ void distribute_powers_kernel(RotatingIterator<Field> target, const Field *powers, u64 power_num, unsigned long long len) {
        unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < len && idx % (power_num + 1) != 0) {
            target[idx] = target[idx] * powers[(idx % (power_num + 1)) - 1];
        }
    }

    template<typename Field>
    cudaError_t distribute_powers(Field *poly, long long rotate, const Field *powers, u64 power_num, u64 len, cudaStream_t stream) {
        int block = 256;
        int grid = (len + block - 1) / block;
        auto iter = make_rotating_iter(poly, rotate, len);
        distribute_powers_kernel<Field><<<grid, block, 0, stream>>>(iter, powers, power_num, len);
        return cudaGetLastError();
    }
}