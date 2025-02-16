#include "common.cuh"

namespace detail {
    template<typename Field>
    __global__ void distribute_zeta_kernel(RotatingIterator<Field> target, const Field *zeta, unsigned long long len) {
        unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < len && idx % 3 != 0) {
            target[idx] = target[idx] * zeta[(idx % 3) - 1];
        }
    }

    template<typename Field>
    cudaError_t distribute_pow_zeta(Field *poly, long long rotate, const Field *zeta, u64 len, cudaStream_t stream) {
        int block = 256;
        int grid = (len + block - 1) / block;
        auto iter = make_rotating_iter(poly, rotate, len);
        distribute_zeta_kernel<Field><<<grid, block, 0, stream>>>(iter, zeta, len);
        return cudaGetLastError();
    }
}