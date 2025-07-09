#pragma once
#include "common.cuh"
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

namespace detail {
// template <typename Field>
// class gen_roots_cub {
//     public:
//     struct get_iterator_to_range {
//         __host__ __device__ __forceinline__ auto operator()(u32 index) {
//             return thrust::make_constant_iterator(input_d[index]);
//         }
//         Field *input_d;
//     };
//     struct get_ptr_to_range {
//         __host__ __device__ __forceinline__ auto operator()(u32 index) {
//             return output_d + offsets_d[index];
//         }
//         Field *output_d;
//         u32 *offsets_d;
//     };
//     struct get_run_length {
//         __host__ __device__ __forceinline__ auto operator()(u32 index) {
//             return offsets_d[index + 1] - offsets_d[index];
//         }
//         uint32_t *offsets_d;
//     };
//     struct mont_mul {
//         __device__ __forceinline__ Field operator()(const Field &a, const Field &b) {
//             return a * b;
//         }
//     };
//     __host__ __forceinline__ cudaError_t operator() (u32 * roots, u32 len, const Field &unit) {
//         if (len == 0) return cudaSuccess;
//         if (len == 1) {
//             Field::one().store(roots);
//             return cudaSuccess;
//         }
        
//         const u32 num_ranges = 2;
//         Field input[num_ranges] = {Field::one(), unit}; // {one, unit}
//         Field * input_d;
//         CUDA_CHECK(cudaMalloc(&input_d, num_ranges * sizeof(Field)));
//         CUDA_CHECK(cudaMemcpy(input_d, input, num_ranges * sizeof(Field), cudaMemcpyHostToDevice));
//         u32 offset[] = {0, 1, len};
//         u32 * offset_d;
//         CUDA_CHECK(cudaMalloc(&offset_d, (num_ranges + 1) * sizeof(u32)));
//         CUDA_CHECK(cudaMemcpy(offset_d, offset, (num_ranges + 1) * sizeof(u32), cudaMemcpyHostToDevice));
//         Field * output_d;
//         CUDA_CHECK(cudaMalloc(&output_d, len * sizeof(Field)));

//         // Returns a constant iterator to the element of the i-th run
//         thrust::counting_iterator<u32> iota(0);
//         auto iterators_in = thrust::make_transform_iterator(iota, get_iterator_to_range{input_d});
//         // Returns the run length of the i-th run
//         auto sizes = thrust::make_transform_iterator(iota, get_run_length{offset_d});
//         // Returns pointers to the output range for each run
//         auto ptrs_out = thrust::make_transform_iterator(iota, get_ptr_to_range{output_d, offset_d});
//         // Determine temporary device storage requirements
//         void *tmp_storage_d = nullptr;
//         size_t temp_storage_bytes = 0;
//         CUDA_CHECK(cub::DeviceCopy::Batched(tmp_storage_d, temp_storage_bytes, iterators_in, ptrs_out, sizes, num_ranges));
//         // Allocate temporary storage
//         CUDA_CHECK(cudaMalloc(&tmp_storage_d, temp_storage_bytes));
//         // Run batched copy algorithm (used to perform runlength decoding)
//         // output_d       <-- [one, unit, unit, ... , unit]
//         CUDA_CHECK(cub::DeviceCopy::Batched(tmp_storage_d, temp_storage_bytes, iterators_in, ptrs_out, sizes, num_ranges));
//         CUDA_CHECK(cudaFree(tmp_storage_d));
//         CUDA_CHECK(cudaFree(input_d));
//         CUDA_CHECK(cudaFree(offset_d));

//         tmp_storage_d = nullptr;
//         temp_storage_bytes = 0;
//         mont_mul op;
//         CUDA_CHECK(cub::DeviceScan::InclusiveScan(tmp_storage_d, temp_storage_bytes, output_d, op, len));
//         CUDA_CHECK(cudaMalloc(&tmp_storage_d, temp_storage_bytes));
//         CUDA_CHECK(cub::DeviceScan::InclusiveScan(tmp_storage_d, temp_storage_bytes, output_d, op, len));
//         CUDA_CHECK(cudaMemcpy(roots, output_d, len * sizeof(Field), cudaMemcpyDeviceToHost));
//         CUDA_CHECK(cudaFree(output_d));
//         CUDA_CHECK(cudaFree(tmp_storage_d));
        
//         return cudaSuccess;
//     }
// };

template <typename Field>
void gen_pq_omegas(u32* pq, u32* omegas, u32 max_deg, u32 len, const Field& unit) {
    // pq: [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
    static const u32 WORDS = Field::LIMBS;
    memset(pq, 0, (1 << max_deg >> 1) * sizeof(u32) * WORDS);
    Field::one().store(pq);
    auto twiddle = unit.pow(len >> max_deg);
    if (max_deg > 1) {
        twiddle.store(pq + WORDS);
        auto last = twiddle;
        for (u64 i = 2; i < (1 << max_deg >> 1); i++) {
            last = last * twiddle;
            last.store(pq + i * WORDS);
        }
    }

    // omegas: [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]

    unit.store(omegas);
    auto last = unit;
    for (u32 i = 1; i < 32; i++) {
        last = last.square();
        last.store(omegas + i * WORDS);
    }
}
} // namespace detail