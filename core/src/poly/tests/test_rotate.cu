#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/rotate.cuh"
#include "../src/common.cuh"
#include <iostream>
#include <chrono>
#include <vector>

using mont::u32;
using mont::u64;
using mont::usize;
using mont::i64;
using Field = bn254_fr::Element;

void cpu_rotate(Field* src, Field* dst, u64 len, i64 shift) {
    if (shift == 0) {
        memcpy(dst, src, sizeof(Field) * len);
        return;
    }
    
    if (shift > 0) {
        memcpy(dst + len - shift, src, sizeof(Field) * shift);
        memcpy(dst, src + shift, sizeof(Field) * (len - shift));
    } else {
        shift = -shift;
        memcpy(dst, src + len - shift, sizeof(Field) * shift);
        memcpy(dst + shift, src, sizeof(Field) * (len - shift));
    }
}

TEST_CASE("gpu rotation") {
    std::cout << "testing the gpu rotation" << std::endl;
    
    const u32 len = 1024;  // Test with 1024 elements
    Field *src = new Field[len];
    Field *dst = new Field[len];
    Field *dst_truth = new Field[len];

    // Initialize source array with random values
    for (u64 i = 0; i < len; i++) {
        src[i] = Field::host_random();
    }

    // Test cases with different shifts
    std::vector<i64> test_shifts = {0, 1, -1, (i64)(len/2), -(i64)(len/2), (i64)(len-1), -(i64)(len-1)};
    
    for (const i64& shift : test_shifts) {
        std::cout << "Testing rotation with shift = " << shift << std::endl;
        
        // Compute CPU truth
        cpu_rotate(src, dst_truth, len, shift);

        // Allocate GPU memory
        Field *src_d, *dst_d;
        cudaMalloc(&src_d, len * sizeof(Field));
        cudaMalloc(&dst_d, len * sizeof(Field));
        cudaMemcpy(src_d, src, len * sizeof(Field), cudaMemcpyHostToDevice);

        // Perform GPU rotation
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        detail::rotate<Field>(src_d, dst_d, len, shift, stream);
        cudaStreamSynchronize(stream);

        // Copy result back to host
        cudaMemcpy(dst, dst_d, len * sizeof(Field), cudaMemcpyDeviceToHost);

        // Verify results
        for (u64 i = 0; i < len; i++) {
            INFO("Mismatch at index ", i, " with shift ", shift);
            CHECK(dst[i] == dst_truth[i]);
            i64 j = (i + shift) % len;
            if (j < 0) j += len;
            CHECK(dst[i] == src[j]);
        }

        // Cleanup GPU resources
        cudaStreamDestroy(stream);
        cudaFree(src_d);
        cudaFree(dst_d);
    }

    // Cleanup host resources
    delete[] src;
    delete[] dst;
    delete[] dst_truth;
}

TEST_CASE("gpu rotation reversibility") {
    std::cout << "testing rotation reversibility" << std::endl;
    
    const u32 len = 1024;
    Field *original = new Field[len];
    Field *intermediate = new Field[len];
    Field *final_result = new Field[len];

    // Initialize with random values
    for (u64 i = 0; i < len; i++) {
        original[i] = Field::host_random();
    }

    // Allocate GPU memory
    Field *d_original, *d_intermediate, *d_final;
    cudaMalloc(&d_original, len * sizeof(Field));
    cudaMalloc(&d_intermediate, len * sizeof(Field));
    cudaMalloc(&d_final, len * sizeof(Field));
    
    // Copy initial data to GPU
    cudaMemcpy(d_original, original, len * sizeof(Field), cudaMemcpyHostToDevice);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Test reversibility with different shifts
    std::vector<i64> test_shifts = {1, -1, (i64)(len/4), -(i64)(len/4), (i64)(len/2), -(i64)(len/2)};
    
    for (const i64& shift : test_shifts) {
        std::cout << "Testing reversibility with shift = " << shift << std::endl;
        
        // Forward rotation
        detail::rotate<Field>(d_original, 
                            d_intermediate, 
                            len, shift, stream);
        
        // Backward rotation (opposite shift)
        detail::rotate<Field>(d_intermediate, d_final, 
                            len, -shift, stream);
        
        cudaStreamSynchronize(stream);

        // Copy result back to host
        cudaMemcpy(final_result, d_final, len * sizeof(Field), cudaMemcpyDeviceToHost);

        // Verify the final result matches the original
        for (u64 i = 0; i < len; i++) {
            INFO("Mismatch at index ", i, " after forward/backward rotation with shift ", shift);
            CHECK(final_result[i] == original[i]);
        }
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_original);
    cudaFree(d_intermediate);
    cudaFree(d_final);
    delete[] original;
    delete[] intermediate;
    delete[] final_result;
}