#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/batched_invert.cuh"
#include <iostream>
#include <chrono>

using mont::u32;
using mont::u64;
using mont::usize;
using Field = bn254_fr::Element;
using Number = mont::Number<Field::LIMBS>;

TEST_CASE("gpu invert") {
    std::cout << "testing the gpu invert" << std::endl;
    Field * p, *q;
    u32 log_len = 10;
    u32 len = 1 << log_len;
    p = new Field [len];
    q = new Field [len];

    for (u64 i = 0; i < len; i++) {
        if (i % 7 == 0) {
            p[i] = Field::zero();
        } else {
            p[i] = Field::host_random();
        }
    }

    Field *p_d;
    cudaMalloc(&p_d, len * sizeof(Field));
    cudaMemcpy(p_d, p, len * sizeof(Field), cudaMemcpyHostToDevice);

    void *temp_buffer;
    usize buffer_size = 0;
    auto p_ptr = PolyPtr{reinterpret_cast<u32*>(p_d), len, 0, 0, len};
    detail::batched_invert<Field>(nullptr, &buffer_size, p_ptr, 0);
    cudaMalloc(&temp_buffer, buffer_size);

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);

    detail::batched_invert<Field>(temp_buffer, 0, p_ptr, 0);

    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, end_gpu);
    std::cout << "gpu Time: " << milliseconds << "ms" << std::endl;
    cudaMemcpy(q, p_d, len * sizeof(Field), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i <= len - 1; i++) {
        CHECK(q[i] == p[i].invert());
    }

    cudaFree(p_d);
    cudaFree(temp_buffer);

    delete[] p;
    delete[] q;
}