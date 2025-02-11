#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/scan_mul.cuh"
#include "../src/rotate.cuh"
#include <iostream>
#include <chrono>

using mont::u32;
using mont::u64;
using mont::usize;
using Field = bn254_fr::Element;
using Number = mont::Number<Field::LIMBS>;

template <typename Field>
void prefix_product(Field * p, Field *q, Field x0, u64 len) {
    q[0] = x0;
    for (int i = 1; i < len; i++) {
        q[i] = q[i - 1] * p[i - 1];
    }
}

TEST_CASE("gpu prefix product") {
    std::cout << "testing the gpu prefix product" << std::endl;
    Field * p, *q, *q_truth;
    u32 log_len = 24;
    u32 len = 1 << log_len;
    p = new Field [len];
    q = new Field [len];
    q_truth = new Field [len];

    for (u64 i = 0; i < len; i++) {
        p[i] = Field::host_random();
    }
    Field x0 = Field::host_random();
    
    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    prefix_product(p, q_truth, x0, len);
    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "baseline Time: " << elapsed.count() << "s" << std::endl;

    Field *p_d, *q_d, *x0_d;
    cudaMalloc(&p_d, len * sizeof(Field));
    cudaMalloc(&q_d, len * sizeof(Field));
    cudaMalloc(&x0_d, sizeof(Field));
    cudaMemcpy(q_d, p, len * sizeof(Field), cudaMemcpyHostToDevice);
    cudaMemcpy(x0_d, &x0, sizeof(Field), cudaMemcpyHostToDevice);

    int rotation = 1;
    detail::rotate<Field>(q_d, p_d, len, rotation, 0);

    void *temp_buffer;
    usize buffer_size = 0;
    detail::scan_mul<Field>(nullptr, &buffer_size, reinterpret_cast<u32*>(p_d), 0, reinterpret_cast<u32*>(q_d), 0, reinterpret_cast<u32*>(x0_d), len, 0);
    cudaMalloc(&temp_buffer, buffer_size);

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);

    detail::scan_mul<Field>(temp_buffer, nullptr,  reinterpret_cast<u32*>(p_d), 1, reinterpret_cast<u32*>(q_d), 0, reinterpret_cast<u32*>(x0_d), len, 0);

    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, end_gpu);
    std::cout << "gpu Time: " << milliseconds << "ms" << std::endl;
    cudaMemcpy(q, q_d, len * sizeof(Field), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i <= len - 1; i++) {
        CHECK(q[i] == q_truth[i]);
    }

    cudaFree(p_d);
    cudaFree(q_d);
    cudaFree(x0_d);

    delete[] p;
    delete[] q;
    delete[] q_truth;
}