#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/kate_division.cuh"
#include <iostream>
#include <chrono>

using mont::u32;
using mont::u64;
using mont::usize;
using Field = bn254_fr::Element;
using Number = mont::Number<Field::LIMBS>;

template <typename Field>
void kate_divison(u32 len_p, Field *p, Field b, Field *q) {
    b = Field::zero()-b;
    Field tmp = Field::zero();
    q[len_p - 1] = Field::zero();
    for (long long i = len_p - 2; i >= 0; i--) {
        q[i] = p[i + 1] - tmp;
        tmp = q[i] * b;
    }
}

TEST_CASE("gpu kate division") {
    std::cout << "testing the gpu kate division" << std::endl;
    Field * p, *q, *q_truth;
    u32 log_len = 24;
    u32 len = 1 << log_len;
    p = new Field [len];
    q = new Field [len];
    q_truth = new Field [len];

    for (u64 i = 0; i < len; i++) {
        p[i] = Field::host_random();
    }
    Field b = Field::host_random();
    
    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    kate_divison(len, p, b, q_truth);
    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "baseline Time: " << elapsed.count() << "s" << std::endl;

    Field *p_d, *q_d, *b_d;
    cudaMalloc(&p_d, len * sizeof(Field));
    cudaMalloc(&q_d, len * sizeof(Field));
    cudaMalloc(&b_d, sizeof(Field));
    cudaMemcpy(p_d, p, len * sizeof(Field), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, &b, sizeof(Field), cudaMemcpyHostToDevice);

    void *temp_buffer;
    usize buffer_size = 0;
    auto p_ptr = ConstPolyPtr{reinterpret_cast<const u32*>(p_d), len, 0, 0, len};
    auto q_ptr = PolyPtr{reinterpret_cast<u32*>(q_d), len, 0, 0, len};
    detail::kate_division<Field>(nullptr, &buffer_size, log_len, p_ptr, b_d, q_ptr, 0);
    cudaMalloc(&temp_buffer, buffer_size);

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);

    detail::kate_division<Field>(temp_buffer, nullptr, log_len, p_ptr, b_d, q_ptr, 0);

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
    cudaFree(b_d);

    delete[] p;
    delete[] q;
    delete[] q_truth;
}