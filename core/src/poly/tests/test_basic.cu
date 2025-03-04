#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/poly_basic.cuh"
#include <iostream>

using mont::u32;
using mont::u64;
using Field = bn254_fr::Element;
using Number = mont::Number<Field::LIMBS>;

u64 len = 1 << 24;

TEST_CASE("naive poly add") {
    std::cout << "testing the naive poly add" << std::endl;
    Field * a, *dst;
    a = new Field [len];
    dst = new Field [len];

    for (u64 i = 0; i < len; i++) {
        a[i] = Field::host_random();
    }

    Field *a_d;
    cudaMalloc(&a_d, len * Field::LIMBS * sizeof(u32));

    u32 block = 1024;
    u32 grid = (len - 1) / block + 1;
    auto a_iter = detail::SliceIterator<Field>(a_d, len);
    detail::poly_add<Field><<<grid, block >>>(a_iter, a_iter, a_iter, len);

    cudaMemcpy(a_d, a, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    detail::poly_add<Field><<<grid, block >>>(a_iter, a_iter, a_iter, len);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;

    cudaMemcpy(dst, a_d, len * Field::LIMBS * sizeof(u32), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i < len; i++) {
        CHECK(dst[i] == a[i] + a[i]);
    }

    delete[] a;
    delete[] dst;
    cudaFree(a_d);
}

TEST_CASE("naive poly mul") {
    std::cout << "testing the naive poly mul" << std::endl;
    Field * a, *dst;
    a = new Field [len];
    dst = new Field [len];

    for (u64 i = 0; i < len; i++) {
        a[i] = Field::host_random();
    }

    Field *a_d;
    cudaMalloc(&a_d, len * Field::LIMBS * sizeof(u32));

    u32 block = 1024;
    u32 grid = (len - 1) / block + 1;

    cudaMemcpy(a_d, a, len * Field::LIMBS * sizeof(u32), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    auto a_iter = detail::SliceIterator<Field>(a_d, len);

    detail::poly_mul<Field><<<grid, block >>>(a_iter, a_iter, a_iter, len);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;

    cudaMemcpy(dst, a_d, len * Field::LIMBS * sizeof(u32), cudaMemcpyDeviceToHost);

    for (u64 i = 0; i < len; i++) {
        CHECK(dst[i] == a[i] * a[i]);
    }

    delete[] a;
    delete[] dst;
    cudaFree(a_d);
}
