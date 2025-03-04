#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/poly_eval.cuh"
#include "rotate.cuh"

typedef bn254_fr::Element Field;

Field* gen_poly(uint len) {
    Field* poly = new Field[len];
    for (uint i = 0; i < len; i++) {
        poly[i] = Field::host_random();
    }
    return poly;
}

Field eval_cpu(Field * poly, Field x, uint len) {
    Field res = Field::zero();
    for (int i = len - 1; i >= 0; i--) {
        res = res * x + poly[i];
    }
    return res;
}

TEST_CASE("eval_basic") {
    auto len = 1 << 24;
    auto poly = gen_poly(len);
    auto x = Field::host_random();
    Field* poly_d;
    cudaMalloc(&poly_d, len * Field::LIMBS * sizeof(uint));
    cudaMemcpy(poly_d, poly, len * Field::LIMBS * sizeof(uint), cudaMemcpyHostToDevice);
    uint *res_d;
    cudaMalloc(&res_d, Field::LIMBS * sizeof(uint));
    Field *x_d;
    cudaMalloc(&x_d, Field::LIMBS * sizeof(uint));
    cudaMemcpy(x_d, &x, Field::LIMBS * sizeof(uint), cudaMemcpyHostToDevice);
    uint *temp_buf;
    unsigned long temp_buf_size = 0;
    auto poly_ptr = ConstPolyPtr{reinterpret_cast<const uint*>(poly_d), len, 0, 0, len};
    detail::poly_eval<Field>(nullptr, &temp_buf_size, poly_ptr, res_d, x_d, 0);
    cudaMalloc(&temp_buf, temp_buf_size);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    detail::poly_eval<Field>(temp_buf, nullptr, poly_ptr, res_d, x_d, 0);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;
    Field res;
    cudaMemcpy(&res, res_d, Field::LIMBS * sizeof(uint), cudaMemcpyDeviceToHost);
    auto res_cpu = eval_cpu(poly, x, len);
    CHECK(res == res_cpu);

    cudaFree(poly_d);
    cudaFree(res_d);
    cudaFree(temp_buf);
    cudaFree(x_d);
    delete [] poly;
}

TEST_CASE("eval with rotate") {
    auto len = 1 << 24;
    auto poly = gen_poly(len);
    auto x = Field::host_random();
    Field* poly_d;
    cudaMalloc(&poly_d, len * Field::LIMBS * sizeof(uint));
    cudaMemcpy(poly_d, poly, len * Field::LIMBS * sizeof(uint), cudaMemcpyHostToDevice);
    uint *res_d;
    cudaMalloc(&res_d, Field::LIMBS * sizeof(uint));
    Field *x_d;
    cudaMalloc(&x_d, Field::LIMBS * sizeof(uint));
    cudaMemcpy(x_d, &x, Field::LIMBS * sizeof(uint), cudaMemcpyHostToDevice);

    Field *rotated_d;
    long long rotate = 1;
    cudaMalloc(&rotated_d, len * Field::LIMBS * sizeof(uint));
    rotating(poly_d, rotated_d, len, rotate, 0);
    uint *temp_buf;
    unsigned long temp_buf_size = 0;
    auto poly_ptr = ConstPolyPtr{reinterpret_cast<const uint*>(rotated_d), len, rotate, 0, len};
    detail::poly_eval<Field>(nullptr, &temp_buf_size, poly_ptr, 0, 0, 0);
    cudaMalloc(&temp_buf, temp_buf_size);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    detail::poly_eval<Field>(temp_buf, nullptr, poly_ptr, res_d, x_d, 0);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;
    Field res;
    cudaMemcpy(&res, res_d, Field::LIMBS * sizeof(uint), cudaMemcpyDeviceToHost);
    auto res_cpu = eval_cpu(poly, x, len);
    CHECK(res == res_cpu);

    cudaFree(poly_d);
    cudaFree(rotated_d);
    cudaFree(res_d);
    cudaFree(temp_buf);
    cudaFree(x_d);
    delete [] poly;
}