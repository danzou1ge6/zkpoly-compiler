#include <stdio.h> // For printf
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert> // For assert
#include "../src/permute.cuh" // Relative path to permute.cuh
#include "../../ntt/tests/small_field.cuh" // Relative path to small_field.cuh
#include "../src/poly.h" // Relative path to poly.h for PolyPtr definitions
#include "../../common/error/src/check.cuh" // Relative path to check.cuh for CUDA_CHECK
#include "../../common/iter/src/iter.cuh" // Relative path to iter.cuh for make_slice_iter

// Assuming Number is in mont namespace
// Assuming permute is in detail namespace
using namespace mont;
using namespace detail;
using SmallField = small_field::Element;

// Helper function to convert u32 to SmallField
SmallField u32_to_small_field(u32 val) {
    // Assuming SmallField::LIMBS >= 1
    Number<SmallField::LIMBS> num = {0}; // Initialize all limbs to 0
    num.limbs[0] = val; // Set the first limb
    return SmallField::from_number(num);
}

// Helper function to convert SmallField to u32
u32 small_field_to_u32(const SmallField& elem) {
    // Assuming the u32 value is stored in the first limb
    return elem.to_number().limbs[0];
}


int main() {
    using Number = mont::Number<SmallField::LIMBS>; // Assuming LIMBS is defined in SmallField or its Params
    const usize n = 8;

    printf("Starting PermuteTest with SmallField...\n");

    // Host data
    std::vector<u32> h_input_raw = {2, 1, 1, 2, 1, 4, 3, 4};
    std::vector<u32> h_table_raw = {9, 4, 1, 6, 4, 2, 5, 3};
    std::vector<SmallField> h_input(n);
    std::vector<SmallField> h_table(n);
    for (usize i = 0; i < n; ++i) {
        h_input[i] = u32_to_small_field(h_input_raw[i]);
        h_table[i] = u32_to_small_field(h_table_raw[i]);
    }
    printf("Host data prepared.\n");

    // Device data
    SmallField *d_input, *d_table, *d_res_input, *d_res_table;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(SmallField)));
    CUDA_CHECK(cudaMalloc(&d_table, n * sizeof(SmallField)));
    CUDA_CHECK(cudaMalloc(&d_res_input, n * sizeof(SmallField)));
    CUDA_CHECK(cudaMalloc(&d_res_table, n * sizeof(SmallField)));
    printf("Device memory allocated.\n");

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(SmallField), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), n * sizeof(SmallField), cudaMemcpyHostToDevice));
    printf("Data copied to device.\n");

    // PolyPtr setup (using aggregate initialization)
    ConstPolyPtr input_ptr = {
        reinterpret_cast<u32*>(d_input), // ptr
        n,                               // len
        0,                               // offset
        0,                               // rotate
        n                                // whole_len
    };
    ConstPolyPtr table_ptr = {
        reinterpret_cast<u32*>(d_table), // ptr
        n,                               // len
        0,                               // offset
        0,                               // rotate
        n                                // whole_len
    };
    PolyPtr res_input_ptr = {
        reinterpret_cast<u32*>(d_res_input), // ptr
        n,                                  // len
        0,                                  // offset
        0,                                  // rotate
        n                                   // whole_len
    };
    PolyPtr res_table_ptr = {
        reinterpret_cast<u32*>(d_res_table), // ptr
        n,                                  // len
        0,                                  // offset
        0,                                  // rotate
        n                                   // whole_len
    };
    printf("PolyPtr setup complete.\n");

    // Get buffer size
    void* d_temp_buffer = nullptr;
    usize buffer_size = 0;
    cudaStream_t stream = 0; // Use default stream
    CUDA_CHECK(permute<SmallField>(nullptr, &buffer_size, n, input_ptr, table_ptr, res_input_ptr, res_table_ptr, stream));
    printf("Buffer size calculated: %zu bytes\n", buffer_size);

    // Allocate buffer
    CUDA_CHECK(cudaMalloc(&d_temp_buffer, buffer_size));
    printf("Temporary buffer allocated on device.\n");

    // Run permute
    printf("Running permute kernel...\n");
    CUDA_CHECK(permute<SmallField>(d_temp_buffer, &buffer_size, n, input_ptr, table_ptr, res_input_ptr, res_table_ptr, stream));
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel completion before copying back
    printf("Permute kernel finished.\n");

    // Copy results back
    std::vector<SmallField> h_res_input(n);
    std::vector<SmallField> h_res_table(n);
    CUDA_CHECK(cudaMemcpy(h_res_input.data(), d_res_input, n * sizeof(SmallField), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_table.data(), d_res_table, n * sizeof(SmallField), cudaMemcpyDeviceToHost));
    printf("Results copied back to host.\n");

    // Expected results
    // Input sorted: {1, 1, 1, 2, 2, 3, 4, 4}
    // Table sorted: {1, 2, 3, 4, 4, 5, 6, 9}
    // Permuted table corresponding to sorted input
    std::vector<u32> expected_input_raw = {1, 1, 1, 2, 2, 3, 4, 4};
    std::vector<u32> expected_table_raw = {1, 4, 5, 2, 6, 3, 4, 9};
    std::vector<SmallField> expected_input(n);
    std::vector<SmallField> expected_table(n);
    for (usize i = 0; i < n; ++i) {
        expected_input[i] = u32_to_small_field(expected_input_raw[i]);
        expected_table[i] = u32_to_small_field(expected_table_raw[i]);
    }
    printf("Expected results prepared.\n");

    // Verify results
    bool success = true;
    printf("Verifying results...\n");
    for (usize i = 0; i < n; ++i) {
        if (h_res_input[i] != expected_input[i]) {
            printf("Input mismatch at index %zu: got %u, expected %u\n", i, small_field_to_u32(h_res_input[i]), small_field_to_u32(expected_input[i]));
            success = false;
        }
        if (h_res_table[i] != expected_table[i]) {
            printf("Table mismatch at index %zu: got %u, expected %u\n", i, small_field_to_u32(h_res_table[i]), small_field_to_u32(expected_table[i]));
            success = false;
        }
    }

    // Cleanup
    printf("Cleaning up device memory...\n");
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_res_input));
    CUDA_CHECK(cudaFree(d_res_table));
    CUDA_CHECK(cudaFree(d_temp_buffer));
    printf("Cleanup complete.\n");

    if (success) {
        printf("PermuteTest with SmallField PASSED!\n");
        return 0; // Indicate success
    } else {
        printf("PermuteTest with SmallField FAILED!\n");
        return 1; // Indicate failure
    }
}