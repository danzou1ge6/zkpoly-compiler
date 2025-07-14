#include "common.cuh"
#include <cub/cub.cuh>

namespace detail {

// helper for cub radix sort
template<typename Number>
struct decomposer_t {
    __host__ __device__ auto operator()(Number& key) const {
        return key.to_tuple();
    }
};

template<typename Field, typename Number>
__global__ void to_number(SliceIterator<const Field> element, SliceIterator<Number> number, usize len) {
    usize idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    number[idx] = element[idx].to_number();
}

template<typename Field, typename Number>
__global__ void to_field(SliceIterator<const Number> number, SliceIterator<Field> element, usize len) {
    usize idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    element[idx] = Field::from_number(number[idx]);
}

template<typename Number>
__global__ void lookup_table(SliceIterator<const Number> input, SliceIterator<const Number> table, SliceIterator<Number> output_table, bool* table_flag, u32* output_table_offset, usize len) {
    usize idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    if (idx == 0 || input[idx] != input[idx - 1]) {
        output_table_offset[idx] = 1;
        auto goal = input[idx];
        long long found = -1;
        // binary search in increasing order
        u32 left = 0, right = len;
        while (left < right) {
            u32 mid = (left + right) / 2;
            auto current = table[mid];
            if (current < goal) {
                left = mid + 1;
            } else if (goal < current) {
                right = mid;
            } else {
                found = mid;
                output_table[idx] = current;
                table_flag[mid] = false; // mark mid as used
                break;
            }
        }
        assert(found != -1);     
    }
}

template<typename Number>
__global__ void fill_back(SliceIterator<const Number> input, SliceIterator<const Number> table, SliceIterator<Number> output_table, u32* output_table_offset, usize len) {
    usize idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    if (idx != 0 && input[idx] == input[idx - 1]) {
        output_table[idx] = table[idx - output_table_offset[idx]];
    }
}

__global__ void init_flag(bool *table_flag, usize len) {
    usize idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    table_flag[idx] = true;
}

__device__ u32 selected_number = 0;

template<typename Field>
cudaError_t permute(void *temp_buffer, usize *buffer_size, usize usable, ConstPolyPtr input, ConstPolyPtr table, PolyPtr res_input, PolyPtr res_table, cudaStream_t stream) {
    typedef mont::Number<Field::LIMBS> Number;
    usize temp_poly_size = usable * sizeof(Number);
    // round up to 16 bytes
    usize table_flag_size = div_ceil(usable * sizeof(bool), 16) * 16;
    usize output_table_offset_size = usable * sizeof(u32);

    auto input_iter = make_slice_iter<Field>(input);
    auto table_iter = make_slice_iter<Field>(table);
    auto useful_input_iter = make_slice_iter<Number>(res_input);
    auto useful_table_iter = make_slice_iter<Number>(res_table);
    auto res_input_iter = make_slice_iter<Field>(res_input);
    auto res_table_iter = make_slice_iter<Field>(res_table);

    auto useful_input_ptr = reinterpret_cast<Number*>(res_input.ptr);
    auto useful_table_ptr = reinterpret_cast<Number*>(res_table.ptr);
    auto temp_poly_ptr = reinterpret_cast<Number*>(temp_buffer);
    auto table_flag_ptr = reinterpret_cast<bool*>(reinterpret_cast<char*>(temp_buffer) + temp_poly_size);
    auto output_table_offset_ptr = reinterpret_cast<u32*>(reinterpret_cast<char*>(temp_buffer) + temp_poly_size + table_flag_size);
    auto cub_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(temp_buffer) + temp_poly_size + table_flag_size + output_table_offset_size);

    auto temp_poly_iter = make_slice_iter<Number>(PolyPtr{
        .ptr = reinterpret_cast<u32*>(temp_poly_ptr),
        .len = usable,
        .rotate = 0,
        .offset = 0,
        .whole_len = usable
    });

    if (temp_buffer == nullptr) {
        usize temp_sort_size = 0, temp_scan_size = 0, temp_select_size = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_sort_size, temp_poly_ptr, useful_input_ptr, usable, decomposer_t<Number>()));
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_scan_size, output_table_offset_ptr, output_table_offset_ptr, usable));
        CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, temp_select_size, temp_poly_ptr, table_flag_ptr, &selected_number, usable));
        // round up to 4 bytes
        temp_select_size = div_ceil(temp_select_size, 4) * 4;
        usize cub_size = std::max(temp_sort_size, temp_scan_size);
        cub_size = std::max(cub_size, temp_select_size + sizeof(u32)); // + sizeof(u32) for selected_number
        *buffer_size = temp_poly_size + table_flag_size + output_table_offset_size + cub_size;
        return cudaSuccess;
    }

    assert(input.len == table.len);
    assert(usable == res_input.len);
    assert(usable == res_table.len);
    assert(usable <= input.len);

    // as cub does not support iter for sort
    assert(res_input.offset == 0);
    assert(res_input.rotate == 0);
    assert(res_table.offset == 0);
    assert(res_table.rotate == 0);

    u32 block_size = 256;
    u32 grid_size = div_ceil(usable, block_size);

    // init table_flag
    init_flag<<<grid_size, block_size, 0, stream>>>(table_flag_ptr, usable);
    CUDA_CHECK(cudaGetLastError());
    // clear the output_table_offset
    CUDA_CHECK(cudaMemsetAsync(output_table_offset_ptr, 0, output_table_offset_size, stream));

    to_number<<<grid_size, block_size, 0, stream>>>(input_iter, temp_poly_iter, usable);
    CUDA_CHECK(cudaGetLastError());

    usize temp_sort_size = 0;
    void *d_temp_sort = nullptr;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        d_temp_sort, temp_sort_size,
        temp_poly_ptr, useful_input_ptr,
        usable, decomposer_t<Number>(), stream
    ))
    d_temp_sort = cub_ptr;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        d_temp_sort, temp_sort_size,
        temp_poly_ptr, useful_input_ptr,
        usable, decomposer_t<Number>(), stream
    ))

    to_number<<<grid_size, block_size, 0, stream>>>(table_iter, useful_table_iter, usable);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        d_temp_sort, temp_sort_size,
        useful_table_ptr, temp_poly_ptr,
        usable, decomposer_t<Number>(), stream
    ))

    // after the above sort, useful_input is the sorted input
    // temp_poly_ptr is the sorted table
    // useful_table is ready for filling

    lookup_table<<<grid_size, block_size, 0, stream>>>(
        static_cast<SliceIterator<const Number>>(useful_input_iter), 
        static_cast<SliceIterator<const Number>>(temp_poly_iter),
        useful_table_iter, table_flag_ptr, output_table_offset_ptr, usable
    );
    CUDA_CHECK(cudaGetLastError());

    // compute the prefix sum of table_flag and output_table_offset
    void *d_temp_scan = nullptr;
    usize temp_scan_size = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        d_temp_scan, temp_scan_size,
        output_table_offset_ptr, output_table_offset_ptr,
        usable, stream
    ))
    d_temp_scan = cub_ptr;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        d_temp_scan, temp_scan_size,
        output_table_offset_ptr, output_table_offset_ptr,
        usable, stream
    ))

    // select the remaining table
    void *d_temp_select = nullptr;
    usize temp_select_size = 0;

    CUDA_CHECK(cub::DeviceSelect::Flagged(
        d_temp_select, temp_select_size,
        temp_poly_ptr, table_flag_ptr,
        &selected_number, usable, stream
    ))
    u32 *d_res_size = reinterpret_cast<u32*>(reinterpret_cast<char*>(cub_ptr) + div_ceil(temp_select_size, 4) * 4);
    d_temp_select = cub_ptr;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        d_temp_select, temp_select_size,
        temp_poly_ptr, table_flag_ptr,
        d_res_size, usable, stream
    ))
    
    // fill back the table
    fill_back<<<grid_size, block_size, 0, stream>>>(
        static_cast<SliceIterator<const Number>>(useful_input_iter), 
        static_cast<SliceIterator<const Number>>(temp_poly_iter),
        useful_table_iter, output_table_offset_ptr, usable
    );
    CUDA_CHECK(cudaGetLastError());

    // convert back to field
    to_field<<<grid_size, block_size, 0, stream>>>(
        static_cast<SliceIterator<const Number>>(useful_table_iter), res_table_iter, usable
    );
    CUDA_CHECK(cudaGetLastError());
    to_field<<<grid_size, block_size, 0, stream>>>(
        static_cast<SliceIterator<const Number>>(useful_input_iter), res_input_iter, usable
    );
    CUDA_CHECK(cudaGetLastError());

    return cudaSuccess;
}
}