#!/usr/bin/env python3
import re
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class KernelParam:
    is_scalar: bool
    is_mut: bool
    var_name: str # Keep a representative name
    c_index: int  # Index within the C function's vars or mut_vars array
    alloc_index: int = field(init=False) # Overall index in the params list for allocation naming

def parse_cuda_kernel(file_content: str) -> Tuple[str, List[KernelParam], List[int], int, int]:
    """解析CUDA kernel文件,提取函数名,参数信息,以及最大索引"""
    # 提取函数名
    match = re.search(r'extern "C" cudaError_t (\w+)', file_content)
    if not match:
        match = re.search(r'__global__\s+void\s+(\w+)\s*\(', file_content)
        if not match:
            raise ValueError("找不到 extern C 或 __global__ 函数定义")
        print(f"Warning: Found __global__ function '{match.group(1)}', assuming extern \"C\" wrapper exists with the same name.")
    func_name = match.group(1)

    # 提取kernel ID
    kernel_ids = []
    if match := re.search(r'// Included names: ([\d, ]+)', file_content):
        kernel_ids = [int(x.strip()) for x in match.group(1).strip().rstrip(',').split(',')]

    # --- 更保守的解析逻辑 ---
    param_info: Dict[Tuple[bool, int], Dict] = {}
    max_const_c_idx = -1
    max_mut_c_idx = -1

    lines = file_content.split('\n')
    for line in lines:
        c_idx = -1
        is_mut = False
        # 默认是标量，只有明确看到 make_slice_iter 才认为是数组
        is_scalar_usage = True
        var_name = ""
        found = False

        # 检查数组用法 (make_slice_iter) - 这是唯一判断为数组的情况
        const_iter_match = re.search(r'make_slice_iter<FUSED_FIELD>\(vars\[(\d+)\]', line)
        mut_iter_match = re.search(r'make_slice_iter<FUSED_FIELD>\(mut_vars\[(\d+)\]', line)

        if const_iter_match:
            c_idx = int(const_iter_match.group(1))
            is_mut = False
            is_scalar_usage = False # 判断为数组
            name_match = re.search(r'auto\s+(iter\d+)', line)
            var_name = name_match.group(1) if name_match else f"const_iter_{c_idx}"
            found = True
            max_const_c_idx = max(max_const_c_idx, c_idx)
        elif mut_iter_match:
            c_idx = int(mut_iter_match.group(1))
            is_mut = True
            is_scalar_usage = False # 判断为数组
            name_match = re.search(r'auto\s+(iter\d+)', line)
            var_name = name_match.group(1) if name_match else f"mut_iter_{c_idx}"
            found = True
            max_mut_c_idx = max(max_mut_c_idx, c_idx)
        else:
            # 检查其他用法 (包括 reinterpret_cast<...*>(... .ptr)) - 都视为标量
            const_any_match = re.search(r'vars\[(\d+)\]', line)
            mut_any_match = re.search(r'mut_vars\[(\d+)\]', line)

            if const_any_match:
                c_idx = int(const_any_match.group(1))
                is_mut = False
                is_scalar_usage = True # 视为标量
                # Try to get a descriptive name if possible
                name_match = re.search(r'auto\s+((?:var|iter)\d+)', line)
                var_name = name_match.group(1) if name_match else f"const_ref_{c_idx}"
                found = True
                max_const_c_idx = max(max_const_c_idx, c_idx)
            elif mut_any_match:
                c_idx = int(mut_any_match.group(1))
                is_mut = True
                is_scalar_usage = True # 视为标量
                name_match = re.search(r'auto\s+((?:var|iter)\d+)', line)
                var_name = name_match.group(1) if name_match else f"mut_ref_{c_idx}"
                found = True
                max_mut_c_idx = max(max_mut_c_idx, c_idx)

        # 更新 param_info 字典 (只记录第一次遇到的类型)
        if found:
            key = (is_mut, c_idx)
            if key not in param_info:
                param_info[key] = {'is_scalar': is_scalar_usage, 'var_name': var_name}
            # else: # 如果已存在，不再更新，保留第一次判断的类型
            #    pass

    # --- 从 param_info 构建最终的 params 列表 ---
    params = []
    alloc_idx_counter = 0
    sorted_keys = sorted(param_info.keys())

    for key in sorted_keys:
        is_mut, c_idx = key
        info = param_info[key]
        param = KernelParam(
            is_scalar=info['is_scalar'],
            is_mut=is_mut,
            var_name=info['var_name'],
            c_index=c_idx
        )
        param.alloc_index = alloc_idx_counter
        params.append(param)
        alloc_idx_counter += 1

    return func_name, params, kernel_ids, max_const_c_idx, max_mut_c_idx


def generate_benchmark(kernel_file: str, output_file: str, array_size: int = 1024):
    """生成benchmark文件"""
    # 解析kernel文件
    try:
        with open(kernel_file) as f:
            content = f.read()
            func_name, params, kernel_ids, max_const_c_idx, max_mut_c_idx = parse_cuda_kernel(content)
    except Exception as e:
        print(f"Error parsing kernel file '{kernel_file}': {e}")
        raise

    # 获取原始 .cu 文件名，用于 #include
    original_cu_filename = os.path.basename(kernel_file)

    # 输出要处理的kernel信息
    print(f"Processing kernel: {func_name}")
    if kernel_ids:
        print(f"Kernel IDs: {kernel_ids}")

    # 分离参数
    const_vars = [p for p in params if not p.is_mut]
    mut_vars = [p for p in params if p.is_mut]

    # 输出参数统计
    print(f"\nParameter Summary:")
    num_const_scalar = len([p for p in const_vars if p.is_scalar])
    num_const_array = len([p for p in const_vars if not p.is_scalar])
    num_mut_scalar = len([p for p in mut_vars if p.is_scalar])
    num_mut_array = len([p for p in mut_vars if not p.is_scalar])
    print(f"- Max Const C Index: {max_const_c_idx}")
    print(f"- Max Mut C Index: {max_mut_c_idx}")
    print(f"- Constant scalars (parsed): {num_const_scalar}")
    print(f"- Constant arrays (parsed): {num_const_array}")
    print(f"- Mutable scalars (parsed): {num_mut_scalar}")
    print(f"- Mutable arrays (parsed): {num_mut_array}")
    print(f"- Total unique parameters parsed: {len(params)}")

    # 计算 vector 大小
    vars_vec_size = max_const_c_idx + 1 if max_const_c_idx >= 0 else 0
    mut_vars_vec_size = max_mut_c_idx + 1 if max_mut_c_idx >= 0 else 0


    # 生成C++代码
    with open(output_file, 'w') as f:
        # 包含头文件
        f.write(f"""// Benchmark generated for {original_cu_filename}
#include <cuda_runtime.h>
#include "../../common/mont/src/field_impls.cuh"
#include "../../common/iter/src/iter.cuh"
#include <vector>
#include <random>
#include <chrono>
#include <cstdio> // Include for printf
#include <stdexcept> // For runtime_error
#include <string> // For error messages
#include <map> // For storing pointers by alloc_index

// Include the original kernel definition file
// Make sure this path is correct relative to the compilation environment
#include "{original_cu_filename}"

using namespace mont;
using namespace iter;

// 定义FUSED_FIELD类型 (如果原始文件没有定义，可能需要取消注释或调整)
#ifndef FUSED_FIELD
#define FUSED_FIELD bn254_fr::Element
#warning "FUSED_FIELD was not defined, defaulting to bn254_fr::Element. Please verify."
#endif


// Helper function to check CUDA errors
static void checkCudaError(cudaError_t err, const char* file, int line) {{
    if (err != cudaSuccess) {{
        std::string msg = "CUDA error: ";
        msg += cudaGetErrorString(err);
        msg += " in ";
        msg += file;
        msg += " at line ";
        msg += std::to_string(line);
        throw std::runtime_error(msg);
    }}
}}
#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)


// 生成随机field元素
void generate_random_field(FUSED_FIELD* data, size_t n) {{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dis;

    constexpr size_t limbs_per_element = sizeof(FUSED_FIELD) / sizeof(uint32_t);
    static_assert(sizeof(FUSED_FIELD) % sizeof(uint32_t) == 0, "FUSED_FIELD size must be a multiple of uint32_t size");

    uint32_t* u32_data = reinterpret_cast<uint32_t*>(data);
    for(size_t i = 0; i < n * limbs_per_element; i++) {{
        u32_data[i] = dis(gen);
    }}
}}

// 外部函数声明 (由包含的 .cu 文件提供)
// extern "C" cudaError_t {func_name}(ConstPolyPtr const* vars, PolyPtr const* mut_vars, unsigned long long len, cudaStream_t stream);

int main() {{
    std::vector<FUSED_FIELD*> h_buffers_cleanup;
    std::vector<FUSED_FIELD*> d_buffers_cleanup;

    try {{
        CHECK_CUDA_ERROR(cudaSetDevice(0));

        size_t array_size = {array_size};
        constexpr int NUM_WARMUP = 5;
        constexpr int NUM_RUNS = 100;
        size_t total_array_bytes = 0;

        printf("Initializing benchmark for kernel '%s' with array_size = %zu\\n", "{func_name}", array_size);

        // --- Memory Allocation ---
        printf("Allocating host and device memory...\\n");
        std::map<int, FUSED_FIELD*> h_buffer_map;
        std::map<int, FUSED_FIELD*> d_buffer_map;

        // Python loop generates C++ allocation code for each parameter
""")
        # --- Start of Python generating C++ allocation ---
        for param in params:
            alloc_idx = param.alloc_index
            is_scalar = param.is_scalar # Use the potentially more conservative is_scalar value
            is_mut = param.is_mut
            element_count_str = "1" if is_scalar else "array_size"
            buffer_size_bytes_str = f"sizeof(FUSED_FIELD) * {element_count_str}"
            base_name = f"{'mut_' if is_mut else ''}buffer_alloc{alloc_idx}"

            if not is_scalar:
                f.write(f"        total_array_bytes += {buffer_size_bytes_str}; // {base_name}\n")

            f.write(f"""
        // Allocate for param alloc_index={alloc_idx} ({base_name}, scalar={is_scalar})
        {{
            size_t element_count = {element_count_str};
            size_t buffer_size_bytes = {buffer_size_bytes_str};
            FUSED_FIELD* h_ptr = nullptr;
            FUSED_FIELD* d_ptr = nullptr;
            try {{
                h_ptr = new FUSED_FIELD[element_count];
                h_buffers_cleanup.push_back(h_ptr);
                generate_random_field(h_ptr, element_count);
                h_buffer_map[{alloc_idx}] = h_ptr;

                CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ptr, buffer_size_bytes));
                d_buffers_cleanup.push_back(d_ptr);
                d_buffer_map[{alloc_idx}] = d_ptr;

                CHECK_CUDA_ERROR(cudaMemcpy(d_ptr, h_ptr, buffer_size_bytes, cudaMemcpyHostToDevice));
            }} catch (...) {{
                 printf("Error during allocation/copy for alloc_index={alloc_idx}. Rethrowing.\\n");
                 throw;
            }}
        }}
""")
        # --- End of Python generating C++ allocation ---

        f.write(f"""
        printf("Memory allocation and copy complete. Total array bytes: %.2f MB\\n", total_array_bytes / (1024.0 * 1024.0));


        // --- PolyPtr Construction ---
        printf("Constructing PolyPtr/ConstPolyPtr vectors...\\n");
        std::vector<ConstPolyPtr> vars_vec;
        std::vector<PolyPtr> mut_vars_vec;
        // Resize based on max index + 1 and initialize elements
        const ConstPolyPtr default_const_poly = {{nullptr, 0, 0, 0, 0}};
        const PolyPtr default_mut_poly = {{nullptr, 0, 0, 0, 0}};
        vars_vec.resize({vars_vec_size}, default_const_poly);
        mut_vars_vec.resize({mut_vars_vec_size}, default_mut_poly);
        printf("vars_vec size: %zu, mut_vars_vec size: %zu\\n", vars_vec.size(), mut_vars_vec.size());


        // Python loop generates C++ PolyPtr construction code
""")
        # --- Start of Python generating C++ PolyPtr construction ---
        # Iterate through all params to fill the vectors correctly
        for param in params:
            alloc_idx = param.alloc_index
            c_idx = param.c_index
            is_scalar = param.is_scalar # Use the potentially more conservative is_scalar value
            is_mut = param.is_mut
            element_count_str = "1" if is_scalar else "array_size"
            d_ptr_access = f"d_buffer_map.at({alloc_idx})"

            if not is_mut:
                # Place into vars_vec at the correct c_index
                f.write(f"""
        // Construct ConstPolyPtr for c_index={c_idx} (from alloc_index={alloc_idx}, scalar={is_scalar})
        if ({c_idx} >= vars_vec.size()) {{
            throw std::runtime_error("Const parameter c_index {c_idx} out of bounds for vars_vec size " + std::to_string(vars_vec.size()));
        }}
        try {{
            vars_vec[{c_idx}] = {{
                .ptr = reinterpret_cast<const u32*>(reinterpret_cast<const uint32_t*>({d_ptr_access})),
                .len = {element_count_str}, // Use correct len based on parsed type
                .rotate = 0,
                .offset = 0,
                .whole_len = {element_count_str} // Use correct len based on parsed type
            }};
        }} catch (const std::out_of_range& oor) {{
             throw std::runtime_error("Failed to find device pointer for alloc_index {alloc_idx} in d_buffer_map");
        }}
""")
            else:
                # Place into mut_vars_vec at the correct c_index
                f.write(f"""
        // Construct PolyPtr for c_index={c_idx} (from alloc_index={alloc_idx}, scalar={is_scalar})
        if ({c_idx} >= mut_vars_vec.size()) {{
            throw std::runtime_error("Mutable parameter c_index {c_idx} out of bounds for mut_vars_vec size " + std::to_string(mut_vars_vec.size()));
        }}
         try {{
            mut_vars_vec[{c_idx}] = {{
                .ptr = reinterpret_cast<u32*>(reinterpret_cast<uint32_t*>({d_ptr_access})),
                .len = {element_count_str}, // Use correct len based on parsed type
                .rotate = 0,
                .offset = 0,
                .whole_len = {element_count_str} // Use correct len based on parsed type
            }};
        }} catch (const std::out_of_range& oor) {{
             throw std::runtime_error("Failed to find device pointer for alloc_index {alloc_idx} in d_buffer_map");
        }}
""")
        # --- End of Python generating C++ PolyPtr construction ---

        f.write(f"""
        printf("PolyPtr construction complete.\\n");

        // Optional debug print
        /*
        printf("\\n--- Debug PolyPtr Contents ---\\n");
        for(size_t i = 0; i < vars_vec.size(); ++i) {{
            printf("vars_vec[%zu]: ptr=%p, len=%llu\\n", i, (void*)vars_vec[i].ptr, vars_vec[i].len);
        }}
        for(size_t i = 0; i < mut_vars_vec.size(); ++i) {{
            printf("mut_vars_vec[%zu]: ptr=%p, len=%llu\\n", i, (void*)mut_vars_vec[i].ptr, mut_vars_vec[i].len);
        }}
        printf("----------------------------\\n\\n");
        */

        // --- CUDA Stream and Events ---
        cudaStream_t stream;
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

        // --- Benchmarking ---
        printf("Warming up for %d iterations...\\n", NUM_WARMUP);
        for(int i = 0; i < NUM_WARMUP; i++) {{
            CHECK_CUDA_ERROR({func_name}(vars_vec.data(), mut_vars_vec.data(), array_size, stream));
        }}
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        printf("Warmup complete.\\n");

        printf("Running benchmark for %d iterations...\\n", NUM_RUNS);
        auto start = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < NUM_RUNS; i++) {{
            CHECK_CUDA_ERROR({func_name}(vars_vec.data(), mut_vars_vec.data(), array_size, stream));
        }}
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double total_ms = duration.count() / 1000.0;
        double avg_ms = total_ms / NUM_RUNS;

        // --- Results ---
        printf("\\n--- Benchmark Results ---\\n");
        printf("Kernel: %s\\n", "{func_name}");
        printf("Array Size (N): %zu elements\\n", array_size);
        printf("Total Array Data: %.2f MB\\n", total_array_bytes / (1024.0 * 1024.0));
        printf("Iterations: %d\\n", NUM_RUNS);
        printf("Total Time: %.3f ms\\n", total_ms);
        printf("Average Kernel Time: %.3f ms\\n", avg_ms);
        if (total_array_bytes > 0 && avg_ms > 0) {{
            double throughput_gb_s = (total_array_bytes / 1e9) / (avg_ms / 1000.0);
            printf("Throughput (Arrays Only): %.2f GB/s\\n", throughput_gb_s);
        }} else {{
            printf("Throughput (Arrays Only): N/A (No array data or zero time)\\n");
        }}
        printf("-------------------------\\n");

        // --- Cleanup ---
        printf("\\nCleaning up resources...\\n");
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

        for (FUSED_FIELD* h_ptr : h_buffers_cleanup) {{
            if(h_ptr) delete[] h_ptr;
        }}
        h_buffers_cleanup.clear();

        for (FUSED_FIELD* d_ptr : d_buffers_cleanup) {{
             if (d_ptr) {{
                 cudaFree(d_ptr);
             }}
        }}
        d_buffers_cleanup.clear();

        printf("Cleanup complete.\\n");
        return 0;

    }} catch (const std::exception& e) {{
        fprintf(stderr, "Error: %s\\n", e.what());
        fprintf(stderr, "Attempting cleanup after error...\\n");
        for (FUSED_FIELD* h_ptr : h_buffers_cleanup) {{
             if(h_ptr) delete[] h_ptr;
        }}
        for (FUSED_FIELD* d_ptr : d_buffers_cleanup) {{
             if (d_ptr) cudaFree(d_ptr);
        }}
        return 1;
    }}
}}
""") # End of C++ code block

def main():
    parser = argparse.ArgumentParser(description='生成CUDA kernel的benchmark文件')
    parser.add_argument('kernel_file', help='输入的kernel文件路径(.cu 或 .cuh)')
    parser.add_argument('--size', type=int, default=1024,
                      help='array大小 (默认: 1024)')
    parser.add_argument('--output', '-o', type=str,
                      help='输出文件路径 (默认: 在输入文件同目录下生成 *_bench.cu)')

    args = parser.parse_args()

    if not os.path.exists(args.kernel_file):
        print(f"Error: 找不到输入文件: {args.kernel_file}")
        return 1
    if not os.path.isfile(args.kernel_file):
         print(f"Error: 输入路径不是一个文件: {args.kernel_file}")
         return 1

    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(args.kernel_file)
        if ext.lower() not in ['.cu', '.cuh']:
             print(f"Warning: Input file '{args.kernel_file}' does not have a .cu or .cuh extension. Output will be '{base}_bench.cu'.")
        output_file = base + '_bench.cu'


    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: 创建输出目录失败 '{output_dir}': {e}")
            return 1

    try:
        generate_benchmark(args.kernel_file, output_file, args.size)
        print(f"\nBenchmark文件已生成:")
        print(f"- 输入文件: {os.path.abspath(args.kernel_file)}")
        print(f"- 输出文件: {os.path.abspath(output_file)}")
        print(f"- Array大小: {args.size}")
    except Exception as e:
        print(f"Error: 生成benchmark文件失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())