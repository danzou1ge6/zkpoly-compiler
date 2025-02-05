#pragma once
#include "../../common/mont/src/field.cuh"
#include <array>
#include <thread>

namespace detail {

    using mont::u32;
    using mont::u64;
    using mont::usize;

    const u32 THREADS_PER_WARP = 32;

    constexpr __forceinline__ u32 pow2(u32 n) {
        return n == 0 ? 1 : 2 * pow2(n - 1);
    }

    constexpr __forceinline__ __host__ __device__ u32 div_ceil(u32 a, u32 b) {
        return (a + b - 1) / b;
    }

    constexpr __forceinline__ int log2_floor(int n) {
        return (n == 1) ? 0 : 1 + log2_floor(n / 2);
    }

    constexpr __forceinline__ int log2_ceil(int n) {
        // Check if n is a power of 2
        if ((n & (n - 1)) == 0)
            return log2_floor(n);
        else
            return 1 + log2_floor(n);
    }

    template <typename T, u32 D1, u32 D2>
    struct Array2D {
        T *buf;

        __forceinline__ Array2D() {}
        __forceinline__ Array2D(T *buf) : buf(buf) {}
        __host__ __device__ __forceinline__ T &get(u32 i, u32 j) {
            return buf[i * D2 + j];
        }
        __host__ __device__ __forceinline__ const T &get_const(u32 i, u32 j) const {
            return buf[i * D2 + j];
        }
        __host__ __device__ __forceinline__ T *addr(u32 i, u32 j) {
            return buf + i * D2 + j;
        }
    };

    template <u32 BITS = 255, u32 WINDOW_SIZE = 22,
    u32 TARGET_WINDOWS = 1, bool DEBUG = true>
    struct MsmConfig {
        static constexpr u32 lambda = BITS;
        static constexpr u32 s = WINDOW_SIZE; // must <= 31
        static constexpr u32 n_buckets = pow2(s - 1); // [1, 2^{s-1}] buckets, using signed digit to half the number of buckets

        // # of logical windows
        static constexpr u32 actual_windows = div_ceil(lambda, s);
        
        // stride for precomputation(same as # of windows), if >= actual_windows, no precomputation; if = 1, precompute all
        static constexpr u32 n_windows = actual_windows < TARGET_WINDOWS ? actual_windows : TARGET_WINDOWS;

        static constexpr u32 window_bits = log2_ceil(n_windows);
        
        // lines of points to be stored in memory, 1 for no precomputation
        static constexpr u32 n_precompute = div_ceil(actual_windows, n_windows);

        static constexpr bool debug = DEBUG;
    };

    template <typename Config, typename Element, typename Point, typename PointAffine>
    class MSM {
        std::array<const u32*, Config::n_precompute> h_points;
        Point **d_buckets_sum_buf;
        Array2D<Point, Config::n_windows, Config::n_buckets> *buckets_sum;
        unsigned short *mutex_buf;
        unsigned short **initialized_buf;
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> *initialized;
        // record for number of logical windows in each actual window
        u32 h_points_per_window[Config::n_windows];
        u32 h_points_offset[Config::n_windows + 1];
        u32 *d_points_offset;
        u32 *cnt_zero;
        u64 *indexs;
        // indexs is used to store the bucket id, point index and sign
        // Config::s bits for bucket id, 1 bit for sign, Config::window_bits for window id, the rest for point index
        // max log(2^30(max points) * precompute) + Config::s bits are needed
        static_assert(log2_ceil(Config::n_precompute) + 30 + Config::window_bits + Config::s + 1 <= 64, "Index too large");
        // for sorting, after sort, points with same bucket id are gathered, gives pointer to original index
        u32 **scalers;
        void *d_temp_storage_sort;
        usize temp_storage_bytes_sort;
        u32 **points;
        cudaEvent_t *begin_scaler_copy, *end_scaler_copy;
        cudaEvent_t *begin_point_copy, *end_point_copy;
        Point **reduce_buffer;
        Point **h_reduce_buffer;
        u32 num_sm, reduce_blocks;
        u32 stage_scaler, stage_point, stage_point_transporting;
        const u32 batch_per_run;
        const u32 parts;
        const u32 max_scaler_stages;
        const u32 max_point_stages;
        const int device;
        const u64 len;
        bool head = true;
        bool points_set = false;

        cudaError_t run(const u32 batches, std::vector<const u32*>::const_iterator h_scalers, bool first_run, cudaStream_t stream);
 
        public:

        MSM(u64 len, u32 batch_per_run, u32 parts, u32 scaler_stages, u32 point_stages, int device);

        void set_points(std::array<const u32*, Config::n_precompute> host_points);

        cudaError_t alloc_gpu(void *buffer, usize *buffer_size, cudaStream_t stream);

        cudaError_t msm(const std::vector<const u32*>& h_scalers, std::vector<Point> &h_result, cudaStream_t stream);

        ~MSM();
    };

    template <typename Config, typename Point, typename PointAffine>
    class MSMPrecompute {
        static cudaError_t run(u64 len, u64 part_len, std::array<u32*, Config::n_precompute> h_points, cudaStream_t stream = 0);
        public:
        // synchronous function
        static cudaError_t precompute(u64 len, std::array<u32*, Config::n_precompute> h_points, int max_devices = 1);
    };

     // each msm will be decomposed into multiple msm instances, each instance will be run on a single GPU
    template <typename Config, typename Element, typename Point, typename PointAffine>
    class MultiGPUMSM {
        u32 parts, scaler_stages, point_stages, batch_per_run;
        u64 len, part_len;
        const std::vector<u32> cards;
        std::vector<MSM<Config, Element, Point, PointAffine>> msm_instances;
        std::vector<cudaStream_t> streams;
        // TODO: use thread pool
        std::vector<std::thread> threads;
        public:
        MultiGPUMSM(u64 len, u32 batch_per_run, u32 parts, u32 scaler_stages, u32 point_stages, std::vector<u32> cards);

        cudaError_t alloc_gpu(void *const*buffer, usize *buffer_size);

        void set_points(std::array<const u32*, Config::n_precompute> host_points);

        cudaError_t msm(const std::vector<const u32*>& h_scalers, std::vector<Point> &h_result);
    };
}