#include "msm.cuh"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <array>
#include <chrono>
#include <cub/cub.cuh>
#include <iostream>
#include <thread>
#include "../../common/error/src/check.cuh"

#ifdef __CUDA_ARCH__
#define likely(x) (__builtin_expect((x), 1))
#define unlikely(x) (__builtin_expect((x), 0))
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif 

namespace detail {

    template <u32 windows, u32 bits_per_window>
    __host__ __device__ __forceinline__ void signed_digit(int (&r)[windows]) {
        static_assert(bits_per_window < 32, "bits_per_window must be less than 32");
        #pragma unroll
        for (u32 i = 0; i < windows - 1; i++) {
            if ((u32)r[i] >= 1u << (bits_per_window - 1)) {
                r[i] -= 1 << bits_per_window;
                r[i + 1] += 1;
            }
        }
        assert((u32)r[windows - 1] < 1u << (bits_per_window - 1));
    }

    // divide scalers into windows
    // count number of zeros in each window
    template <typename Config, typename Element>
    __global__ void distribute_windows(
        const u32 *scalers,
        const u64 len,
        u32* cnt_zero,
        u64* indexs,
        u32* points_offset
    ) {
        u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
        u32 stride = gridDim.x * blockDim.x;
        
        u32 cnt_zero_local = 0;
        // Count into block-wide counter
        for (u32 i = tid; i < len; i += stride) {
            int bucket[Config::actual_windows];
            auto scaler = Element::load(scalers + i * Element::LIMBS).to_number();
            scaler.bit_slice<Config::actual_windows, Config::s>(bucket);
            signed_digit<Config::actual_windows, Config::s>(bucket);

            #pragma unroll
            for (u32 window_id = 0; window_id < Config::actual_windows; window_id++) {
                auto sign = bucket[window_id] < 0;
                auto bucket_id = sign ? -bucket[window_id] : bucket[window_id];
                u32 physical_window_id = window_id % Config::n_windows;
                u32 point_group = window_id / Config::n_windows;
                if (bucket_id == 0) {
                    cnt_zero_local++;
                }
                u64 index = bucket_id | (sign << Config::s) | (physical_window_id << (Config::s + 1)) 
                | ((point_group * len + i) << (Config::s + 1 + Config::window_bits));
                indexs[(points_offset[physical_window_id] + point_group) * len + i] = index;
            }
        }
        atomicAdd(cnt_zero, cnt_zero_local);
    }

    __device__ __forceinline__ void lock(unsigned short *mutex_ptr, u32 wait_limit = 16) {
        u32 time = 1;
        while (atomicCAS(mutex_ptr, (unsigned short int)0, (unsigned short int)1) != 0) {
            __nanosleep(time);
            if likely(time < wait_limit) time *= 2;
        }
        __threadfence();
    }

    __device__ __forceinline__ void unlock(unsigned short *mutex_ptr) {
        __threadfence();
        atomicCAS(mutex_ptr, (unsigned short int)1, (unsigned short int)0);
    }

    template <typename Config, typename Point>
    __device__ __forceinline__ void sum_back(
        Point &acc,
        u32 window_id,
        u32 key,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> mutex,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> initialized,
        Array2D<Point, Config::n_windows, Config::n_buckets> sum
    ) {
        auto mutex_ptr = mutex.addr(window_id, key - 1);
        lock(mutex_ptr);
        if (initialized.get(window_id, key - 1)) {
            sum.get(window_id, key - 1) = sum.get(window_id, key - 1) + acc;
        } else {
            sum.get(window_id, key - 1) = acc;
            initialized.get(window_id, key - 1) = 1;
        }
        unlock(mutex_ptr);
    }

    template <typename Config, u32 WarpPerBlock, typename Point, typename PointAffine>
    __global__ void bucket_sum(
        const u64 len,
        const u32 *cnt_zero,
        const u64 *indexs,
        const u32 *points,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> mutex,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> initialized,
        Array2D<Point, Config::n_windows, Config::n_buckets> sum
    ) {
        extern __shared__ u32 smem[];
        Array2D<u32, WarpPerBlock * THREADS_PER_WARP, PointAffine::N_WORDS * 2 + 4> point_buffer(smem);
        // __shared__ u32 point_buffer[WarpPerBlock * THREADS_PER_WARP][PointAffine::N_WORDS * 2 + 4]; // +4 for padding and alignment
        const static u32 key_mask = (1u << Config::s) - 1;
        const static u32 sign_mask = 1u << Config::s;
        const static u32 window_mask = (1u << Config::window_bits) - 1;

        const u32 gtid = threadIdx.x + blockIdx.x * blockDim.x;
        const u32 threads = blockDim.x * gridDim.x;
        const u32 zero_num = *cnt_zero;

        u32 work_len = div_ceil(len - zero_num, threads);
        const u32 start_id = work_len * gtid;
        const u32 end_id = min((u64)start_id + work_len, len - zero_num);
        if (start_id >= end_id) return;

        indexs += zero_num;

        int stage = 0;
        uint4 *smem_ptr0 = reinterpret_cast<uint4*>(point_buffer.addr(threadIdx.x, 0));
        uint4 *smem_ptr1 = reinterpret_cast<uint4*>(point_buffer.addr(threadIdx.x, PointAffine::N_WORDS));

        bool first = true; // only the first bucket and the last bucket may have conflict with other threads
        auto pip_thread = cuda::make_pipeline(); // pipeline for this thread

        u64 index = indexs[start_id];
        u64 pointer = index >> (Config::s + 1 + Config::window_bits);
        u32 key = index & key_mask;
        u32 sign = (index & sign_mask) != 0;
        u32 window_id = (index >> (Config::s + 1)) & window_mask;
        
        // used to special handle the last bucket
        u64 last_index = indexs[end_id - 1];
        u32 last_key = last_index & key_mask;
        u32 last_window_id = (last_index >> (Config::s + 1)) & window_mask;

        pip_thread.producer_acquire();
        cuda::memcpy_async(smem_ptr0, reinterpret_cast<const uint4*>(points + pointer * PointAffine::N_WORDS), sizeof(PointAffine), pip_thread);
        stage ^= 1;
        pip_thread.producer_commit();

        auto acc = Point::identity();

        for (u32 i = start_id + 1; i < end_id; i++) {
            u64 next_index = indexs[i];
            pointer = next_index >> (Config::s + 1 + Config::window_bits);

            uint4 *g2s_ptr, *s2r_ptr;
            if (stage == 0) {
                g2s_ptr = smem_ptr0;
                s2r_ptr = smem_ptr1;
            } else {
                g2s_ptr = smem_ptr1;
                s2r_ptr = smem_ptr0;
            }

            pip_thread.producer_acquire();
            cuda::memcpy_async(g2s_ptr, reinterpret_cast<const uint4*>(points + pointer * PointAffine::N_WORDS), sizeof(PointAffine), pip_thread);
            pip_thread.producer_commit();
            stage ^= 1;
            
            cuda::pipeline_consumer_wait_prior<1>(pip_thread);
            auto p = PointAffine::load(reinterpret_cast<u32*>(s2r_ptr));
            pip_thread.consumer_release();

            if (sign) p = p.neg();
            acc = acc + p;

            u32 next_key = next_index & key_mask;
            u32 next_window_id = (next_index >> (Config::s + 1)) & window_mask;

            if unlikely(next_key != key || next_window_id != window_id) {
                if unlikely(first) {
                    unsigned short *mutex_ptr;
                    mutex_ptr = mutex.addr(window_id, key - 1);
                    lock(mutex_ptr);
                    if (initialized.get(window_id, key - 1)) {
                        sum.get(window_id, key - 1) = sum.get(window_id, key - 1) + acc;
                    } else {
                        sum.get(window_id, key - 1) = acc;
                        initialized.get(window_id, key - 1) = 1;
                    }
                    unlock(mutex_ptr);
                    first = false;
                } else {
                    sum.get(window_id, key - 1) = acc;
                    initialized.get(window_id, key - 1) = 1;
                }

                if (initialized.get(next_window_id, next_key - 1) && (next_key != last_key || next_window_id != last_window_id)) {
                    acc = sum.get(next_window_id, next_key - 1);
                } else {
                    acc = Point::identity();
                }
            }
            key = next_key;
            sign = (next_index & sign_mask) != 0;
            window_id = next_window_id;
        }

        pip_thread.consumer_wait();
        auto p = PointAffine::load(reinterpret_cast<u32*>(stage == 0 ? smem_ptr1 : smem_ptr0));
        pip_thread.consumer_release();

        if (sign) p = p.neg();
        acc = acc + p;
        
        // here may have conflict with other threads
        // in the case when several threads calculate the same bucket
        // we do intra warp reduction first
        // use dynamic warp reduction, if no conflict, only extra shfl is used
        // if all threads have same peer, full warp reduction is used

        u32 mask[5] = {0xFFFFFFFF, 0x55555555, 0x11111111, 0x01010101, 0x00010001};
        u32 lane_id = threadIdx.x & 31;
        bool different_peer = false;
        #pragma unroll
        for (u32 lg_delta = 0; lg_delta < 5; lg_delta++) {
            if (lg_delta != 0 && __all_sync(mask[lg_delta], different_peer)) {
                // all threads have different peer, write back directly
                sum_back<Config>(acc, window_id, key, mutex, initialized, sum);
                break;
            }
            u32 delta = 1 << lg_delta;
            u32 peer_window_id = __shfl_xor_sync(mask[lg_delta], window_id, delta);
            u32 peer_key = __shfl_xor_sync(mask[lg_delta], key, delta);
            Point peer_acc = acc.shuffle_down(delta, mask[lg_delta]);

            different_peer = window_id != peer_window_id || key != peer_key;
            if (lane_id % 2 != 0) {
                if (different_peer) {
                    // write back by myself
                    sum_back<Config>(acc, window_id, key, mutex, initialized, sum);
                }
                break;
            } else {
                if (!different_peer) {
                    // add up the peer
                    acc = acc + peer_acc;
                }
                if (lg_delta == 4) {
                    // write back by myself
                    sum_back<Config>(acc, window_id, key, mutex, initialized, sum);
                }
            }
            lane_id /= 2;
            __syncwarp();
        }
        // direct write back
        // auto mutex_ptr = mutex.addr(window_id, key - 1);
        // lock(mutex_ptr);
        // if (initialized.get(window_id, key - 1)) {
        //     sum.get(window_id, key - 1) = sum.get(window_id, key - 1) + acc;
        // } else {
        //     sum.get(window_id, key - 1) = acc;
        //     initialized.get(window_id, key - 1) = 1;
        // }
        // unlock(mutex_ptr);
    }

    template<typename Config, u32 WarpPerBlock, typename Point>
    __launch_bounds__(256,1)
    __global__ void reduceBuckets(
        Array2D<Point, Config::n_windows, Config::n_buckets> buckets_sum, 
        Point *reduceMemory,
        Array2D<unsigned short, Config::n_windows, Config::n_buckets> initialized
    ) {

        assert(gridDim.x % Config::n_windows == 0);

        __shared__ u32 smem[WarpPerBlock][Point::N_WORDS + 4]; // +4 for padding

        const u32 total_threads_per_window = gridDim.x / Config::n_windows * blockDim.x;
        u32 window_id = blockIdx.x / (gridDim.x / Config::n_windows);

        u32 wtid = (blockIdx.x % (gridDim.x / Config::n_windows)) * blockDim.x + threadIdx.x;
          
        const u32 buckets_per_thread = div_ceil(Config::n_buckets, total_threads_per_window);

        Point sum, sum_of_sums;

        sum = Point::identity();
        sum_of_sums = Point::identity();

        for(u32 i=buckets_per_thread; i > 0; i--) {
            u32 loadIndex = wtid * buckets_per_thread + i;
            if(loadIndex <= Config::n_buckets && initialized.get(window_id, loadIndex - 1)) {
                sum = sum + buckets_sum.get(window_id, loadIndex - 1);
            }
            sum_of_sums = sum_of_sums + sum;
        }

        u32 scale = wtid * buckets_per_thread;

        sum = sum.multiple(scale);

        sum_of_sums = sum_of_sums + sum;

        // Reduce within the block
        // 1. reduce in each warp
        // 2. store to smem
        // 3. reduce in warp1
        u32 warp_id = threadIdx.x / 32;
        u32 lane_id = threadIdx.x % 32;

        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(16);
        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(8);
        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(4);
        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(2);
        sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(1);

        if (lane_id == 0) {
            sum_of_sums.store(smem[warp_id]);
        }

        __syncthreads();

        if (warp_id > 0) return;

        if (threadIdx.x < WarpPerBlock) {
            sum_of_sums = Point::load(smem[threadIdx.x]);
        } else {
            sum_of_sums = Point::identity();
        }

        // Reduce in warp1
        if constexpr (WarpPerBlock > 16) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(16);
        if constexpr (WarpPerBlock > 8) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(8);
        if constexpr (WarpPerBlock > 4) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(4);
        if constexpr (WarpPerBlock > 2) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(2);
        if constexpr (WarpPerBlock > 1) sum_of_sums = sum_of_sums + sum_of_sums.shuffle_down(1);

        // Store to global memory
        if (threadIdx.x == 0) {
            for (u32 i = 0; i < window_id * Config::s; i++) {
               sum_of_sums = sum_of_sums.self_add();
            }
            reduceMemory[blockIdx.x] = sum_of_sums;
        }
    }

    template <typename Config, typename Point, typename PointAffine>
    __global__ void precompute_kernel(u32 *points, u64 len) {
        u64 gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid >= len) return;
        auto p = PointAffine::load(points + gid * PointAffine::N_WORDS).to_point();
        for (u32 i = 1; i < Config::n_precompute; i++) {
            #pragma unroll
            for (u32 j = 0; j < Config::n_windows * Config::s; j++) {
                p = p.self_add();
            }

            p.to_affine().store(points + (gid + i * len) * PointAffine::N_WORDS);
        }
    }

    template <typename Config, typename Element, typename Point, typename PointAffine>
    cudaError_t MSM<Config, Element, Point, PointAffine>::run(const u32 batches, std::vector<const u32*>::const_iterator h_scalers, bool first_run, cudaStream_t stream) {

        u64 part_len = div_ceil(len, parts);

        for (u32 i = 0; i < batches; i++) {
            CUDA_CHECK(cudaMemsetAsync(initialized_buf[i], 0, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
        }

        auto mutex = Array2D<unsigned short, Config::n_windows, Config::n_buckets>(mutex_buf);

        int begin = 0, end = parts, stride = 1;

        if (head) {
            begin = 0;
            end = parts;
            stride = 1;
            head = false;
        } else {
            begin = parts - 1;
            end = -1;
            stride = -1;
            head = true;
        }
        u32 first_len = std::min(part_len, len - begin * part_len);
        u32 points_transported = first_run ? 0 : first_len;
        
        cudaStream_t copy_stream;
        CUDA_CHECK(cudaStreamCreate(&copy_stream));
        for (u32 i = 0; i < max_scaler_stages; i++) {
            CUDA_CHECK(cudaEventRecord(begin_scaler_copy[i], stream));
        }
        for (u32 i = 0; i < max_point_stages; i++) {
            CUDA_CHECK(cudaEventRecord(begin_point_copy[i], stream));
        }
        
        u32 points_per_transfer;

        for (int p = begin; p != end; p += stride) {
            u64 offset = p * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            if (p + stride != end) {
                auto next_len = std::min(part_len, len - (p + stride) * part_len);
                points_per_transfer = div_ceil(next_len, batches);
            }

            for (int j = 0; j < batches; j++) {
                CUDA_CHECK(cudaStreamWaitEvent(copy_stream, begin_scaler_copy[stage_scaler], cudaEventWaitDefault));
                CUDA_CHECK(cudaMemcpyAsync(scalers[stage_scaler], *(h_scalers + j) + offset * Element::LIMBS, sizeof(u32) * Element::LIMBS * cur_len, cudaMemcpyHostToDevice, copy_stream));
                CUDA_CHECK(cudaEventRecord(end_scaler_copy[stage_scaler], copy_stream));

                CUDA_CHECK(cudaMemsetAsync(cnt_zero, 0, sizeof(u32), stream));
                
                CUDA_CHECK(cudaStreamWaitEvent(stream, end_scaler_copy[stage_scaler], cudaEventWaitDefault));
                cudaEvent_t start, stop;
                float elapsedTime = 0.0;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                if constexpr (Config::debug) {
                    cudaEventRecord(start, stream);
                }
                
                u32 block_size = 512;
                u32 grid_size = num_sm;
                distribute_windows<Config, Element><<<grid_size, block_size, 0, stream>>>(
                    scalers[stage_scaler],
                    cur_len,
                    cnt_zero,
                    indexs + part_len * Config::actual_windows,
                    d_points_offset
                );

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(begin_scaler_copy[stage_scaler], stream));

                if constexpr (Config::debug) {
                    cudaEventRecord(stop, stream);
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    cudaEventElapsedTime(&elapsedTime, start, stop);
                    std::cout << "MSM distribute_windows time:" << elapsedTime << std::endl;
                }

                if constexpr (Config::debug) {
                    cudaEventRecord(start, stream);
                }

                cub::DeviceRadixSort::SortKeys(
                    d_temp_storage_sort, temp_storage_bytes_sort,
                    indexs + part_len * Config::actual_windows, indexs,
                    Config::actual_windows * cur_len, 0, Config::s, stream
                );

                CUDA_CHECK(cudaGetLastError());
                if constexpr (Config::debug) {
                    cudaEventRecord(stop, stream);
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    cudaEventElapsedTime(&elapsedTime, start, stop);
                    std::cout << "MSM sort time:" << elapsedTime << std::endl;
                }

                // wait before the first point copy
                if (points_transported == 0) CUDA_CHECK(cudaStreamWaitEvent(copy_stream, begin_point_copy[stage_point_transporting], cudaEventWaitDefault));
                
                if (j == 0) {
                    u32 point_left = cur_len - points_transported;
                    if (point_left > 0) for (int i = 0; i < Config::n_precompute; i++) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            points[stage_point_transporting] + (i * cur_len + points_transported) * PointAffine::N_WORDS,
                            h_points[i] + (offset + points_transported) * PointAffine::N_WORDS,
                            sizeof(PointAffine) * point_left, cudaMemcpyHostToDevice, copy_stream
                        ));
                    }
                    CUDA_CHECK(cudaEventRecord(end_point_copy[stage_point_transporting], copy_stream));
                    points_transported = 0;
                    if (p + stride != end) stage_point_transporting = (stage_point_transporting + 1) % max_point_stages;
                } else if(p + stride != end) {
                    u64 next_offset = (p + stride) * part_len;

                    for (int i = 0; i < Config::n_precompute; i++) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            points[stage_point_transporting] + (i * cur_len + points_transported) * PointAffine::N_WORDS,
                            h_points[i] + (next_offset + points_transported) * PointAffine::N_WORDS,
                            sizeof(PointAffine) * points_per_transfer, cudaMemcpyHostToDevice, copy_stream
                        ));
                    }
                    points_transported += points_per_transfer;
                }

                if (j == 0) CUDA_CHECK(cudaStreamWaitEvent(stream, end_point_copy[stage_point], cudaEventWaitDefault));

                if constexpr (Config::debug) {
                    cudaEventRecord(start, stream);
                }

                // Do bucket sum
                block_size = 256;
                grid_size = num_sm;
                constexpr u32 warp_num = 8;

                usize shared_size = (PointAffine::N_WORDS * 2 + 4) * warp_num * THREADS_PER_WARP * sizeof(u32);

                auto sum_kernel = bucket_sum<Config, warp_num, Point, PointAffine>;

                CUDA_CHECK(cudaFuncSetAttribute(sum_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

                sum_kernel<<<grid_size, block_size, shared_size, stream>>>(
                    cur_len * Config::actual_windows,
                    cnt_zero,
                    indexs,
                    points[stage_point],
                    mutex,
                    initialized[j],
                    buckets_sum[j]
                );
                CUDA_CHECK(cudaGetLastError());

                if constexpr (Config::debug) {
                    cudaEventRecord(stop, stream);
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    cudaEventElapsedTime(&elapsedTime, start, stop);
                    std::cout << "MSM bucket sum time:" << elapsedTime << std::endl;
                }

                if (j == batches - 1) CUDA_CHECK(cudaEventRecord(begin_point_copy[stage_point], stream));

                stage_scaler = (stage_scaler + 1) % max_scaler_stages;
            }

            if (p + stride != end) {
                stage_point = (stage_point + 1) % max_point_stages;
            }
        }

        CUDA_CHECK(cudaStreamDestroy(copy_stream));

        cudaEvent_t start, stop;
        float ms;

        if constexpr (Config::debug) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }

        u32 grid = reduce_blocks; 

        // start reduce
        for (int j = 0; j < batches; j++) {
            if constexpr (Config::debug) {
                cudaEventRecord(start, stream);
            }
            
            reduceBuckets<Config, 8> <<< grid, 256, 0, stream >>> (buckets_sum[j], reduce_buffer[j], initialized[j]);

            CUDA_CHECK(cudaGetLastError());

            if constexpr (Config::debug) {
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms, start, stop);
                std::cout << "MSM bucket reduce time:" << ms << std::endl;
            }
        }

        return cudaSuccess;
    }
 
    template <typename Config, typename Element, typename Point, typename PointAffine>
    MSM<Config, Element, Point, PointAffine>::MSM(u64 len, u32 batch_per_run, u32 parts, u32 scaler_stages, u32 point_stages, int device)
    : len(len), batch_per_run(batch_per_run), parts(parts), max_scaler_stages(scaler_stages), max_point_stages(point_stages),
    stage_scaler(0), stage_point(0), stage_point_transporting(0), device(device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        num_sm = deviceProp.multiProcessorCount;
        reduce_blocks = div_ceil(num_sm, Config::n_windows) * Config::n_windows;
        d_buckets_sum_buf = new Point*[batch_per_run];
        buckets_sum = new Array2D<Point, Config::n_windows, Config::n_buckets>[batch_per_run];
        initialized_buf = new unsigned short*[batch_per_run];
        initialized = new Array2D<unsigned short, Config::n_windows, Config::n_buckets>[batch_per_run];
        scalers = new u32*[scaler_stages];
        points = new u32*[point_stages];
        begin_scaler_copy = new cudaEvent_t[scaler_stages];
        end_scaler_copy = new cudaEvent_t[scaler_stages];
        begin_point_copy = new cudaEvent_t[point_stages];
        end_point_copy = new cudaEvent_t[point_stages];
        reduce_buffer = new Point*[batch_per_run];
        h_reduce_buffer = new Point*[batch_per_run];
        
        for (int j = 0; j < batch_per_run; j++) {
            cudaMallocHost(&h_reduce_buffer[j], sizeof(Point) * reduce_blocks, cudaHostAllocDefault);
        }
        h_points_offset[0] = 0;
        for (u32 i = 0; i < Config::n_windows; i++) {
            h_points_per_window[i] = div_ceil(Config::actual_windows - i, Config::n_windows);
            h_points_offset[i + 1] = h_points_offset[i] + h_points_per_window[i];
        }
        // points_offset[Config::n_windows] should be the total number of points
        assert(h_points_offset[Config::n_windows] == Config::actual_windows);
    }

    template <typename Config, typename Element, typename Point, typename PointAffine>
    void MSM<Config, Element, Point, PointAffine>::set_points(std::array<const u32*, Config::n_precompute> host_points) {
        for (u32 i = 0; i < Config::n_precompute; i++) {
            h_points[i] = host_points[i];
        }
        points_set = true;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine>
    cudaError_t MSM<Config, Element, Point, PointAffine>::alloc_gpu(void *buffer, usize *buffer_size, cudaStream_t stream) {
        CUDA_CHECK(cudaSetDevice(device));
        u64 part_len = div_ceil(len, parts);
        usize bucket_sum_size = sizeof(Point) * Config::n_windows * Config::n_buckets;
        usize initialized_size = sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows;
        usize mutex_size = sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows;
        usize points_offset_size = sizeof(u32) * (Config::n_windows + 1);
        usize cnt_zero_size = sizeof(u32);
        usize indexs_size = sizeof(u64) * Config::actual_windows * part_len * 2;
        usize scalers_size = sizeof(u32) * Element::LIMBS * part_len;
        usize points_size = sizeof(u32) * PointAffine::N_WORDS * part_len * Config::n_precompute;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            nullptr, temp_storage_bytes_sort,
            indexs + part_len * Config::actual_windows, indexs,
            Config::actual_windows * part_len, 0, Config::s, stream
        ));
        usize reduce_buffer_size = sizeof(Point) * reduce_blocks;
        if (buffer == nullptr) {
            usize total_size = 0;
            total_size += bucket_sum_size * batch_per_run;
            total_size += initialized_size * batch_per_run;
            total_size += mutex_size;
            total_size += points_offset_size;
            total_size += cnt_zero_size;
            total_size += indexs_size;
            total_size += scalers_size * max_scaler_stages;
            total_size += points_size * max_point_stages;
            total_size += temp_storage_bytes_sort;
            total_size += reduce_buffer_size * batch_per_run;
            *buffer_size = total_size;
            return cudaSuccess;
        }
        char *ptr = (char*)buffer; // use char* for pointer arithmetic
        // for alignment, we need to allocate the Point and Element buffers first
        for (int i = 0; i < max_point_stages; i++) {
            points[i] = (u32*)ptr;
            ptr += points_size;
        }
        for (int i = 0; i < max_scaler_stages; i++) {
            scalers[i] = (u32*)ptr;
            ptr += scalers_size;
        }
        for (u32 i = 0; i < batch_per_run; i++) {
            d_buckets_sum_buf[i] = (Point*)ptr;
            ptr += bucket_sum_size;
            buckets_sum[i] = Array2D<Point, Config::n_windows, Config::n_buckets>(d_buckets_sum_buf[i]);
            reduce_buffer[i] = (Point*)ptr;
            ptr += reduce_buffer_size;
        }
        indexs = (u64*)ptr;
        ptr += indexs_size;
        for (u32 i = 0; i < batch_per_run; i++) {
            initialized_buf[i] = (unsigned short*)ptr;
            ptr += initialized_size;
            initialized[i] = Array2D<unsigned short, Config::n_windows, Config::n_buckets>(initialized_buf[i]);
        }
        mutex_buf = (unsigned short*)ptr;
        CUDA_CHECK(cudaMemsetAsync(mutex_buf, 0, sizeof(unsigned short) * (Config::n_buckets) * Config::n_windows, stream));
        ptr += mutex_size;
        d_points_offset = (u32*)ptr;
        CUDA_CHECK(cudaMemcpyAsync(d_points_offset, h_points_offset, sizeof(u32) * (Config::n_windows + 1), cudaMemcpyHostToDevice, stream));
        ptr += points_offset_size;
        cnt_zero = (u32*)ptr;
        ptr += cnt_zero_size;
        d_temp_storage_sort = ptr;
        ptr += temp_storage_bytes_sort;

        for (int i = 0; i < max_scaler_stages; i++) {
            CUDA_CHECK(cudaEventCreate(begin_scaler_copy + i));
            CUDA_CHECK(cudaEventCreate(end_scaler_copy + i));
        }
        for (int i = 0; i < max_point_stages; i++) {
            CUDA_CHECK(cudaEventCreate(begin_point_copy + i));
            CUDA_CHECK(cudaEventCreate(end_point_copy + i));
        }
        return cudaSuccess;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine>
    cudaError_t MSM<Config, Element, Point, PointAffine>::msm(const std::vector<const u32*>& h_scalers, std::vector<Point> &h_result, cudaStream_t stream) {
        // note: this function is not async, it will block until all computation is done
        // this is because the host reduce is done on the cpu
        assert(points_set);
        assert(h_scalers.size() == h_result.size());
        CUDA_CHECK(cudaSetDevice(device));

        const u32 batches = h_scalers.size();

        // TODO: use thread pool
        std::thread host_reduce_thread; // overlap host reduce with GPU computation

        auto host_reduce = [](Point **reduce_buffer, typename std::vector<Point>::iterator h_result, u32 n_reduce, u32 batches, cudaEvent_t start_reduce) {
            cudaEventSynchronize(start_reduce);
            // host timer
            std::chrono::high_resolution_clock::time_point start, end;
            if constexpr (Config::debug) {
                start = std::chrono::high_resolution_clock::now();
            }
            for (u32 j = 0; j < batches; j++, h_result++) {
                *h_result = Point::identity();
                for (u32 i = 0; i < n_reduce; i++) {
                    *h_result = *h_result + reduce_buffer[j][i];
                }
            }
            if constexpr (Config::debug) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << "MSM host reduce time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;
            }
        };

        for (u32 i = 0; i < batches; i += batch_per_run) {
            u32 cur_batch = std::min(batch_per_run, batches - i);
            cudaEvent_t start_reduce;
            CUDA_CHECK(cudaEventCreateWithFlags(&start_reduce, cudaEventBlockingSync));
            CUDA_CHECK(run(cur_batch, h_scalers.begin() + i, i == 0, stream));

            if (i > 0) host_reduce_thread.join();
            
            for (int j = 0; j < cur_batch; j++) {
                CUDA_CHECK(cudaMemcpyAsync(h_reduce_buffer[j], reduce_buffer[j], sizeof(Point) * reduce_blocks, cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaEventRecord(start_reduce, stream));

            host_reduce_thread = std::thread(host_reduce, h_reduce_buffer, h_result.begin() + i, reduce_blocks, cur_batch, start_reduce);
        }

        host_reduce_thread.join();

        CUDA_CHECK(cudaGetLastError());

        return cudaSuccess;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine>
    MSM<Config, Element, Point, PointAffine>::~MSM() {
        delete [] d_buckets_sum_buf;
        delete [] buckets_sum;
        delete [] initialized_buf;
        delete [] initialized;
        delete [] scalers;
        delete [] points;
        delete [] begin_scaler_copy;
        delete [] end_scaler_copy;
        delete [] begin_point_copy;
        delete [] end_point_copy;
        delete [] reduce_buffer;
        for (int j = 0; j < batch_per_run; j++) {
            cudaFreeHost(h_reduce_buffer[j]);
        }
        delete [] h_reduce_buffer;
    }

    template <typename Config, typename Point, typename PointAffine>
    cudaError_t MSMPrecompute<Config, Point, PointAffine>::run(u64 len, u64 part_len, std::array<u32*, Config::n_precompute> h_points, cudaStream_t stream) {
        if constexpr (Config::n_precompute == 1) {
            return cudaSuccess;
        }
        u32 *points;
        CUDA_CHECK(cudaMallocAsync(&points, sizeof(PointAffine) * part_len * Config::n_precompute, stream));
        for (int i = 0; i * part_len < len; i++) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);

            CUDA_CHECK(cudaMemcpyAsync(points, h_points[0] + offset * PointAffine::N_WORDS, sizeof(PointAffine) * part_len, cudaMemcpyHostToDevice, stream));

            u32 grid = div_ceil(cur_len, 256);
            u32 block = 256;
            precompute_kernel<Config, Point, PointAffine><<<grid, block, 0, stream>>>(points, cur_len);

            for (int j = 1; j < Config::n_precompute; j++) {
                CUDA_CHECK(cudaMemcpyAsync(h_points[j] + offset * PointAffine::N_WORDS, points + j * cur_len * PointAffine::N_WORDS, sizeof(PointAffine) * cur_len, cudaMemcpyDeviceToHost, stream));
            }
        }
        CUDA_CHECK(cudaFreeAsync(points, stream));
        return cudaSuccess;
    }

    // synchronous function
    template <typename Config, typename Point, typename PointAffine>
    cudaError_t MSMPrecompute<Config, Point, PointAffine>::precompute(u64 len, std::array<u32*, Config::n_precompute> h_points, int max_devices) {
        int devices;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        devices = std::min(devices, max_devices);
        if (Config::debug) std::cout << "Using " << devices << " devices" << std::endl;
        std::vector<cudaStream_t> streams;
        streams.resize(devices);
        u64 part_len = div_ceil(len, devices);
        for (u32 i = 0; i < devices; i++) {
            if (Config::debug) std::cout << "Precomputing on device " << i << std::endl;
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            std::array<u32*, Config::n_precompute> cur_points;
            for (int j = 0; j < Config::n_precompute; j++) {
                cur_points[j] = h_points[j] + offset * PointAffine::N_WORDS;
            }
            CUDA_CHECK(run(cur_len, std::min(cur_len, 1ul << 20), cur_points, streams[i]));
        }
        for (u32 i = 0; i < devices; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        return cudaSuccess;
    }

    // each msm will be decomposed into multiple msm instances, each instance will be run on a single GPU
    template <typename Config, typename Element, typename Point, typename PointAffine>
    MultiGPUMSM<Config, Element, Point, PointAffine>::MultiGPUMSM(u64 len, u32 batch_per_run, u32 parts, u32 scaler_stages, u32 point_stages, std::vector<u32> cards)
    : len(len), part_len(div_ceil(len, cards.size())), batch_per_run(batch_per_run), parts(parts),
    scaler_stages(scaler_stages), point_stages(point_stages), cards(cards) {
        msm_instances.reserve(cards.size());
        streams.resize(cards.size());
        threads.resize(cards.size());
        for (u32 i = 0; i < cards.size(); i++) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            msm_instances.emplace_back(cur_len, batch_per_run, parts, scaler_stages, point_stages, cards[i]);
        }
    }

    template <typename Config, typename Element, typename Point, typename PointAffine>
    cudaError_t MultiGPUMSM<Config, Element, Point, PointAffine>::alloc_gpu(void * const* buffer, usize *buffer_size) {
        for (u32 i = 0; i < cards.size(); i++) {
            CUDA_CHECK(cudaSetDevice(cards[i]));
            if (buffer == nullptr) {
                CUDA_CHECK(msm_instances[i].alloc_gpu(nullptr, buffer_size + i, 0));
            } else {
                CUDA_CHECK(cudaStreamCreate(&streams[i]));
                CUDA_CHECK(msm_instances[i].alloc_gpu(buffer[i], nullptr, streams[i]));
            }
        }
        return cudaSuccess;
    }

    template <typename Config, typename Element, typename Point, typename PointAffine>
    void MultiGPUMSM<Config, Element, Point, PointAffine>::set_points(std::array<const u32*, Config::n_precompute> host_points) {
        for (u32 i = 0; i < cards.size(); i++) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            std::array<const u32*, Config::n_precompute> cur_points;
            for (int j = 0; j < Config::n_precompute; j++) {
                cur_points[j] = host_points[j] + offset * PointAffine::N_WORDS;
            }
            msm_instances[i].set_points(cur_points);
        }
    }

    template <typename Config, typename Element, typename Point, typename PointAffine>
    cudaError_t MultiGPUMSM<Config, Element, Point, PointAffine>::msm(const std::vector<const u32*>& h_scalers, std::vector<Point> &h_result) {
        assert(h_scalers.size() == h_result.size());

        // pre-card results
        std::vector<std::vector<Point>> results(cards.size());
        for (u32 i = 0; i < cards.size(); i++) {
            results[i].resize(h_result.size());
        }
        
        auto run_msm = [](MSM<Config, Element, Point, PointAffine> &msm, const std::vector<const u32*> &h_scalers, std::vector<Point> &h_result, cudaStream_t stream) {
            msm.msm(h_scalers, h_result, stream);
        };

        for (u32 i = 0; i < cards.size(); i++) {
            u64 offset = i * part_len;
            u64 cur_len = std::min(part_len, len - offset);
            std::vector<const u32*> cur_scalers = h_scalers;
            for (u32 j = 0; j < h_scalers.size(); j++) {
                cur_scalers[j] += offset * Element::LIMBS;
            }
            threads[i] = std::thread(run_msm, std::ref(msm_instances[i]), cur_scalers, std::ref(results[i]), streams[i]);
        }

        for (u32 i = 0; i < cards.size(); i++) {
            threads[i].join();
        }

        CUDA_CHECK(cudaGetLastError());

        for (u32 i = 0; i < h_result.size(); i++) {
            h_result[i] = Point::identity();
            for (u32 j = 0; j < cards.size(); j++) {
                h_result[i] = h_result[i] + results[j][i];
            }
        }
        return cudaSuccess;
    }
}