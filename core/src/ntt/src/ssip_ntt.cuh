#pragma once
#include "common.cuh"

namespace detail {

template <typename Field>
__global__ void ssip_ntt_stage1_warp_no_twiddle (SliceIterator<Field> x, u32 log_len, u32 log_stride, u32 deg, u32 group_sz, const u32 * roots) {
    constexpr usize WORDS = Field::LIMBS;
    static_assert(WORDS % 4 == 0);
    constexpr u32 io_group = 1 << (log2_int(WORDS - 1) - 1);
    extern __shared__ u32 s[];

    const u32 lid = threadIdx.x & (group_sz - 1);
    const u32 lsize = group_sz;
    const u32 group_id = threadIdx.x / group_sz;
    const u32 group_num = blockDim.x / group_sz;
    const u32 index = blockIdx.x * group_num + group_id;

    auto u = s + group_id * ((1 << deg) + 1) * WORDS;

    const u32 lgp = log_stride - deg + 1;
    const u32 end_stride = 1 << lgp; //stride of the last butterfly

    // each segment is independent
    const u32 segment_start = (index >> lgp) << (lgp + deg);
    const u32 segment_id = index & (end_stride - 1);
    
    const u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round

    x += (segment_start + segment_id); // use u64 to avoid overflow

    const u32 io_id = lid & (io_group - 1);
    const u32 lid_start = lid - io_id;
    const u32 shared_read_stride = (lsize << 1) + 1;
    const u32 cur_io_group = io_group < lsize ? io_group : lsize;
    const u32 io_per_thread = io_group / cur_io_group;

    int io_st, io_ed, io_stride;

    io_st = lid_start;
    io_ed = lid_start + cur_io_group;
    io_stride = 1;

    // Read data
    for (int i = io_st; i != io_ed; i += io_stride) {
        for (u32 j = 0; j < io_per_thread; j++) {
            u32 io = io_id + j * cur_io_group;
            if (io * 4 < WORDS) {
                u32 group_id = i & (subblock_sz - 1);
                u64 gpos = group_id << (lgp);
                uint4 a, b;
                a = reinterpret_cast<uint4*> (&x[gpos])[io];
                b = reinterpret_cast<uint4*> (&x[(gpos+ (end_stride << (deg - 1)))])[io];

                u[(i) + (0 + io * 4) * shared_read_stride] = a.x;
                u[(i) + (1 << (deg - 1)) + (0 + io * 4) * shared_read_stride] = b.x;
                u[(i) + (1 + io * 4) * shared_read_stride] = a.y;
                u[(i) + (1 << (deg - 1)) + (1 + io * 4) * shared_read_stride] = b.y;
                u[(i) + (2 + io * 4) * shared_read_stride] = a.z;
                u[(i) + (1 << (deg - 1)) + (2 + io * 4) * shared_read_stride] = b.z;
                u[(i) + (3 + io * 4) * shared_read_stride] = a.w;
                u[(i) + (1 << (deg - 1)) + (3 + io * 4) * shared_read_stride] = b.w;
            }
        }
    }

    __syncthreads();

    const u32 pqshift = log_len - 1 - log_stride;

    for(u32 rnd = 0; rnd < deg; rnd += 6) {
        u32 sub_deg = min(6, deg - rnd);
        u32 warp_sz = 1 << (sub_deg - 1);
        u32 warp_id = lid / warp_sz;
        
        u32 lgp = deg - rnd - sub_deg;
        u32 end_stride_warp = 1 << lgp;

        u32 segment_start_warp = (warp_id >> lgp) << (lgp + sub_deg);
        u32 segment_id_warp = warp_id & (end_stride_warp - 1);
        
        u32 laneid = lid & (warp_sz - 1);

        u32 bit = subblock_sz >> rnd;
        u32 i0 = segment_start_warp + segment_id_warp + laneid * end_stride_warp;
        u32 i1 = i0 + bit;

        auto a = Field::load(u + i0, shared_read_stride);
        auto b = Field::load(u + i1, shared_read_stride);

        for (u32 i = 0; i < sub_deg; i++) {
            if (i != 0) {
                u32 lanemask = 1 << (sub_deg - i - 1);
                Field tmp;
                tmp = ((lid / lanemask) & 1) ? a : b;

                #pragma unroll
                for (u32 j = 0; j < WORDS; j++) {
                    tmp.n.limbs[j] = __shfl_xor_sync(0xffffffff, tmp.n.limbs[j], lanemask);
                }

                if ((lid / lanemask) & 1) a = tmp;
                else b = tmp;
            }

            auto tmp = a;
            a = a + b;
            b = tmp - b;
            u32 bit = (1 << sub_deg) >> (i + 1);
            u64 di = ((lid & (bit - 1)) * end_stride_warp + segment_id_warp) * end_stride + segment_id;

            if (di != 0) {
                auto w = Field::load(roots + (di << (rnd + i) << pqshift) * WORDS);
                b = b * w;
            }
        }            

        i0 = segment_start_warp + segment_id_warp + laneid * 2 * end_stride_warp;
        i1 = i0 + end_stride_warp;
        a.store(u + i0, shared_read_stride);
        b.store(u + i1, shared_read_stride);

        __syncthreads();
    }

    // Write back
    for (int i = io_st; i != io_ed; i += io_stride) {
        for (u32 j = 0; j < io_per_thread; j++) {
            u32 io = io_id + j * cur_io_group;
            if (io * 4 < WORDS) {
                u32 group_id = i & (subblock_sz - 1);
                u64 gpos = group_id << (lgp + 1);
                uint4 a = make_uint4(u[(i << 1) + (0 + io * 4) * shared_read_stride], u[(i << 1) + (1 + io * 4) * shared_read_stride], u[(i << 1) + (2 + io * 4) * shared_read_stride], u[(i << 1) + (3 + io * 4) * shared_read_stride]);
                uint4 b = make_uint4(u[(i << 1) + 1 + (0 + io * 4) * shared_read_stride], u[(i << 1) + 1 + (1 + io * 4) * shared_read_stride], u[(i << 1) + 1 + (2 + io * 4) * shared_read_stride], u[(i << 1) + 1 + (3 + io * 4) * shared_read_stride]);
                reinterpret_cast<uint4*> (&x[gpos])[io] = a;
                reinterpret_cast<uint4*> (&x[(gpos + end_stride)])[io] = b;
            }
        }
    }
}

template <typename Field>
__global__ void ssip_ntt_stage2_warp_no_share_no_twiddle (SliceIterator<Field> data, u32 log_len, u32 log_stride, u32 deg, u32 group_sz, const u32 * roots) {
    const static usize WORDS = Field::LIMBS;
    static_assert(WORDS % 4 == 0);
    constexpr u32 io_group = 1 << (log2_int(WORDS - 1) - 1);

    using barrier = cuda::barrier<cuda::thread_scope_block>;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__  barrier bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x); // Initialize the barrier with expected arrival count
    }
    __syncthreads();

    const int warp_id = static_cast<int>(threadIdx.x) / io_group;
    uint4 thread_data[io_group];

    // Specialize WarpExchange for a virtual warp of a threads owning b integer items each
    using WarpExchangeT = cub::WarpExchange<uint4, io_group, io_group>;

    // Allocate shared memory for WarpExchange
    extern __shared__ typename WarpExchangeT::TempStorage temp_storage_uint4[];

    const u32 lid = threadIdx.x & (group_sz - 1);
    const u32 lsize = group_sz;
    const u32 group_id = threadIdx.x / group_sz;
    const u32 group_num = blockDim.x / group_sz;
    const u32 index = blockIdx.x * group_num + group_id;

    u32 log_end_stride = (log_stride - deg + 1);
    u32 end_stride = 1 << log_end_stride; //stride of the last butterfly
    u32 end_pair_stride = 1 << (log_len - log_stride - 2 + deg); // the stride between the last pair of butterfly

    // each segment is independent
    // uint segment_stride = end_pair_stride << 1; // the distance between two segment
    u32 log_segment_num = (log_len - log_stride - 1 - deg); // log of # of blocks in a segment
        
    u32 segment_start = (index >> log_segment_num) << (log_segment_num + (deg << 1)); // segment_start = index / segment_num * segment_stride;

    u32 segment_id = index & ((1 << log_segment_num) - 1); // segment_id = index & (segment_num - 1);
        
    u32 subblock_sz = 1 << (deg - 1); // # of neighbouring butterfly in the last round
    u32 subblock_offset = (segment_id >> log_end_stride) << (deg + log_end_stride); // subblock_offset = (segment_id / (end_stride)) * (2 * subblock_sz * end_stride);
    u32 subblock_id = segment_id & (end_stride - 1);

    data += ((segment_start + subblock_offset + subblock_id));

    const u32 io_id = lid & (io_group - 1);
    const u32 lid_start = lid - io_id;
    const u32 cur_io_group = io_group < lsize ? io_group : lsize;

    Field a, b, c, d;

    // Read data
    if (cur_io_group == io_group) {
        #pragma unroll
        for (int tti = 0; tti < io_group; tti ++) {
            int i = tti + lid_start;
            if (io_id * 4 < WORDS) {
                u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                u32 group_id = i & (subblock_sz - 1);
                u64 gpos = group_offset + (group_id << (log_end_stride));

                thread_data[tti] = reinterpret_cast<uint4*>(&data[gpos])[io_id];
            }
        }
        WarpExchangeT(temp_storage_uint4[warp_id]).StripedToBlocked(thread_data, thread_data);
        a = Field::load(reinterpret_cast<u32*>(thread_data));
        __syncwarp();

        #pragma unroll
        for (int tti = 0; tti < io_group; tti ++) { 
            int i = tti + lid_start;
            if (io_id * 4 < WORDS) {
                u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                u32 group_id = i & (subblock_sz - 1);
                u64 gpos = group_offset + (group_id << (log_end_stride));

                thread_data[tti] = reinterpret_cast<uint4*>(&data[(gpos+ (end_stride << (deg - 1)))])[io_id];
            }
        }

        WarpExchangeT(temp_storage_uint4[warp_id]).StripedToBlocked(thread_data, thread_data);
        b = Field::load(reinterpret_cast<u32*>(thread_data));
        __syncwarp();

        #pragma unroll 
        for (int tti = 0; tti < io_group; tti ++) { 
            int i = tti + lid_start;
            if (io_id * 4 < WORDS) {
                u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                u32 group_id = i & (subblock_sz - 1);
                u64 gpos = group_offset + (group_id << (log_end_stride));

                thread_data[tti] = reinterpret_cast<uint4*>(&data[(gpos+ end_pair_stride)])[io_id];
            }
        }

        WarpExchangeT(temp_storage_uint4[warp_id]).StripedToBlocked(thread_data, thread_data);
        c = Field::load(reinterpret_cast<u32*>(thread_data));
        __syncwarp();

        #pragma unroll 
        for (int tti = 0; tti < io_group; tti ++) { 
            int i = tti + lid_start;
            if (io_id * 4 < WORDS) {
                u32 group_offset = (i >> (deg - 1)) << (log_len - log_stride - 1);
                u32 group_id = i & (subblock_sz - 1);
                u64 gpos = group_offset + (group_id << (log_end_stride));

                thread_data[tti] = reinterpret_cast<uint4*>(&data[(gpos + (end_stride << (deg - 1)) + end_pair_stride)])[io_id];
            }
        }

        WarpExchangeT(temp_storage_uint4[warp_id]).StripedToBlocked(thread_data, thread_data);
        d = Field::load(reinterpret_cast<u32*>(thread_data));
        __syncwarp();
    } else {
        u32 group_offset = (lid >> (deg - 1)) << (log_len - log_stride - 1);
        u32 group_id = lid & (subblock_sz - 1);
        u64 gpos = group_offset + (group_id << (log_end_stride));

        a = data[(gpos)];
        b = data[(gpos + (end_stride << (deg - 1)))];
        c = data[(gpos + end_pair_stride)];
        d = data[(gpos + (end_stride << (deg - 1)) + end_pair_stride)];
    }
    
    barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */

    const u32 pqshift = log_len - 1 - log_stride;

    for (u32 i = 0; i < deg; i++) {
        if (i != 0) {
            u32 lanemask = 1 << (deg - i - 1);
            Field tmp, tmp1;
            tmp = ((lid / lanemask) & 1) ? a : b;
            tmp1 = ((lid / lanemask) & 1) ? c : d;

            #pragma unroll
            for (u32 j = 0; j < WORDS; j++) {
                tmp.n.limbs[j] = __shfl_xor_sync(0xffffffff, tmp.n.limbs[j], lanemask);
                    tmp1.n.limbs[j] = __shfl_xor_sync(0xffffffff, tmp1.n.limbs[j], lanemask);
                }

                if ((lid / lanemask) & 1) a = tmp, c = tmp1;
                else b = tmp, d = tmp1;
        }

        auto tmp1 = a;
        auto tmp2 = c;

        a = a + b;
        c = c + d;
        b = tmp1 - b;
        d = tmp2 - d;

        u32 bit = subblock_sz >> i;
        u64 di = (lid & (bit - 1)) * end_stride + subblock_id;

        if (di != 0) {
            auto w = Field::load(roots + (di << i << pqshift) * WORDS);
            b = b * w;
            d = d * w;
        }
    }
        
    bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/

    // Write back
    if (cur_io_group == io_group) {
        a.store(reinterpret_cast<u32*>(thread_data));
        WarpExchangeT(temp_storage_uint4[warp_id]).BlockedToStriped(thread_data, thread_data);
        __syncwarp();

        #pragma unroll 
        for (int tti = 0; tti < io_group; tti ++) { 
            int ti = tti + lid_start;
            u32 p;
            u32 second_half_l, gap;
            u32 lid_l;
            u32 group_offset, group_id;
            u64 gpos;

            p = __brev((ti << 1)) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            if (io_id * 4  < WORDS) {
                reinterpret_cast<uint4*>(&data[(gpos + second_half_l * end_pair_stride + gap * end_stride)])[io_id] = thread_data[tti];
            }
        }

        b.store(reinterpret_cast<u32*>(thread_data));
        WarpExchangeT(temp_storage_uint4[warp_id]).BlockedToStriped(thread_data, thread_data);
        __syncwarp();

        #pragma unroll 
        for (int tti = 0; tti < io_group; tti ++) { 
            int ti = tti + lid_start;
            u32 p;
            u32 second_half_l, gap;
            u32 lid_l;
            u32 group_offset, group_id;
            u64 gpos;

            p = __brev((ti << 1) + 1) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            if (io_id * 4 < WORDS) {
                reinterpret_cast<uint4*>(&data[(gpos + second_half_l * end_pair_stride + gap * end_stride)])[io_id] = thread_data[tti];
            }
        }

        c.store(reinterpret_cast<u32*>(thread_data));
        WarpExchangeT(temp_storage_uint4[warp_id]).BlockedToStriped(thread_data, thread_data);
        __syncwarp();

        #pragma unroll 
        for (int tti = 0; tti < io_group; tti ++) { 
            int ti = tti + lid_start;
            u32 p;
            u32 second_half_l, gap;
            u32 lid_l;
            u32 group_offset, group_id;
            u64 gpos;

            p = __brev((ti << 1) + lsize * 2) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            if (io_id * 4 < WORDS) {
                reinterpret_cast<uint4*>(&data[(gpos + second_half_l * end_pair_stride + gap * end_stride)])[io_id] = thread_data[tti];
            }
        }

        d.store(reinterpret_cast<u32*>(thread_data));
        WarpExchangeT(temp_storage_uint4[warp_id]).BlockedToStriped(thread_data, thread_data);

        #pragma unroll 
        for (int tti = 0; tti < io_group; tti ++) { 
            int ti = tti + lid_start;
            u32 p;
            u32 second_half_l, gap;
            u32 lid_l;
            u32 group_offset, group_id;
            u64 gpos;

            p = __brev((ti << 1) + 1 + lsize * 2) >> (32 - (deg << 1));
            second_half_l = (p >= lsize * 2);

            lid_l = (p - second_half_l * lsize * 2);
            gap = lid_l & 1;
            lid_l = lid_l >> 1;
            

            group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
            group_id = lid_l & (subblock_sz - 1);
            gpos = group_offset + (group_id << (log_end_stride + 1));
            
            if (io_id * 4 < WORDS) {
                reinterpret_cast<uint4*>(&data[(gpos + second_half_l * end_pair_stride + gap * end_stride)])[io_id] = thread_data[tti];
            }
        }

    } else {
        
        u32 p;
        u32 second_half_l, gap;
        u32 lid_l;
        u32 group_offset, group_id;
        u64 gpos;

        p = __brev((lid << 1)) >> (32 - (deg << 1));
        second_half_l = (p >= lsize * 2);

        lid_l = (p - second_half_l * lsize * 2);
        gap = lid_l & 1;
        lid_l = lid_l >> 1;
        

        group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
        group_id = lid_l & (subblock_sz - 1);
        gpos = group_offset + (group_id << (log_end_stride + 1));
        
        data[(gpos + second_half_l * end_pair_stride + gap * end_stride)] = a;

        p = __brev((lid << 1) + 1) >> (32 - (deg << 1));
        second_half_l = (p >= lsize * 2);

        lid_l = (p - second_half_l * lsize * 2);
        gap = lid_l & 1;
        lid_l = lid_l >> 1;
        

        group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
        group_id = lid_l & (subblock_sz - 1);
        gpos = group_offset + (group_id << (log_end_stride + 1));
        
        data[(gpos + second_half_l * end_pair_stride + gap * end_stride)] = b;

        p = __brev((lid << 1) + lsize * 2) >> (32 - (deg << 1));
        second_half_l = (p >= lsize * 2);

        lid_l = (p - second_half_l * lsize * 2);
        gap = lid_l & 1;
        lid_l = lid_l >> 1;
        

        group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
        group_id = lid_l & (subblock_sz - 1);
        gpos = group_offset + (group_id << (log_end_stride + 1));
        
        data[(gpos + second_half_l * end_pair_stride + gap * end_stride)] = c;

        p = __brev((lid << 1) + lsize * 2 + 1) >> (32 - (deg << 1));
        second_half_l = (p >= lsize * 2);

        lid_l = (p - second_half_l * lsize * 2);
        gap = lid_l & 1;
        lid_l = lid_l >> 1;
        

        group_offset = (lid_l >> (deg - 1)) << (log_len - log_stride - 1);
        group_id = lid_l & (subblock_sz - 1);
        gpos = group_offset + (group_id << (log_end_stride + 1));
        
        data[(gpos + second_half_l * end_pair_stride + gap * end_stride)] = d;
    }
}

template <typename Field>
cudaError_t ssip_ntt(PolyPtr x, const u32 *twiddle, u32 log_len, cudaStream_t stream, const u32 max_threads_stage1_log, const u32 max_threads_stage2_log) {
    static const usize WORDS = Field::LIMBS;
    constexpr u32 io_group = 1 << (log2_int(WORDS - 1) - 1);

    if (log_len == 0) return cudaSuccess;

    u64 len = 1 << log_len;
    assert(len == x.len);

    auto x_iter = make_slice_iter<Field>(x);

    // plan partition for NTT stages
    u32 total_deg_stage1 = (log_len + 1) / 2;
    u32 total_deg_stage2 = log_len / 2;

    u32 max_deg_stage1 = max_threads_stage1_log + 1;
    u32 max_deg_stage2 = (max_threads_stage2_log + 2) / 2; // 4 elements per thread

    u32 deg_stage1 = get_deg(total_deg_stage1, max_deg_stage1);
    u32 deg_stage2 = get_deg(total_deg_stage2, max_deg_stage2); 

    int log_stride = log_len - 1;
    while (log_stride >= (int)log_len / 2) {
        u32 deg = std::min(deg_stage1, (log_stride + 1 - log_len / 2));

        u32 group_num = std::min((int)(len / (1 << deg)), 1 << (max_threads_stage1_log - (deg - 1)));
        u32 block_sz = (1 << (deg - 1)) * group_num;

        assert(block_sz <= (1 << max_threads_stage1_log));
        u32 block_num = len / 2 / block_sz;
        assert(block_num * 2 * block_sz == len);
        
        u32 shared_size = (sizeof(u32) * ((1 << deg) + 1) * WORDS) * group_num;
        auto kernel = ssip_ntt_stage1_warp_no_twiddle<Field>;
        // CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));

        kernel <<< block_num, block_sz, shared_size, stream >>>(x_iter, log_len, log_stride, deg, 1 << (deg - 1), twiddle);
        CUDA_CHECK(cudaGetLastError());

        log_stride -= deg;
    }
    assert (log_stride == log_len / 2 - 1);
    while (log_stride >= 0) {
        u32 deg = std::min((int)deg_stage2, log_stride + 1);
        
        u32 group_num = std::min((int)(len / (1 << (deg << 1))), 1 << (max_threads_stage2_log - 2 * (deg - 1)));
        u32 block_sz = (1 << ((deg - 1) << 1)) * group_num;
        assert(block_sz <= (1 << max_threads_stage2_log));
        u32 block_num = len / 4 / block_sz;
        assert(block_num * 4 * block_sz == len);
        usize shared_size = (sizeof(typename cub::WarpExchange<uint4, io_group, io_group>::TempStorage) * (block_sz / io_group)); 

        auto kernel = ssip_ntt_stage2_warp_no_share_no_twiddle<Field>;
        kernel <<< block_num, block_sz, shared_size, stream >>>(x_iter, log_len, log_stride, deg, ((1 << (deg << 1)) >> 2), twiddle);
        CUDA_CHECK(cudaGetLastError());

        log_stride -= deg;
    }
    return cudaSuccess;
}

} // namespace detail