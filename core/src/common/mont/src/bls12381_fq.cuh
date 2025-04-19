#pragma once
#include "field.cuh"

// base field for BLS12-381
namespace bls12381_fq
{
    // bls12381_fq
    // 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
    // const auto params = Params {
    //     .m = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 0xffffaaab),
    //     .mm2 = BIG_INTEGER_CHUNKS12(0x340223d4, 0x72ffcd34, 0x96374f6c, 0x869759ae, 0xc8ee9709, 0xe70a257e, 0xce61a541, 0xed61ec48, 0x3d57fffd, 0x62a7ffff, 0x73fdffff, 0xffff5556),
    //     .r_mod = BIG_INTEGER_CHUNKS12(0x15f65ec3, 0xfa80e493, 0x5c071a97, 0xa256ec6d, 0x77ce5853, 0x70525745, 0x5f489857, 0x53c758ba, 0xebf4000b, 0xc40c0002, 0x76090000, 0x0002fffd),
    //     .r2_mod = BIG_INTEGER_CHUNKS12(0x11988fe5, 0x92cae3aa, 0x9a793e85, 0xb519952d, 0x67eb88a9, 0x939d83c0, 0x8de5476c, 0x4c95b6d5, 0x0a76e6a6, 0x09d104f1, 0xf4df1f34, 0x1c341746),
    //     .m_prime = 4294770685
    // };

    using Number = mont::Number<12>;
    using mont::u32;

    namespace device_constants
    {
        // m = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
        const __device__ Number m = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 0xffffaaab);
        const __device__ Number m_sub2 = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 4294945449);
        const __device__ Number r_mod = BIG_INTEGER_CHUNKS12(0x15f65ec3, 0xfa80e493, 0x5c071a97, 0xa256ec6d, 0x77ce5853, 0x70525745, 0x5f489857, 0x53c758ba, 0xebf4000b, 0xc40c0002, 0x76090000, 0x0002fffd);
        const __device__ Number r2_mod = BIG_INTEGER_CHUNKS12(0x11988fe5, 0x92cae3aa, 0x9a793e85, 0xb519952d, 0x67eb88a9, 0x939d83c0, 0x8de5476c, 0x4c95b6d5, 0x0a76e6a6, 0x09d104f1, 0xf4df1f34, 0x1c341746);
    }

    namespace host_constants
    {
        const Number m = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 0xffffaaab);
        const Number m_sub2 = BIG_INTEGER_CHUNKS12(0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 4294945449);
        const Number r_mod = BIG_INTEGER_CHUNKS12(0x15f65ec3, 0xfa80e493, 0x5c071a97, 0xa256ec6d, 0x77ce5853, 0x70525745, 0x5f489857, 0x53c758ba, 0xebf4000b, 0xc40c0002, 0x76090000, 0x0002fffd);
        const Number r2_mod = BIG_INTEGER_CHUNKS12(0x11988fe5, 0x92cae3aa, 0x9a793e85, 0xb519952d, 0x67eb88a9, 0x939d83c0, 0x8de5476c, 0x4c95b6d5, 0x0a76e6a6, 0x09d104f1, 0xf4df1f34, 0x1c341746);
    }

    struct Params
    {
        static const mont::usize LIMBS = 12;
        static const __host__ __device__ __forceinline__ Number m()
        {
    #ifdef __CUDA_ARCH__
        return device_constants::m;
    #else
        return host_constants::m;
    #endif
        }
        // m - 2
        static const __host__ __device__ __forceinline__ Number m_sub2()
        {
    #ifdef __CUDA_ARCH__
        return device_constants::m_sub2;
    #else
        return host_constants::m_sub2;
    #endif
        }
        // m' = -m^(-1) mod b where b = 2^32
        static const u32 m_prime = 4294770685;
        // r_mod = R mod m,
        static const __host__ __device__ __forceinline__ Number r_mod()
        {
    #ifdef __CUDA_ARCH__
        return device_constants::r_mod;
    #else
        return host_constants::r_mod;
    #endif
        }
        // r2_mod = R^2 mod m
        static const __host__ __device__ __forceinline__ Number r2_mod()
        {

    #ifdef __CUDA_ARCH__
        return device_constants::r2_mod;
    #else
        return host_constants::r2_mod;
    #endif
        }
    };

    using Element = mont::Element<Params>;
}