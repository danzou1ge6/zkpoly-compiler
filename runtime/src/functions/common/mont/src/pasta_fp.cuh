#pragma once
#include "field.cuh"

namespace pasta_fp
{
    // pasta_fp
    // 28948022309329048855892746252171976963363056481941560715954676764349967630337
    // const auto params_pasta_fp = mont256::Params {
    //     .m = BIG_INTEGER_CHUNKS8(0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x224698fc, 0x094cf91b, 0x992d30ed, 0x00000001),
    //     .r_mod = BIG_INTEGER_CHUNKS8(0x3fffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x992c350b, 0xe41914ad, 0x34786d38, 0xfffffffd),
    //     .r2_mod = BIG_INTEGER_CHUNKS8(0x96d41af, 0x7b9cb714, 0x7797a99b, 0xc3c95d18, 0xd7d30dbd, 0x8b0de0e7, 0x8c78ecb3, 0x0000000f),
    //     .m_prime = 4294967295
    // };

    using Number = mont::Number<8>;
    using mont::u32;

    namespace device_constants
    {
        // m = 28948022309329048855892746252171976963363056481941560715954676764349967630337
        const __device__ Number m = BIG_INTEGER_CHUNKS8(0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x224698fc, 0x094cf91b, 0x992d30ed, 0x00000001);
        const __device__ Number m_sub2 = BIG_INTEGER_CHUNKS8(0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x224698fc, 0x094cf91b, 0x992d30ec, 0xffffffff);
        const __device__ Number r_mod = BIG_INTEGER_CHUNKS8(0x3fffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x992c350b, 0xe41914ad, 0x34786d38, 0xfffffffd);
        const __device__ Number r2_mod = BIG_INTEGER_CHUNKS8(0x96d41af, 0x7b9cb714, 0x7797a99b, 0xc3c95d18, 0xd7d30dbd, 0x8b0de0e7, 0x8c78ecb3, 0x0000000f);
    }

    namespace host_constants
    {
        const Number m = BIG_INTEGER_CHUNKS8(0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x224698fc, 0x094cf91b, 0x992d30ed, 0x00000001);
        const Number m_sub2 = BIG_INTEGER_CHUNKS8(0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x224698fc, 0x094cf91b, 0x992d30ec, 0xffffffff);
        const Number r_mod = BIG_INTEGER_CHUNKS8(0x3fffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x992c350b, 0xe41914ad, 0x34786d38, 0xfffffffd);
        const Number r2_mod = BIG_INTEGER_CHUNKS8(0x96d41af, 0x7b9cb714, 0x7797a99b, 0xc3c95d18, 0xd7d30dbd, 0x8b0de0e7, 0x8c78ecb3, 0x0000000f);
    }

    struct Params
    {
        static const mont::usize LIMBS = 8;
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
        static const u32 m_prime = 4294967295;
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