#pragma once
#include "field.cuh"

namespace bls12381_fr
{
    // bls12381_fr
    // 52435875175126190479447740508185965837690552500527637822603658699938581184513
    // const auto params = mont256::Params {
    //     .m = BIG_INTEGER_CHUNKS8(0x73eda753, 0x299d7d48, 0x3339d808, 0x09a1d805, 0x53bda402, 0xfffe5bfe, 0xffffffff, 0x00000001),
    //     .r_mod = BIG_INTEGER_CHUNKS8(0x1824b159, 0xacc5056f, 0x998c4fef, 0xecbc4ff5, 0x5884b7fa, 0x00034802, 0x00000001, 0xfffffffe),
    //     .r2_mod = BIG_INTEGER_CHUNKS8(0x748d9d9, 0x9f59ff11, 0x05d31496, 0x7254398f, 0x2b6cedcb, 0x87925c23, 0xc999e990, 0xf3f29c6d),
    //     .m_prime = 4294967295
    // };

    using Number = mont::Number<8>;
    using mont::u32;

    namespace device_constants
    {
        // m = 52435875175126190479447740508185965837690552500527637822603658699938581184513
        const __device__ Number m = BIG_INTEGER_CHUNKS8(0x73eda753, 0x299d7d48, 0x3339d808, 0x09a1d805, 0x53bda402, 0xfffe5bfe, 0xffffffff, 0x00000001);
        const __device__ Number m_sub2 = BIG_INTEGER_CHUNKS8(0x73eda753, 0x299d7d48, 0x3339d808, 0x09a1d805, 0x53bda402, 0xfffe5bfe, 0xfffffffe, 0xffffffff);
        const __device__ Number r_mod = BIG_INTEGER_CHUNKS8(0x1824b159, 0xacc5056f, 0x998c4fef, 0xecbc4ff5, 0x5884b7fa, 0x00034802, 0x00000001, 0xfffffffe);
        const __device__ Number r2_mod = BIG_INTEGER_CHUNKS8(0x748d9d9, 0x9f59ff11, 0x05d31496, 0x7254398f, 0x2b6cedcb, 0x87925c23, 0xc999e990, 0xf3f29c6d);
    }

    namespace host_constants
    {
        const Number m = BIG_INTEGER_CHUNKS8(0x73eda753, 0x299d7d48, 0x3339d808, 0x09a1d805, 0x53bda402, 0xfffe5bfe, 0xffffffff, 0x00000001);
        const Number m_sub2 = BIG_INTEGER_CHUNKS8(0x73eda753, 0x299d7d48, 0x3339d808, 0x09a1d805, 0x53bda402, 0xfffe5bfe, 0xfffffffe, 0xffffffff);
        const Number r_mod = BIG_INTEGER_CHUNKS8(0x1824b159, 0xacc5056f, 0x998c4fef, 0xecbc4ff5, 0x5884b7fa, 0x00034802, 0x00000001, 0xfffffffe);
        const Number r2_mod = BIG_INTEGER_CHUNKS8(0x748d9d9, 0x9f59ff11, 0x05d31496, 0x7254398f, 0x2b6cedcb, 0x87925c23, 0xc999e990, 0xf3f29c6d);
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