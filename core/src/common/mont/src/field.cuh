#ifndef MONT_H
#define MONT_H

#include <iostream>
#include <iomanip>
#include <tuple>
#include <random>
#include <cub/cub.cuh>

#define BIG_INTEGER_CHUNKS8(c7, c6, c5, c4, c3, c2, c1, c0) {c0, c1, c2, c3, c4, c5, c6, c7}
#define BIG_INTEGER_CHUNKS16(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0) \
  {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15}

#include "ptx.cuh"

namespace mont
{
  using u32 = u_int32_t;
  using u64 = u_int64_t;
  using i64 = int64_t;
  using i64 = int64_t;
  using usize = size_t;

  // Arithmatics on host, in raw pointer form
  namespace host_arith
  {
    __host__ __forceinline__ u32 addc(u32 a, u32 b, u32 &carry)
    {
      u64 ret = (u64)a + (u64)b + (u64)carry;
      carry = (u32)(ret >> 32);
      return ret;
    }

    __host__ __forceinline__ u32 subb(u32 a, u32 b, u32 &borrow)
    {
      u64 ret = (u64)a - (u64)b - (u64)(borrow >> 31);
      borrow = (u32)(ret >> 32);
      return ret;
    }

    __host__ __forceinline__ u32 madc(u32 a, u32 b, u32 c, u32 &carry)
    {
      u64 ret = (u64)b * (u64)c + (u64)a + (u64)carry;
      carry = (u32)(ret >> 32);
      return ret;
    }

    template <usize N>
    __host__ __forceinline__ u32 sub(u32 *r, const u32 *a, const u32 *b)
    {
      u32 carry = 0;
#pragma unroll
      for (usize i = 0; i < N; i++)
        r[i] = host_arith::subb(a[i], b[i], carry);
      return carry;
    }

    template <usize N>
    __host__ __forceinline__ u32 add(u32 *r, const u32 *a, const u32 *b)
    {
      u32 carry = 0;
#pragma unroll
      for (usize i = 0; i < N; i++)
        r[i] = host_arith::addc(a[i], b[i], carry);
      return carry;
    }

    template <usize N>
    __host__ __forceinline__ void sub_modulo(u32 *r, const u32 *a, const u32 *b, const u32 *m)
    {
      u32 borrow = sub<N>(r, a, b);
      if (borrow)
        add<N>(r, r, m);
    }

    template <usize N>
    __host__ __forceinline__ void add_modulo(u32 *r, const u32 *a, const u32 *b, const u32 *m)
    {
      add<N>(r, a, b);
      sub_modulo<N>(r, r, m, m);
    }

    template <usize N>
    __host__ __forceinline__ void multiply(u32 *r, const u32 *a, const u32 *b)
    {
      u32 carry = 0;

#pragma unroll
      for (usize j = 0; j < N; j++)
        r[j] = madc(0, a[0], b[j], carry);
      r[N] = carry;
      carry = 0;

#pragma unroll
      for (usize i = 1; i < N; i++)
      {
#pragma unroll
        for (usize j = 0; j < N; j++)
          r[i + j] = madc(r[i + j], a[i], b[j], carry);
        r[N + i] = carry;
        carry = 0;
      }
    }

    template <usize N>
    __host__ __forceinline__ void montgomery_reduction(u32 *a, const u32 *m, const u32 m_prime)
    {
      u32 k, carry;
      u32 carry2 = 0;
#pragma unroll
      for (usize i = 0; i < N; i++)
      {
        k = a[i] * m_prime;
        carry = 0;
        madc(a[i], k, m[0], carry);
#pragma unroll
        for (usize j = 1; j < N; j++)
          a[i + j] = madc(a[i + j], k, m[j], carry);
        a[i + N] = addc(a[i + N], carry2, carry);
        carry2 = carry;
      }
    }

    template <usize N, bool MODULO>
    __host__ __forceinline__ void montgomery_multiplication(u32 *r, const u32 *a, const u32 *b, const u32 *m, const u32 m_prime)
    {
      u32 product[2 * N];
      multiply<N>(product, a, b);
      montgomery_reduction<N>(product, m, m_prime);
      memcpy(r, product + N, N * sizeof(u32));
      if (MODULO)
        sub_modulo<N>(r, r, m, m);
    }

    template <usize N>
    __host__ __forceinline__ void random(u32 *r, const u32 *m)
    {
      if constexpr (N == 0)
        return;
      else
      {

        u32 x = (u32)std::rand() % m[N - 1];
        r[N - 1] = x;
        if (x == m[0])
        {
          return random<N - 1>(r, m);
        }
        for (int i = (int)N - 2; i >= 0; i--)
          r[i] = std::rand() % INT32_MAX;
      }
    }

  }

  // Arithmatics on device, in raw pointer form
  namespace device_arith
  {
    // Multiply even limbs of a big number `a` with u32 `b`, writing result to `r`.
    // `a` has `N` limbs, and so does `r`.
    //   | 0      | a2     | 0       | a0    |
    // *                             | b     |
    //   -------------------------------------
    //                     | a0 * b          |
    // + | a2 * b          |
    //   -------------------------------------
    //   | r                                 |
    template <usize N>
    __device__ __forceinline__ void
    multiply_n_1_even(u32 *r, const u32 *a, const u32 b)
    {
#pragma unroll
      for (usize i = 0; i < N; i += 2)
      {
        r[i] = ptx::mul_lo(a[i], b);
        r[i + 1] = ptx::mul_hi(a[i], b);
      }
    }

    // Multiply even limbs of big number `a` with u32 `b`, adding result to `c`, with an optional carry-in `carry_in`.
    // `a` has `N` limbs.
    // Final result written to `r`.
    // `CARRY_IN` controls whether to enable the parameter `carry_in`.
    // Both `a` and `r` has `N` limbs.
    //   | 0      | a2     | 0       | a0    |
    // *                             | b     |
    //   -------------------------------------
    //                     | a0 * b          |
    //   | a2 * b          |
    // + | c                                 |
    //   -------------------------------------
    //   | acc                               |
    template <usize N, bool CARRY_IN = false, bool CARRY_OUT = false>
    __device__ __forceinline__ u32 mad_n_1_even(u32 *acc, const u32 *a, const u32 b, const u32 *c, const u32 carry_in = 0)
    {
      if (CARRY_IN)
        ptx::add_cc(UINT32_MAX, carry_in);
      acc[0] = CARRY_IN ? ptx::madc_lo_cc(a[0], b, c[0]) : ptx::mad_lo_cc(a[0], b, c[0]);
      acc[1] = ptx::madc_hi_cc(a[0], b, c[1]);

#pragma unroll
      for (usize i = 2; i < N; i += 2)
      {
        acc[i] = ptx::madc_lo_cc(a[i], b, c[i]);
        acc[i + 1] = ptx::madc_hi_cc(a[i], b, c[i + 1]);
      }

      if (CARRY_OUT)
        return ptx::addc(0, 0);
      return 0;
    }

    // Like `mad_n_1_even`, but the result of multiplication is accumulated in `acc`.
    template <usize N, bool CARRY_IN = false, bool CARRY_OUT = false>
    __device__ __forceinline__
        u32
        mac_n_1_even(u32 *acc, const u32 *a, const u32 b, const u32 carry_in = 0)
    {
      return mad_n_1_even<N, CARRY_IN, CARRY_OUT>(acc, a, b, acc, carry_in);
    }

    // Multiplies `a` and `b`, where `a` has `N` limbs.
    // The multiplication result is added to `c` and `d` in the following way.
    // `CARRY_IN` controls `carry_for_low`, and `CARRY_OUT` controls return value.
    //                   | a3    | 0     | a1    | 0    |
    // *                                 | b     |
    //   -----------------------------------------
    //                           | a1 * b        |
    //           | a3 * b        |
    //           | d     | c     | odd1  | odd0  |
    //                                   | cr_l  |        cr_l is `carry_for_low`
    // +         | cr_h  |                                cr_h is `carry_for_high`
    //   -----------------------------------------
    //   | cr    | odd                           |
    //
    //   `even` is same as `mad_n_1_even`
    template <usize N, bool CARRY_OUT = false, bool CARRY_IN = false>
    __device__ __forceinline__
        u32
        mad_row(
            u32 *odd,
            u32 *even,
            const u32 *a,
            const u32 b,
            const u32 c = 0,
            const u32 d = 0,
            const u32 carry_for_high = 0,
            const u32 carry_for_low = 0)
    {
      mac_n_1_even<N - 2, CARRY_IN>(odd, a + 1, b, carry_for_low);
      odd[N - 2] = ptx::madc_lo_cc(a[N - 1], b, c);
      odd[N - 1] = CARRY_OUT ? ptx::madc_hi_cc(a[N - 1], b, d) : ptx::madc_hi(a[N - 1], b, d);
      u32 cr = CARRY_OUT ? ptx::addc(0, 0) : 0;
      mac_n_1_even<N, false>(even, a, b);
      if (CARRY_OUT)
      {
        odd[N - 1] = ptx::addc_cc(odd[N - 1], carry_for_high);
        cr = ptx::addc(cr, 0);
      }
      else
        odd[N - 1] = ptx::addc(odd[N - 1], carry_for_high);
      return cr;
    }

    // Similar to `mad_row`, but with c, d set to zero
    template <usize N, bool CARRY_OUT = false, bool CARRY_IN = false>
    __device__ __forceinline__
        u32
        mac_row(
            u32 *odd,
            u32 *even,
            const u32 *a,
            const u32 b,
            const u32 carry_for_high = 0,
            const u32 carry_for_low = 0)
    {
      mac_n_1_even<N, CARRY_IN>(odd, a + 1, b, carry_for_low);
      mac_n_1_even<N, false>(even, a, b);
      u32 cr = 0;
      if (CARRY_OUT)
      {
        odd[N - 1] = ptx::addc_cc(odd[N - 1], carry_for_high);
        cr = ptx::addc(cr, 0);
      }
      else
        odd[N - 1] = ptx::addc(odd[N - 1], carry_for_high);
      return cr;
    }

    // Let `r` be `a` multiplied by `b`.
    // `a` and `b` both have `N` limbs and `r` has `2 * N` limbs.
    // Implements elementry school multiplication algorithm.
    template <usize N>
    __device__ __forceinline__ void multiply_naive(u32 *r, const u32 *a, const u32 *b)
    {
      u32 *even = r;

      __align__(16) u32 odd[2 * N - 2];
      multiply_n_1_even<N>(even, a, b[0]);
      multiply_n_1_even<N>(odd, a + 1, b[0]);
      mad_row<N>(&even[2], &odd[0], a, b[1]);

#pragma unroll
      for (usize i = 2; i < N - 1; i += 2)
      {
        mad_row<N>(&odd[i], &even[i], a, b[i]);
        mad_row<N>(&even[i + 2], &odd[i], a, b[i + 1]);
      }

      even[1] = ptx::add_cc(even[1], odd[0]);
      usize i;
#pragma unroll
      for (i = 1; i < 2 * N - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], 0);
    }

    // Let `r` be sum of `a` and `b`.
    // `a`, `b`, `r` all have `N` limbs.
    template <usize N>
    __device__ __forceinline__
        u32
        add(u32 *r, const u32 *a, const u32 *b)
    {
      r[0] = ptx::add_cc(a[0], b[0]);
#pragma unroll
      for (usize i = 1; i < N; i++)
        r[i] = ptx::addc_cc(a[i], b[i]);
      return ptx::addc(0, 0);
    }

    // Let `r` be difference of `a` and `b`.
    // `a`, `b`, `r` all have `N` limbs.
    template <usize N>
    __device__ __forceinline__
        u32
        sub(u32 *r, const u32 *a, const u32 *b)
    {
      r[0] = ptx::sub_cc(a[0], b[0]);
#pragma unroll
      for (usize i = 1; i < N; i++)
        r[i] = ptx::subc_cc(a[i], b[i]);
      return ptx::subc(0, 0);
    }

    // Multiplies `a` and `b`, adding result to `in1` and `in2`.
    // `a` and `b` have `N / 2` limbs, while `in1` and `in2` have `N` limbs.
    template <usize N>
    __device__ __forceinline__ void mad2_rows(u32 *r, const u32 *a, const u32 *b, const u32 *in1, const u32 *in2)
    {
      __align__(16) u32 odd[N - 2];
      u32 *even = r;
      u32 first_row_carry = mad_n_1_even<(N >> 1), false, true>(even, a, b[0], in1);
      u32 carry = mad_n_1_even<(N >> 1), false, true>(odd, &a[1], b[0], &in2[1]);

#pragma unroll
      for (usize i = 2; i < ((N >> 1) - 1); i += 2)
      {
        carry = mad_row<(N >> 1), true, false>(
            &even[i], &odd[i - 2], a, b[i - 1], in1[(N >> 1) + i - 2], in1[(N >> 1) + i - 1], carry);
        carry = mad_row<(N >> 1), true, false>(
            &odd[i], &even[i], a, b[i], in2[(N >> 1) + i - 1], in2[(N >> 1) + i], carry);
      }
      mad_row<(N >> 1), false, true>(
          &even[N >> 1], &odd[(N >> 1) - 2], a, b[(N >> 1) - 1], in1[N - 2], in1[N - 1], carry, first_row_carry);

      even[0] = ptx::add_cc(even[0], in2[0]);
      usize i;
#pragma unroll
      for (i = 0; i < N - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], in2[i + 1]);
    }

    // Compute `r = a * b` where `r` has `N * 2` limbs while `a` and `b` have `N` limbs.
    // Implements 1-layer Kruskaba Algorithm.
    template <usize N>
    __device__ __forceinline__ void multiply(u32 *r, const u32 *a, const u32 *b)
    {
      if (N > 2)
      {
        multiply_naive<(N >> 1)>(r, a, b);
        multiply_naive<(N >> 1)>(&r[N], &a[N >> 1], &b[N >> 1]);
        __align__(16) u32 middle_part[N];
        __align__(16) u32 diffs[N];
        u32 carry1 = sub<(N >> 1)>(diffs, &a[N >> 1], a);
        u32 carry2 = sub<(N >> 1)>(&diffs[N >> 1], b, &b[N >> 1]);
        mad2_rows<N>(middle_part, diffs, &diffs[N >> 1], r, &r[N]);
        if (carry1)
          sub<(N >> 1)>(&middle_part[N >> 1], &middle_part[N >> 1], &diffs[N >> 1]);
        if (carry2)
          sub<(N >> 1)>(&middle_part[N >> 1], &middle_part[N >> 1], diffs);
        add<N>(&r[N >> 1], &r[N >> 1], middle_part);

#pragma unroll
        for (usize i = N + (N >> 1); i < 2 * N; i++)
          r[i] = ptx::addc_cc(r[i], 0);
      }
      else if (N == 2)
      {
        __align__(8) uint32_t odd[2];
        r[0] = ptx::mul_lo(a[0], b[0]);
        r[1] = ptx::mul_hi(a[0], b[0]);
        r[2] = ptx::mul_lo(a[1], b[1]);
        r[3] = ptx::mul_hi(a[1], b[1]);
        odd[0] = ptx::mul_lo(a[0], b[1]);
        odd[1] = ptx::mul_hi(a[0], b[1]);
        odd[0] = ptx::mad_lo(a[1], b[0], odd[0]);
        odd[1] = ptx::mad_hi(a[1], b[0], odd[1]);
        r[1] = ptx::add_cc(r[1], odd[0]);
        r[2] = ptx::addc_cc(r[2], odd[1]);
        r[3] = ptx::addc(r[3], 0);
      }
      else if (N == 1)
      {
        r[0] = ptx::mul_lo(a[0], b[0]);
        r[1] = ptx::mul_hi(a[0], b[0]);
      }
    }

    // Computes `r = x >> k` where `r` and `x` have `N` limbs.
    template <usize N>
    __host__ __device__ __forceinline__ void slr(u32 *r, const u32 *x, const u32 k)
    {
      if (k % 32 == 0)
      {
        u32 shift = k / 32;
#pragma unroll
        for (usize i = N - shift; i < N; i++)
        {
          r[i] = 0;
        }

#pragma unroll
        for (usize i = 0; i < N - shift; i++)
        {
          r[i] = x[i + shift];
        }
        return;
      }
#pragma unroll
      for (usize i = 1; i <= N; i++)
      {
        if (k < i * 32)
        {

          u32 k_lo = k - (i - 1) * 32;
          u32 k_hi = i * 32 - k;
#pragma unroll
          for (int j = N - 1; j > N - i; j--)
          {
            r[j] = 0;
          }
          r[N - i] = x[N - 1] >> k_lo;
#pragma unroll
          for (int j = N - i - 1; j >= 0; j--)
            r[j] = (x[j + i] << k_hi) | (x[j + i - 1] >> k_lo);
          return;
        }
      }
    }

    // Apply Montgomery Reduction to `a`, with modulus `m` and `m_prime` satisfying
    //    m m_prime = -1 (mod 2^32)
    template <usize N>
    __device__ __forceinline__ void montgomery_reduction(u32 *a, const u32 *m, const u32 m_prime)
    {
      __align__(16) u32 carries[2 * N] = {0};
#pragma unroll
      for (usize i = 0; i < N - 1; i += 1)
      {
        u32 u = a[i] * m_prime;
        mac_n_1_even<N, false, false>(&a[i], m, u);
        carries[i + N] = ptx::addc(carries[i + N], 0);
        mac_n_1_even<N, false, false>(&a[i + 1], &m[1], u);
        carries[i + N + 1] = ptx::addc(carries[i + N + 1], 0);
      }

      u32 u = a[N - 1] * m_prime;
      mac_n_1_even<N, false, false>(&a[N - 1], m, u);
      carries[2 * N - 1] = ptx::addc(carries[2 * N - 1], 0);
      mac_n_1_even<N, false, false>(&a[N], &m[1], u);

      add<2 * N>(a, a, carries);
    }

    // Computes `r = a - b mod m`.
    // `r`, `a`, `b`, `m` all have limbs `N`
    template <usize N>
    __device__ __forceinline__ void sub_modulo(u32 *r, const u32 *a, const u32 *b, const u32 *m)
    {
      u32 borrow = sub<N>(r, a, b);
      if (borrow)
        add<N>(r, r, m);
    }

    // Computes `r = a + b mod m`.
    // `r`, `a`, `b`, `m` all have limbs `N`
    template <usize N>
    __device__ __forceinline__ void add_modulo(u32 *r, const u32 *a, const u32 *b, const u32 *m)
    {
      add<N>(r, a, b);
      sub_modulo<N>(r, r, m, m);
    }

    // Computes `r = a * b mod m`.
    // `r`, `a`, `b`, `m` all have limbs `N`
    template <usize N, bool MODULO>
    __device__ __forceinline__ void montgomery_multiplication(u32 *r, const u32 *a, const u32 *b, const u32 *m, const u32 m_prime)
    {
      __align__(16) u32 prod[2 * N];
      multiply<N>(prod, a, b);
      montgomery_reduction<N>(prod, m, m_prime);
#pragma unroll
      for (usize i = N; i < 2 * N; i++)
        r[i - N] = prod[i];
      if (MODULO)
        sub_modulo<N>(r, r, m, m);
    }
  }

  namespace device_arith2
  {
    // Copyright Supranational LLC
    // Licensed under the Apache License, Version 2.0, see LICENSE for details.
    // SPDX-License-Identifier: Apache-2.0

    template <usize n>
    __device__ __forceinline__ void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi)
    {
      for (size_t j = 0; j < n; j += 2)
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(acc[j]), "=r"(acc[j + 1])
            : "r"(a[j]), "r"(bi));
    }

    template <usize n>
    __device__ __forceinline__ void cmad_n(uint32_t *acc, const uint32_t *a, uint32_t bi)
    {
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
          : "+r"(acc[0]), "+r"(acc[1])
          : "r"(a[0]), "r"(bi));
      for (size_t j = 2; j < n; j += 2)
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(acc[j]), "+r"(acc[j + 1])
            : "r"(a[j]), "r"(bi));
      // return carry flag
    }

    template <usize n>
    __device__ __forceinline__ void madc_n_rshift(uint32_t *odd, const uint32_t *a, uint32_t bi)
    {
      for (size_t j = 0; j < n - 2; j += 2)
        asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
            : "=r"(odd[j]), "=r"(odd[j + 1])
            : "r"(a[j]), "r"(bi), "r"(odd[j + 2]), "r"(odd[j + 3]));
      asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
          : "=r"(odd[n - 2]), "=r"(odd[n - 1])
          : "r"(a[n - 2]), "r"(bi));
    }

    template <usize n>
    __device__ __forceinline__ void mad_n_redc(uint32_t *even, uint32_t *odd,
                                               const uint32_t *a, uint32_t bi, const uint32_t *m, uint32_t m_prime, bool first = false)
    {
      if (first)
      {
        mul_n<n>(odd, a + 1, bi);
        mul_n<n>(even, a, bi);
      }
      else
      {
        asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
        madc_n_rshift<n>(odd, a + 1, bi);
        cmad_n<n>(even, a, bi);
        asm("addc.u32 %0, %0, 0;" : "+r"(odd[n - 1]));
      }

      uint32_t mi = even[0] * m_prime;

      cmad_n<n>(odd, m + 1, mi);
      cmad_n<n>(even, m, mi);
      asm("addc.u32 %0, %0, 0;" : "+r"(odd[n - 1]));
    }

    template <usize n>
    __device__ __forceinline__ void cadd_n(uint32_t *acc, const uint32_t *a)
    {
      asm("add.cc.u32 %0, %0, %1;" : "+r"(acc[0]) : "r"(a[0]));
      for (size_t i = 1; i < n; i++)
        asm("addc.cc.u32 %0, %0, %1;" : "+r"(acc[i]) : "r"(a[i]));
      // return carry flag
    }

    template <usize n>
    __device__ __forceinline__ void final_sub(uint32_t *even, uint32_t carry, uint32_t *tmp, const uint32_t *m)
    {
      size_t i;
      asm volatile("{ .reg.pred %top;");

      asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(m[0]));
      for (i = 1; i < n; i++)
        asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(m[i]));

      asm volatile("subc.u32 %0, 0, 0; setp.eq.u32 %top, %0, 0;" : "=r"(carry));

      for (i = 0; i < n; i++)
        asm volatile("@%top mov.b32 %0, %1;" : "+r"(even[i]) : "r"(tmp[i]));

      asm volatile("}");
    }

    template <usize n, bool _MODULO>
    __device__ __forceinline__ void montgomery_multiplication(uint32_t *even, const uint32_t *a, const uint32_t *b, const uint32_t *m, uint32_t m_prime)
    {
      uint32_t odd[n];

#pragma unroll
      for (size_t i = 0; i < n; i += 2)
      {
        mad_n_redc<n>(&even[0], &odd[0], &a[0], b[i], m, m_prime, i == 0);
        mad_n_redc<n>(&odd[0], &even[0], &a[0], b[i + 1], m, m_prime);
      }

      // merge |even| and |odd|
      cadd_n<n - 1>(&even[0], &odd[1]);
      asm("addc.u32 %0, %0, 0;" : "+r"(even[n - 1]));

      final_sub<n>(even, 0, &odd[0], m);
    }
  }

  // A big integer
  template <usize LIMBS_>
  struct
      __align__(16)
          Number
  {
    static const usize LIMBS = LIMBS_;
    u32 limbs[LIMBS];

    __device__ __host__ __forceinline__
    Number() {}

    // Constructor: `Number x = {0, 1, 2, 3, 4, 5, 6, 7}` in little endian
    constexpr __forceinline__ Number(const std::initializer_list<uint32_t> &values) : limbs{}
    {
      size_t i = 0;
      for (auto value : values)
      {
        if (i >= LIMBS)
          break;
        limbs[i++] = value;
      }
    }

    static __device__ __host__ __forceinline__
        Number
        load(const u32 *p, u32 stride = 1)
    {
      Number r;
#ifdef __CUDA_ARCH__
      if (stride == 1 && LIMBS % 4 == 0)
      {
#pragma unroll
        for (usize i = 0; i < LIMBS / 4; i++)
        {
          reinterpret_cast<uint4 *>(r.limbs)[i] = reinterpret_cast<const uint4 *>(p)[i];
        }
      }
      else if (stride == 1 && LIMBS % 2 == 0)
      {
#pragma unroll
        for (usize i = 0; i < LIMBS / 2; i++)
        {
          reinterpret_cast<uint2 *>(r.limbs)[i] = reinterpret_cast<const uint2 *>(p)[i];
        }
      }
      else
#endif
      {
#pragma unroll
        for (usize i = 0; i < LIMBS; i++)
          r.limbs[i] = p[i * stride];
      }
      return r;
    }

    __device__ __host__ __forceinline__ void store(u32 * p, u32 stride = 1) const &
    {
#ifdef __CUDA_ARCH__
      if (stride == 1 && LIMBS % 4 == 0)
      {
#pragma unroll
        for (usize i = 0; i < LIMBS / 4; i++)
        {
          reinterpret_cast<uint4 *>(p)[i] = reinterpret_cast<const uint4 *>(limbs)[i];
        }
      }
      else if (stride == 1 && LIMBS % 2 == 0)
      {
#pragma unroll
        for (usize i = 0; i < LIMBS / 2; i++)
        {
          reinterpret_cast<uint2 *>(p)[i] = reinterpret_cast<const uint2 *>(limbs)[i];
        }
      }
      else
#endif
      {
#pragma unroll
        for (usize i = 0; i < LIMBS; i++)
          p[i * stride] = limbs[i];
      }
    }

    static __device__ __host__ __forceinline__
        Number
        zero()
    {
      Number r;
      memset(r.limbs, 0, LIMBS * sizeof(u32));
      return r;
    }

    // Shift logical right by `k` bits
    __host__ __device__ __forceinline__
        Number
        slr(u32 k) const &
    {
      Number r;
      device_arith::slr<LIMBS>(r.limbs, limbs, k);
      return r;
    }

    // Return the [`lo`, `lo + n_bits`) bits in big number.
    // `n_bits` must be smaller than 32.
    __host__ __device__ __forceinline__
        u32
        bit_slice(u32 lo, u32 n_bits)
    {
      Number t = slr(lo);
      return t.limbs[0] & ((1 << n_bits) - 1);
    }

    template <u32 windows, u32 bits_per_window>
    __host__ __device__ __forceinline__ void bit_slice(int(&r)[windows])
    {
      static_assert(bits_per_window <= 31, "Too many bits per window");

      // can be optimized by using hand written __funnelshift sequences
      Number t = *this;
#pragma unroll
      for (u32 i = 0; i < windows; i++)
      {
        r[i] = t.limbs[0] & ((1 << bits_per_window) - 1);
        t = t.slr(bits_per_window);
      }
    }

    // Word-by-word equality
    __host__ __device__ __forceinline__ bool
    operator==(const Number &rhs) const &
    {
      bool r = true;
#pragma unroll
      for (usize i = 0; i < LIMBS; i++)
        r = r && (limbs[i] == rhs.limbs[i]);
      return r;
    }

    __host__ __device__ __forceinline__ bool
    operator!=(const Number &rhs) const &
    {
      bool r = false;
#pragma unroll
      for (usize i = 0; i < LIMBS; i++)
        r = r || (limbs[i] != rhs.limbs[i]);
      return r;
    }

    __host__ __device__ __forceinline__ constexpr bool
    is_zero() const &
    {
      bool r = true;
#pragma unroll
      for (usize i = 0; i < LIMBS; i++)
        r = r && (limbs[i] == 0);
      return r;
    }

    // Return the [`i`, `i + N`) words in big number.
    template <usize N>
    __host__ __device__ __forceinline__
        Number<N>
        limbs_slice(u32 i)
    {
      Number<N> r;
      for (usize j = i; j < N; j++)
        r.limbs[j - i] = limbs[j];
      return r;
    }

    // Big number multiplication.
    // Result takes twice the number of bits of operands.
    __host__ __device__ __forceinline__
        Number<LIMBS * 2>
        operator*(const Number &rhs) const &
    {
      Number<LIMBS * 2> r;
#ifdef __CUDA_ARCH__
      device_arith::multiply<LIMBS>(r.limbs, limbs, rhs.limbs);
#else
      host_arith::multiply<LIMBS>(r.limbs, limbs, rhs.limbs);
#endif
      return r;
    }

    // Big number addition
    __host__ __device__ __forceinline__
        Number
        operator+(const Number &rhs) const &
    {
      Number r;
#ifdef __CUDA_ARCH__
      device_arith::add<LIMBS>(r.limbs, limbs, rhs.limbs);
#else
      host_arith::add<LIMBS>(r.limbs, limbs, rhs.limbs);
#endif
      return r;
    }

    // Big number subtraction
    __host__ __device__ __forceinline__
        Number
        operator-(const Number &rhs) const &
    {
      u32 useless;
      auto r = sub_borrowed(rhs, useless);
      return r;
    }

    // Big number subtraction.
    // Carry of subtraction is written to `borrow_ret`: `UINT32_MAX` if borrow occurred, otherwise 0.
    __host__ __device__ __forceinline__
        Number
        sub_borrowed(const Number &rhs, u32 &borrow_ret) const &
    {
      Number r;
#ifdef __CUDA_ARCH__
      borrow_ret = device_arith::sub<LIMBS>(r.limbs, limbs, rhs.limbs);
#else
      borrow_ret = host_arith::sub<LIMBS>(r.limbs, limbs, rhs.limbs);
#endif
      return r;
    }

    // Big number square
    __host__ __device__ __forceinline__
        Number<LIMBS * 2>
        square() const &
    {
      return *this * *this;
    }

    __device__ __forceinline__ Number shuffle_down(const u32 delta) const &
    {
      Number res;
#pragma unroll
      for (usize i = 0; i < LIMBS; i++)
      {
        res.limbs[i] = __shfl_down_sync(0xFFFFFFFF, limbs[i], delta);
      }
      return res;
    }
  };

  // An element on field defined by `Params`
  template <class Params>
  struct Element
  {
    using ParamsType = Params;
    static const usize LIMBS = Params::LIMBS;

    Number<LIMBS> n;

    __host__ __device__ __forceinline__ Element() {}
    constexpr __host__ __device__ __forceinline__ Element(Number<LIMBS> n) : n(n) {}

    static __host__ __device__ __forceinline__
        Element
        load(const u32 *p, u32 stride = 1)
    {
      Element r;
      r.n = Number<LIMBS>::load(p, stride);
      return r;
    }

    __host__ __device__ __forceinline__ void store(u32 *p, u32 stride = 1) const &
    {
      n.store(p, stride);
    }

    // Addition identity on field
    static __host__ __device__ __forceinline__
        Element
        zero()
    {
      Element r;
      r.n = Number<LIMBS>::zero();
      return r;
    }

    __host__ __device__ __forceinline__ constexpr bool
    is_zero() const &
    {
      return n.is_zero();
    }

    // Word-by-word equality
    __host__ __device__ __forceinline__ bool operator==(const Element &rhs) const &
    {
      return n == rhs.n;
    }

    __host__ __device__ __forceinline__ bool operator!=(const Element &rhs) const &
    {
      return n != rhs.n;
    }

    // Multiplication identity on field
    static __device__ __host__ __forceinline__
        Element
        one()
    {
      Element elem;
      elem.n = Params::r_mod();
      return elem;
    }

    // Field multiplication
    template <bool MODULO = true>
    __host__ __device__ __forceinline__
        Element
        mul(const Element &rhs) const &
    {
      Element r;
#ifdef __CUDA_ARCH__
      device_arith2::montgomery_multiplication<LIMBS, MODULO>(r.n.limbs, n.limbs, rhs.n.limbs, Params::m().limbs, Params::m_prime);
#else
      host_arith::montgomery_multiplication<LIMBS, MODULO>(r.n.limbs, n.limbs, rhs.n.limbs, Params::m().limbs, Params::m_prime);
#endif
      return r;
    }

    __host__ __device__ __forceinline__
        Element
        operator*(const Element &rhs) const &
    {
      return mul<true>(rhs);
    }

    template <bool MODULO = true>
    __host__ __device__ __forceinline__
        Element
        square() const &
    {
      return mul<MODULO>(*this);
    }

    // Field addition
    __host__ __device__ __forceinline__
        Element
        operator+(const Element &rhs) const &
    {
      Element r;
#ifdef __CUDA_ARCH__
      device_arith::add_modulo<LIMBS>(r.n.limbs, n.limbs, rhs.n.limbs, Params::m().limbs);
#else
      host_arith::add_modulo<LIMBS>(r.n.limbs, n.limbs, rhs.n.limbs, Params::m().limbs);
#endif
      return r;
    }

    // Field subtraction
    __host__ __device__ __forceinline__
        Element
        operator-(const Element &rhs) const &
    {
      Element r;
#ifdef __CUDA_ARCH__
      device_arith::sub_modulo<LIMBS>(r.n.limbs, n.limbs, rhs.n.limbs, Params::m().limbs);
#else
      host_arith::sub_modulo<LIMBS>(r.n.limbs, n.limbs, rhs.n.limbs, Params::m().limbs);
#endif
      return r;
    }

    __device__ __host__ __forceinline__
        Element
        sub_modulo_mm2(const Element &rhs) const &
    {
      Element r;
#ifdef __CUDA_ARCH__
      device_arith::sub_modulo<LIMBS>(r.n.limbs, n.limbs, rhs.n.limbs, Params::mm2().limbs);
#else
      host_arith::sub_modulo<LIMBS>(r.n.limbs, n.limbs, rhs.n.limbs, Params::mm2().limbs);
#endif
      return r;
    }

    __device__ __host__ __forceinline__
        Element
        add_modulo_mm2(const Element &rhs) const &
    {
      Element r;
#ifdef __CUDA_ARCH__
      device_arith::add_modulo<LIMBS>(r.n.limbs, n.limbs, rhs.n.limbs, Params::mm2().limbs);
#else
      host_arith::add_modulo<LIMBS>(r.n.limbs, n.limbs, rhs.n.limbs, Params::mm2().limbs);
#endif
      return r;
    }

    __device__ __host__ __forceinline__
        Element
        modulo_m() const &
    {
      Element r;
#ifdef __CUDA_ARCH__
      device_arith::sub_modulo<LIMBS>(r.n.limbs, n.limbs, Params::m().limbs, Params::m().limbs);
#else
      host_arith::sub_modulo<LIMBS>(r.n.limbs, n.limbs, Params::m().limbs, Params::m().limbs);
#endif
      return r;
    }

    __host__ __device__ __forceinline__
        Element
        neg() const &
    {
      if (n.is_zero())
        return Element::zero();
      Element r;
#ifdef __CUDA_ARCH__
      device_arith::sub<LIMBS>(r.n.limbs, Params::m().limbs, n.limbs);
#else
      host_arith::sub<LIMBS>(r.n.limbs, Params::m().limbs, n.limbs);
#endif
      return r;
    }

    // Convert a big number to its representation in field.
    static __host__ __device__ __forceinline__
        Element
        from_number(const Number<LIMBS> &n)
    {
      Element r;
#ifdef __CUDA_ARCH__
      device_arith::montgomery_multiplication<LIMBS, true>(r.n.limbs, n.limbs, Params::r2_mod().limbs, Params::m().limbs, Params::m_prime);
#else
      host_arith::montgomery_multiplication<LIMBS, true>(r.n.limbs, n.limbs, Params::r2_mod().limbs, Params::m().limbs, Params::m_prime);
#endif
      return r;
    }

    // Reverse of `from_number`
    __host__ __device__ __forceinline__
        Number<LIMBS>
        to_number() const &
    {
      Number<2 * LIMBS> n;
      memcpy(n.limbs, this->n.limbs, LIMBS * sizeof(u32));
      memset(n.limbs + LIMBS, 0, LIMBS * sizeof(u32));

      Number<LIMBS> r;
#ifdef __CUDA_ARCH__
      device_arith::montgomery_reduction<LIMBS>(n.limbs, Params::m().limbs, Params::m_prime);
#else
      host_arith::montgomery_reduction<LIMBS>(n.limbs, Params::m().limbs, Params::m_prime);
#endif

      memcpy(r.limbs, n.limbs + LIMBS, LIMBS * sizeof(u32));
      return r;
    }

    // Helper function for `pow`
    static __host__ __device__ __forceinline__ void
    pow_iter(const Element &a, bool &found_one, Element &res, u32 p, u32 deg = 31)
    {
#pragma unroll
      for (int i = deg; i >= 0; i--)
      {
        if (found_one)
          res = res.square();
        if ((p >> i) & 1)
        {
          found_one = true;
          res = res * a;
        }
      }
    }

    // Field power
    __host__ __device__ __forceinline__
        Element
        pow(const Number<LIMBS> &p) const &
    {
      auto res = one();
      bool found_one = false;
#pragma unroll
      for (int i = LIMBS - 1; i >= 0; i--)
        pow_iter(*this, found_one, res, p.limbs[i]);
      return res;
    }

    __host__ __device__ __forceinline__
    Element pow(u64 p) const & {
      auto res = one();
      bool found_one = false;
      #pragma unroll
      for (int i = 63; i >= 0; i--)
      {
        if (found_one)
          res = res.square();
        if ((p >> i) & 1)
        {
          found_one = true;
          res = res * (*this);
        }
      }
      return res;
    }

    __host__ __device__ __forceinline__ Element pow(u32 p, u32 deg = 31) const &
    {
      auto res = one();
      bool found_one = false;
      pow_iter(*this, found_one, res, p, deg);
      return res;
    }

    // Field inversion
    __host__ __device__ __forceinline__ Element invert() const &
    {
      return this->pow(Params::m_sub2());
    }

    __host__ __device__ __forceinline__ bool lt_2m() const &
    {
      auto m = ParamsType::mm2();
      for (int i = LIMBS - 1; i >= 0; i--)
        if (n.limbs[i] < m.limbs[i])
          return true;
        else if (n.limbs[i] > m.limbs[i])
          return false;
      return false;
    }

    // Generate a random field element
    static __host__ __forceinline__
        Element
        host_random()
    {
      Element r;
      host_arith::random<LIMBS>(r.n.limbs, Params::m().limbs);
      return r;
    }

    __device__ __forceinline__ Element shuffle_down(const u32 delta) const &
    {
      Element res;
      res.n = n.shuffle_down(delta);
      return res;
    }
  };

  template <usize LIMBS>
  __forceinline__ std::istream &
  operator>>(std::istream &is, Number<LIMBS> &n)
  {
    is >> std::hex;
    char _;
    is >> _ >> _;
    for (int i = LIMBS - 1; i >= 1; i--)
      is >> n.limbs[i] >> _;
    is >> n.limbs[0];
    return is;
  }

  template <usize LIMBS>
  __forceinline__ std::ostream &
  operator<<(std::ostream &os, const Number<LIMBS> &n)
  {
    os << "0x";
    for (usize i = LIMBS - 1; i >= 1; i--)
      os << std::hex << std::setfill('0') << std::setw(8) << n.limbs[i] << '_';
    os << std::hex << std::setfill('0') << std::setw(8) << n.limbs[0];
    return os;
  }

  template <class Params>
  __forceinline__ std::ostream &
  operator<<(std::ostream &os, const Element<Params> &e)
  {
    auto n = e.to_number();
    os << n;
    return os;
  }

  template <typename Field, u32 io_group>
  __forceinline__ __device__ auto load_exchange(u32 *data, typename cub::WarpExchange<u32, io_group, io_group>::TempStorage temp_storage[]) -> Field
  {
    using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;
    const static usize WORDS = Field::LIMBS;
    const u32 io_id = threadIdx.x & (io_group - 1);
    const u32 lid_start = threadIdx.x - io_id;
    const int warp_id = static_cast<int>(threadIdx.x) / io_group;
    u32 thread_data[io_group];
#pragma unroll
    for (u64 i = lid_start; i != lid_start + io_group; i++)
    {
      if (io_id < WORDS)
      {
        thread_data[i - lid_start] = data[i * WORDS + io_id];
      }
    }
    WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
    __syncwarp();
    return Field::load(thread_data);
  }
  template <typename Field, u32 io_group>
  __forceinline__ __device__ void store_exchange(Field ans, u32 *dst, typename cub::WarpExchange<u32, io_group, io_group>::TempStorage temp_storage[])
  {
    using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;
    const static usize WORDS = Field::LIMBS;
    const u32 io_id = threadIdx.x & (io_group - 1);
    const u32 lid_start = threadIdx.x - io_id;
    const int warp_id = static_cast<int>(threadIdx.x) / io_group;
    u32 thread_data[io_group];
    ans.store(thread_data);
    WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
    __syncwarp();
#pragma unroll
    for (u64 i = lid_start; i != lid_start + io_group; i++)
    {
      if (io_id < WORDS)
      {
        dst[i * WORDS + io_id] = thread_data[i - lid_start];
      }
    }
  }

  __forceinline__ constexpr u32 pow2_ceiling(u32 x)
  {
    u32 r = 2;
    while (r < x)
      r *= 2;
    return r;
  }

  //   // For an array of field elements layouted like
  //   // [0].0    [0].1    ...    [0].N
  //   // [1].0    [1].1    ...    [1].N
  //   // ...
  //   // [M].0    [M].1    ...    [M].N
  //   // where N is LIMBS of each element and M is blockSize.x,
  //   // load the i-th scalar to i-th thread's `dst`
  //   //
  //   // Invariants:
  //   // - `data` must be the same across each `io_group` threads
  //   template <u32 WORDS, u32 io_group = pow2_ceiling(WORDS), typename GetId>
  //   __forceinline__ __device__ void load_exchange_raw(
  //       u32 dst[WORDS],
  //       u32 *data,
  //       GetId gpos,
  //       typename cub::WarpExchange<u32, io_group, io_group>::TempStorage temp_storage[])
  //   {

  //     using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;
  //     const u32 io_id = threadIdx.x & (io_group - 1);
  //     const u32 lid_start = threadIdx.x - io_id;
  //     const int warp_id = static_cast<int>(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / io_group;
  // #pragma unroll
  //     for (u32 i = lid_start; i != lid_start + io_group; i++)
  //     {
  //       if (io_id < WORDS)
  //       {
  //         dst[i - lid_start] = data[gpos(i) * WORDS + io_id];
  //       }
  //     }
  //     WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(dst, dst);
  //     __syncwarp();
  //   }

  //   template <typename Field, u32 io_group = pow2_ceiling(Field::LIMBS), typename GetId>
  //   __forceinline__ __device__ auto load_exchange(u32 *data, GetId gpos, typename cub::WarpExchange<u32, io_group, io_group>::TempStorage temp_storage[]) -> Field
  //   {
  //     u32 thread_data[io_group];
  //     load_exchange_raw<Field::LIMBS>(thread_data, data, gpos, temp_storage);
  //     return Field::load(thread_data);
  //   }

  //   // The reversed effect of `load_exchange_raw`
  //   template <u32 WORDS, u32 io_group = pow2_ceiling(WORDS), typename GetId>
  //   __forceinline__ __device__ void store_exchange_raw(
  //       u32 *dst,
  //       u32 from[WORDS],
  //       GetId get_gpos,
  //       typename cub::WarpExchange<u32, io_group, io_group>::TempStorage temp_storage[])
  //   {
  //     using WarpExchangeT = cub::WarpExchange<u32, io_group, io_group>;
  //     // const static usize WORDS = Field::LIMBS;
  //     const u32 io_id = threadIdx.x & (io_group - 1);
  //     const u32 lid_start = threadIdx.x - io_id;
  //     const int warp_id = static_cast<int>(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / io_group;
  //     WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(from, from);
  //     __syncwarp();
  // #pragma unroll
  //     for (u64 i = lid_start; i != lid_start + io_group; i++)
  //     {
  //       u64 gpos = get_gpos(i);
  //       if (io_id < WORDS)// && gpos < dst_len)
  //       {
  //         dst[gpos * WORDS + io_id] = from[i - lid_start];
  //       }
  //     }
  //   }

  //   template <typename Field, u32 io_group = pow2_ceiling(Field::LIMBS), typename GetId>
  //   __forceinline__ __device__ void store_exchange(Field &ans, u32 *dst, GetId gpos, typename cub::WarpExchange<u32, io_group, io_group>::TempStorage temp_storage[])
  //   {
  //     u32 thread_data[io_group];
  //     ans.store(thread_data);
  //     store_exchange_raw<Field::LIMBS>(dst, thread_data, gpos, temp_storage);
  //   }

}

#endif