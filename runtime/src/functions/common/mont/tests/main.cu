#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <iostream>
#include <ctime>

#include "../src/bn254_fr.cuh"
using bn254_fr::Element;

using Number = mont::Number<8>;
using Number2 = mont::Number<16>;
using mont::u32;
using mont::u64;

__global__ void bn_add(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = na + nb;
  nr.store(r);
}

__global__ void bn_sub(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = na - nb;
  nr.store(r);
}

__global__ void mont_add(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Element::load(a);
  auto nb = Element::load(b);
  auto nr = na + nb;
  nr.store(r);
}

__global__ void mont_sub(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Element::load(a);
  auto nb = Element::load(b);
  auto nr = na - nb;
  nr.store(r);
}

__global__ void bn_mul(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nb = Number::load(b);
  auto nr = na * nb;
  nr.store(r);
}

__global__ void bn_square(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nr = na.square();
  nr.store(r);
}

__global__ void mont_mul(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Element::load(a);
  auto nb = Element::load(b);
  auto nr = na * nb;
  nr.store(r);
}

__global__ void convert_to_mont(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto elem = Element::from_number(na);
  elem.store(r);
}

__global__ void convert_from_mont(u32 *r, const u32 *a, const u32 *b)
{
  auto ea = Element::load(a);
  auto na = ea.to_number();
  na.store(r);
}

__global__ void mont_neg(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Element::load(a);
  auto elem = na.neg();
  elem.store(r);
}

__global__ void mont_square(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto ea = Element::from_number(na);
  auto elem = ea * ea;
  elem.store(r);
}

__global__ void mont_pow(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto ea = Element::from_number(na);
  auto nb = Number::load(b);
  auto ea_pow = ea.pow(nb);
  ea_pow.store(r);
}

__global__ void mont_inv(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto ea = Element::from_number(na);
  auto ea_inv = ea.invert();
  ea_inv.store(r);
}

__global__ void bn_slr(u32 *r, const u32 *a, const u32 *b)
{
  auto na = Number::load(a);
  auto nr = na.slr(*b);
  nr.store(r);
}

__global__ void mont_add_mm2(u32 *r, const u32 *a, const u32 *b)
{
  auto ea = Element::load(a);
  auto eb = Element::load(b);
  auto er = ea.add_modulo_mm2(eb);
  er.store(r);
}

__global__ void mont_sub_mm2(u32 *r, const u32 *a, const u32 *b)
{
  auto ea = Element::load(a);
  auto eb = Element::load(b);
  auto er = ea.sub_modulo_mm2(eb);
  er.store(r);
}

__global__ void mont_modulo_m(u32 *r, const u32 *a, const u32 *b)
{
  auto ea = Element::load(a);
  auto er = ea.modulo_m();
  er.store(r);
}

template <u32 WORDS>
void test_mont_kernel(const u32 r[WORDS], const u32 a[WORDS],
                      const u32 b[WORDS], void kernel(u32 *, const u32 *, const u32 *))
{
  u32 *dr, *da, *db;
  size_t bytes = WORDS * sizeof(u32);
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&dr, bytes);
  cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, da, db);

  err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  u32 got_r[WORDS];
  cudaMemcpy(got_r, dr, bytes, cudaMemcpyDeviceToHost);

  const auto n_got_r = Number::load(got_r);
  const auto nr = Number::load(r);

  if (nr != n_got_r)
    FAIL("Expected ", nr, ", but got ", n_got_r);
}

template <u32 WORDS>
void test_mont_kernel2(const u32 r[WORDS], const u32 a[WORDS],
                       const u32 b[WORDS], void kernel(u32 *, const u32 *, const u32 *))
{
  u32 *dr, *da, *db;
  size_t bytes = WORDS * sizeof(u32);
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&dr, bytes * 2);
  cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, da, db);

  err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  u32 got_r[WORDS * 2];
  cudaMemcpy(got_r, dr, bytes * 2, cudaMemcpyDeviceToHost);

  const auto n_got_r = Number2::load(got_r);
  const auto nr = Number2::load(r);

  if (n_got_r != nr)
  {
    FAIL("Expected\n  ", nr, ", but got\n  ", n_got_r);
  }
}

template <u32 WORDS>
void test_host(const u32 r[WORDS], const u32 a[WORDS],
               const u32 b[WORDS], Number f(const Number &a, const Number &b))
{
  const auto nr = Number::load(r);
  const auto na = Number::load(a);
  const auto nb = Number::load(b);
  const auto n_got_r = f(na, nb);
  REQUIRE(nr == n_got_r);
}

template <u32 WORDS>
void test_host2(const u32 r[WORDS * 2], const u32 a[WORDS],
                const u32 b[WORDS], Number2 f(const Number &a, const Number &b))
{
  const auto nr = Number2::load(r);
  const auto na = Number::load(a);
  const auto nb = Number::load(b);
  const auto n_got_r = f(na, nb);

  if (nr != n_got_r)
  {
    FAIL("Expected\n  ", nr, ", but got\n  ", n_got_r);
  }
}

const u32 WORDS = 8;
namespace instance1
{ // 14801559487980499405531442697394884941228070624928855492114078555480107039513
  const u32 a[WORDS] = BIG_INTEGER_CHUNKS8(0x20b962c2, 0xed033696, 0xa5384118, 0x06c215e1, 0xbb31704c, 0x6a92ae0e, 0x31f7b2f9, 0x1bf8bb19);
  // 11519749216843687781998751461425624010810819986485618779371694913180741450043
  const u32 b[WORDS] = BIG_INTEGER_CHUNKS8(0x1977f26e, 0x7eb3df21, 0x25ffd6e7, 0x558f2112, 0xc1358e32, 0x98c44536, 0x5528af35, 0xe0a0b93b);
  const u32 sum[WORDS] = BIG_INTEGER_CHUNKS8(0x3a315531, 0x6bb715b7, 0xcb3817ff, 0x5c5136f4, 0x7c66fe7f, 0x0356f344, 0x8720622e, 0xfc997454);
  const u32 sum_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x9cd06be, 0x8a85758e, 0x12e7d248, 0xdacfde97, 0x54331636, 0x899d82b3, 0x433e6c9b, 0x0c997453);
  const u32 sub[WORDS] = BIG_INTEGER_CHUNKS8(0x7417054, 0x6e4f5775, 0x7f386a30, 0xb132f4ce, 0xf9fbe219, 0xd1ce68d7, 0xdccf03c3, 0x3b5801de);
  const u32 sub_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x7417054, 0x6e4f5775, 0x7f386a30, 0xb132f4ce, 0xf9fbe219, 0xd1ce68d7, 0xdccf03c3, 0x3b5801de);
  const u32 prod[WORDS * 2] = BIG_INTEGER_CHUNKS16(0x3416fcf, 0x50f86d63, 0xafe17d87, 0x8859b777, 0x1de23d0a, 0xe34aed4a, 0x41b1a792, 0x16a9da55, 0xc7cc070c, 0x3811180f, 0x3a747983, 0x0905f4b1, 0x6a134051, 0x250dd7bf, 0xb3e436f8, 0x01282fc3);
  const u32 prod_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x264f37ba, 0x79c9af81, 0xd214d86f, 0x5e282b59, 0x8377f5bd, 0xb19cce9c, 0x0eb3ee7f, 0x2d53a6a3);
  const u32 a_square[WORDS * 2] = BIG_INTEGER_CHUNKS16(0x42edef0, 0x9f152e2d, 0x1921e182, 0xd29945c4, 0xad0e6750, 0xc2569051, 0xaac0c4af, 0x0d86c313, 0x614d56de, 0xa5d465aa, 0x3469dd8a, 0x519eba57, 0x0c67d0f1, 0x809701a1, 0xed5723c4, 0x4f2d8871);
  const u32 a_square_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x18556b1c, 0x7825e5c0, 0xf218f3a8, 0xef5898bb, 0xe3167b05, 0x04726854, 0xbb5cb44b, 0xffca00e4);
  const u32 pow_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x2eae6fc0, 0x51237495, 0xf2c9d498, 0x1b9a0446, 0xa1de8d70, 0x41693ffb, 0x9e53b3d5, 0x4944d650);
  const u32 a_inv_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x255227fb, 0x7605288b, 0x8f18bd82, 0x23b3e73a, 0x6c2ac926, 0xfdc281b8, 0x6e67667e, 0xf1c551cb);
  const u32 a_slr125[WORDS] = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x1, 0x05cb1617, 0x6819b4b5, 0x29c208c0, 0x3610af0d);

  TEST_CASE("Big number subtraction 1")
  {
    test_mont_kernel<WORDS>(sub, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_sub<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number addition 1")
  {
    test_mont_kernel<WORDS>(sum, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_add<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Fp addition 1")
  {
    test_mont_kernel<WORDS>(sum_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_add<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Fp subtraction 1")
  {
    test_mont_kernel<WORDS>(sub_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_sub<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number multiplication 1")
  {
    test_mont_kernel2<WORDS>(prod, a, b, [](u32 *r, const u32 *a, const u32 *b)
                             { bn_mul<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number square 1")
  {
    test_mont_kernel2<WORDS>(a_square, a, b, [](u32 *r, const u32 *a, const u32 *b)
                             { bn_square<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number shift logical right 1")
  {
    const u32 k[WORDS] = {125};
    test_mont_kernel<WORDS>(a_slr125, a, k, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_slr<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery multiplication 1")
  {
    // Here a, b are viewed as elements
    test_mont_kernel<WORDS>(prod_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_mul<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery square 1")
  {
    test_mont_kernel<WORDS>(a_square_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_square<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery power 1")
  {
    test_mont_kernel<WORDS>(pow_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_pow<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery inversion 1")
  {
    test_mont_kernel<WORDS>(a_inv_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_inv<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number addition 1 (host)")
  {
    test_host<WORDS>(sum, a, b, [](const Number &a, const Number &b)
                     { return a + b; });
  }

  TEST_CASE("Big number subtraction 1 (host)")
  {
    test_host<WORDS>(sub, a, b, [](const Number &a, const Number &b)
                     { return a - b; });
  }

  TEST_CASE("Big number multiplication 1 (host)")
  {
    test_host2<WORDS>(prod, a, b, [](const Number &a, const Number &b)
                      { return a * b; });
  }

  TEST_CASE("Big number square 1 (host)")
  {
    test_host2<WORDS>(a_square, a, b, [](const Number &a, const Number &b)
                      { return a.square(); });
  }

  TEST_CASE("Montgomery multiplication 1 (host)")
  {
    test_host<WORDS>(prod_mont, a, b, [](const Number &a, const Number &b)
                     {
        // Here a, b are viewd as elements. This is a break of abstraction.
        Element ea, eb;
        ea.n = a;
        eb.n = b;
        auto er = ea * eb;
        return er.n; });
  }

  TEST_CASE("Montgomery power 1 (host)")
  {
    test_host<WORDS>(pow_mont, a, b, [](const Number &a, const Number &b)
                     { 
        auto ea = Element::from_number(a);
        auto er = ea.pow(b);
        return er.n; });
  }
}

namespace instance2
{
  // 1340112579788283211323377730335072378553403695737973290727418574213178170393
  const u32 a[WORDS] = BIG_INTEGER_CHUNKS8(0x2f67a12, 0x3c531777, 0xeba1d665, 0xa77ff67f, 0x386a2378, 0x07e174b1, 0xedd8e224, 0x9bdb9c19);
  // 13637549481470042546599681774196962665467303419919163662308042186955339835302
  const u32 b[WORDS] = BIG_INTEGER_CHUNKS8(0x1e269458, 0x2b734095, 0x2c401491, 0xfcd8f52b, 0x1d78d889, 0x279ad499, 0x7fca3082, 0x283fa7a6);
  const u32 sum[WORDS] = BIG_INTEGER_CHUNKS8(0x211d0e6a, 0x67c6580d, 0x17e1eaf7, 0xa458ebaa, 0x55e2fc01, 0x2f7c494b, 0x6da312a6, 0xc41b43bf);
  const u32 sum_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x211d0e6a, 0x67c6580d, 0x17e1eaf7, 0xa458ebaa, 0x55e2fc01, 0x2f7c494b, 0x6da312a6, 0xc41b43bf);
  const u32 sub[WORDS] = BIG_INTEGER_CHUNKS8(0xe4cfe5ba, 0x10dfd6e2, 0xbf61c1d3, 0xaaa70154, 0x1af14aee, 0xe046a018, 0x6e0eb1a2, 0x739bf473);
  const u32 sub_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x1534342c, 0xf211770c, 0x77b2078a, 0x2c2859b1, 0x43253337, 0x5a0010a9, 0xb1f0a736, 0x639bf474);
  const u32 prod[WORDS * 2] = BIG_INTEGER_CHUNKS16(0x59549b, 0xc593ee33, 0x8e5b1b65, 0x76f910e4, 0x844b62b4, 0x2f5e719b, 0x8d9b6752, 0xd2e39bc5, 0xb1bcbbdf, 0x47296730, 0x553d24fe, 0x88c91554, 0x021bad2e, 0x7a92c9b8, 0xfb09258a, 0xa5628736);
  const u32 prod_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x80113a3, 0xa26fad56, 0x3a55196d, 0x46b1079f, 0x1aac1c6c, 0x08da138c, 0xb599e184, 0x2f022266);
  const u32 a_square[WORDS * 2] = BIG_INTEGER_CHUNKS16(0x8c737, 0x1e329eda, 0xb66c09cd, 0x75745fad, 0xe89eb541, 0x54ac2548, 0xfb730edd, 0xb1f1b7f0, 0xeb87f6de, 0x2e6c6398, 0xda1ae077, 0xd755c6b9, 0x33fa5656, 0x7b36cc9b, 0xa873b672, 0xb7f47a71);
  const u32 a_square_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x143448fe, 0xc7827a99, 0xdbd36c44, 0x522d5d26, 0xeb40cab9, 0x401b822e, 0x03c9dde8, 0xe152ae69);
  const u32 pow_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x14fdd058, 0x822cdb78, 0xbed1b37b, 0x2f2dd7e8, 0x120a1894, 0x6a0f5710, 0xdc6e186b, 0xb4db4a0c);
  const u32 a_inv_mont[WORDS] = BIG_INTEGER_CHUNKS8(0xb0e8de0, 0x565c86a3, 0xe394b20e, 0x076c44f4, 0x7ffd13b0, 0x0a3ac42e, 0xe3509394, 0x02623b92);
  const u32 a_slr125[WORDS] = BIG_INTEGER_CHUNKS8(0x0, 0x0, 0x0, 0x0, 0x17b3d091, 0xe298bbbf, 0x5d0eb32d, 0x3bffb3f9);

  TEST_CASE("Big number addition 2")
  {
    test_mont_kernel<WORDS>(sum, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_add<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number subtraction 2")
  {
    test_mont_kernel<WORDS>(sub, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_sub<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Fp addition 2")
  {
    test_mont_kernel<WORDS>(sum_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_add<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Fp subtraction 2")
  {
    test_mont_kernel<WORDS>(sub_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_sub<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number multiplication 2")
  {
    test_mont_kernel2<WORDS>(prod, a, b, [](u32 *r, const u32 *a, const u32 *b)
                             { bn_mul<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number square 2")
  {
    test_mont_kernel2<WORDS>(a_square, a, b, [](u32 *r, const u32 *a, const u32 *b)
                             { bn_square<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number shift logical right 2")
  {
    const u32 k[WORDS] = BIG_INTEGER_CHUNKS8(0, 0, 0, 0, 0, 0, 0, 125);
    test_mont_kernel<WORDS>(a_slr125, a, k, [](u32 *r, const u32 *a, const u32 *b)
                            { bn_slr<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery multiplication 2")
  {
    test_mont_kernel<WORDS>(prod_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_mul<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery square 2")
  {
    test_mont_kernel<WORDS>(a_square_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_square<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery power 2")
  {
    test_mont_kernel<WORDS>(pow_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_pow<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Montgomery inversion 2")
  {
    test_mont_kernel<WORDS>(a_inv_mont, a, b, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_inv<<<1, 1>>>(r, a, b); });
  }

  TEST_CASE("Big number addition 2 (host)")
  {
    test_host<WORDS>(sum, a, b, [](const Number &a, const Number &b)
                     { return a + b; });
  }

  TEST_CASE("Big number subtraction 2 (host)")
  {
    test_host<WORDS>(sub, a, b, [](const Number &a, const Number &b)
                     { return a - b; });
  }

  TEST_CASE("Big number multiplication 2 (host)")
  {
    test_host2<WORDS>(prod, a, b, [](const Number &a, const Number &b)
                      { return a * b; });
  }

  TEST_CASE("Big number square 2 (host)")
  {
    test_host2<WORDS>(a_square, a, b, [](const Number &a, const Number &b)
                      { return a.square(); });
  }

  TEST_CASE("Montgomery multiplication 2 (host)")
  {
    test_host<WORDS>(prod_mont, a, b, [](const Number &a, const Number &b)
                     {
        // Here a, b are viewd as elements. This is a break of abstraction.
        Element ea, eb;
        ea.n = a;
        eb.n = b;
        auto er = ea * eb;
        return er.n; });
  }

  TEST_CASE("Montgomery power 2 (host)")
  {
    test_host<WORDS>(pow_mont, a, b, [](const Number &a, const Number &b)
                     { 
        auto ea = Element::from_number(a);
        auto er = ea.pow(b);
        return er.n; });
  }
}

TEST_CASE("Convert to and from Montgomery")
{
  const u32 x[WORDS] = BIG_INTEGER_CHUNKS8(0x14021876, 0x4dbe5ba4, 0xabcc4ca3, 0x4be34308, 0x508480a4, 0xcb5d23b7, 0xdd6e0720, 0xb40134fb);
  const u32 x_mont[WORDS] = BIG_INTEGER_CHUNKS8(0x14ea7d56, 0xb86c7f42, 0xc5fbb651, 0xe30ef1c5, 0xa93ab5ae, 0xa99221ab, 0xe00c9f14, 0xb594b3f0);

  test_mont_kernel<WORDS>(x_mont, x, x, [](u32 *r, const u32 *a, const u32 *b)
                          { convert_to_mont<<<1, 1>>>(r, a, b); });
  test_mont_kernel<WORDS>(x, x_mont, x_mont, [](u32 *r, const u32 *a, const u32 *b)
                          { convert_from_mont<<<1, 1>>>(r, a, b); });
}

TEST_CASE("Fp negation")
{
  const u32 x[WORDS] = {0, 0, 0, 0, 0, 0, 0, 0};
  test_mont_kernel<WORDS>(x, x, x, [](u32 *r, const u32 *a, const u32 *b)
                          { mont_neg<<<1, 1>>>(r, a, b); });
}

TEST_CASE("Convert to and from Montgomery (host)")
{
  std::srand(std::time(nullptr));
  auto e = Element::host_random();
  auto n = e.to_number();
  auto e1 = Element::from_number(n);
  REQUIRE(e1 == e);
}

namespace instance3 {
  const u32 x[WORDS] = BIG_INTEGER_CHUNKS8(0x31ed3847, 0xcfae97c3, 0x94d0daed, 0xbaa91e44, 0xaa4cf8c3, 0x67de9f72, 0x53222181, 0x6cc4902a);
  const u32 y[WORDS] = BIG_INTEGER_CHUNKS8(0x3567be11, 0xea6cb86f, 0x5be9dde9, 0x2bb995ae, 0x4e9d604b, 0xe30cf8cf, 0xaf50a0b4, 0x322ac735);
  const u32 r_sub[WORDS] = BIG_INTEGER_CHUNKS8(0x5d4e171b, 0xa7a51fa7, 0xa9878871, 0x91f23950, 0xac176908, 0x784487c5, 0x2b956bf5, 0x1a99c8f7);
  const u32 r_add[WORDS] = BIG_INTEGER_CHUNKS8(0x68c5973, 0xf7b80fdf, 0x801a2d69, 0xe3600338, 0xa882887e, 0x5778b71f, 0x7aaed70d, 0xbeef575d);
  const u32 x_mod_m[WORDS] = BIG_INTEGER_CHUNKS8(0x188e9d4, 0xee7cf799, 0xdc809537, 0x3927c5e7, 0x8219107a, 0xee252ee1, 0x0f402bed, 0x7cc49029);

  TEST_CASE("Subtraction modulo mm2")
  {
    test_mont_kernel<WORDS>(r_sub, x, y, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_sub_mm2<<<1, 1>>>(r, a, b); });
  }
  TEST_CASE("Addition modulo mm2")
  {
    test_mont_kernel<WORDS>(r_add, x, y, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_add_mm2<<<1, 1>>>(r, a, b); });
  }
  TEST_CASE("Modulo m")
  {
    test_mont_kernel<WORDS>(x_mod_m, x, y, [](u32 *r, const u32 *a, const u32 *b)
                            { mont_modulo_m<<<1, 1>>>(r, a, b); });
  }
}
