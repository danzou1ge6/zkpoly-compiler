#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <iostream>

#include "../src/bn254.cuh"

using mont::u32;
using namespace bn254;

__global__ void to_affine_kernel(PointAffine *pr, const Point *p)
{
  *pr = p->to_affine();
}

__global__ void from_affine_kernel(Point *pr, const PointAffine *p)
{
  
  *pr = p->to_point();
}

__global__ void is_on_curve_kernel(bool *r, const Point *p)
{
  
  *r = p->is_on_curve();
}

__global__ void is_on_curve_kernel(bool *r, const PointAffine *p)
{
  
  *r = p->is_on_curve();
}

__global__ void self_add_kernel(Point *pr, const Point *p)
{
  
  *pr = p->self_add();
}

__global__ void add_kernel(Point *pr, const Point *pa, const Point *pb)
{
  
  *pr = *pa + *pb;
}

__global__ void add_kernel(Point *pr, const Point *pa, const PointAffine *pb)
{
  
  *pr = *pa + *pb;
}

__global__ void eq_kernel(bool *r, const Point *pa, const Point *pb)
{
  
  *r = *pa == *pb;
}

__global__ void eq_kernel(bool *r, const PointAffine *pa, const PointAffine *pb)
{
  
  *r = *pa == *pb;
}

__global__ void multiple_kernel(Point *r, const Point *p, u32 n)
{
  
  *r = p->multiple(n);
}

void to_affine(PointAffine *pr, const Point *p)
{
  to_affine_kernel<<<1, 1>>>(pr, p);
}

void from_affine(Point *pr, const PointAffine *p)
{
  from_affine_kernel<<<1, 1>>>(pr, p);
}

void is_on_curve(bool *r, const Point *p)
{
  is_on_curve_kernel<<<1, 1>>>(r, p);
}

void is_on_curve(bool *r, const PointAffine *p)
{
  is_on_curve_kernel<<<1, 1>>>(r, p);
}

void self_add(Point *pr, const Point *p)
{
  self_add_kernel<<<1, 1>>>(pr, p);
}

void add(Point *pr, const Point *pa, const Point *pb)
{
  add_kernel<<<1, 1>>>(pr, pa, pb);
}

void add(Point *pr, const Point *pa, const PointAffine *pb)
{
  add_kernel<<<1, 1>>>(pr, pa, pb);
}

void eq(bool *r, const Point *pa, const Point *pb)
{
  eq_kernel<<<1, 1>>>(r, pa, pb);
}

void eq(bool *r, const PointAffine *pa, const PointAffine *pb)
{
  eq_kernel<<<1, 1>>>(r, pa, pb);
}

void multiple(Point *r, const Point *p, const u32 *n)
{
  multiple_kernel<<<1, 1>>>(r, p, *n);
}


bool elements_lt_2m(Point &p)
{
  return p.x.lt_2m() && p.y.lt_2m() && p.zz.lt_2m() && p.zzz.lt_2m();
}

template <typename R, typename T>
R launch_kernel1(T &a, void kernel(R *r, const T *a))
{
  R *dr;
  T *dt;
  cudaMalloc(&dt, sizeof(T));
  cudaMalloc(&dr, sizeof(R));
  cudaMemcpy(dt, &a, sizeof(T), cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, dt);

  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  R r;
  cudaMemcpy(&r, dr, sizeof(R), cudaMemcpyDeviceToHost);
  return r;
}

template <typename R, typename T1, typename T2>
R launch_kernel2(T1 &a, T2 &b, void kernel(R *r, const T1 *t1, const T2 *t2))
{
  R *dr;
  T1 *dt1;
  T2 *dt2;
  cudaMalloc(&dt1, sizeof(T1));
  cudaMalloc(&dt2, sizeof(T2));
  cudaMalloc(&dr, sizeof(R));
  cudaMemcpy(dt1, &a, sizeof(T1), cudaMemcpyHostToDevice);
  cudaMemcpy(dt2, &b, sizeof(T2), cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  kernel(dr, dt1, dt2);

  if (err != cudaSuccess)
    FAIL("CUDA error: ", cudaGetErrorString(err));

  R r;
  cudaMemcpy(&r, dr, sizeof(R), cudaMemcpyDeviceToHost);
  return r;
}

// All in Montgomery representation
// x = 17588249314949534242365104770887097184252481281543984788935084765326940834124
// x_mont = 15118613892300968952598744084540872064370112947361953331248357527216742275022
const u32 p1_x[8] = BIG_INTEGER_CHUNKS8(0x216cd50c, 0x64543f25, 0x23ea3cd2, 0xb993a6dd, 0x83f1909e, 0xfa432c22, 0xb05c2787, 0x423517ce);
// y = 12357679330807769286224777493623770042961632047811765801488126600704678244454
// y_mont = 15153950746392142037506258921283735096067572820412603689246231231476536618320
const u32 p1_y[8] = BIG_INTEGER_CHUNKS8(0x2180d509, 0x28461b27, 0xa030a1a7, 0x99911e11, 0xba9ce74f, 0xb488be66, 0x48fee199, 0x2592ad50);
// x = 12608279712251949873380244143875219782586201208786743136308610799954805067194
// x_mont = 19167055408201548293887119995526340569960432486177240658994856871947018094871
const u32 p2_x[8] = BIG_INTEGER_CHUNKS8(0x2a602b3e, 0x1b4f3db1, 0xbe03a8be, 0x55ed2444, 0xd8128f84, 0x59d8a382, 0x20aca79d, 0xf58ac917);
// y = 16073978196211952434062052344575429549195455908288056916460411561436103132293
// y_mont = 14394749309957023594572365872892788643083852397597524596381726046325896367924
const u32 p2_y[8] = BIG_INTEGER_CHUNKS8(0x1fd323ae, 0xc7ed1080, 0xf3890048, 0x493e9f2f, 0x243c445d, 0xd7b99618, 0xaefe84ff, 0xa1dbcf34);

// x = 1429041557224557752552626303000099472202069419966672611522353279561688898742
const u32 pr_x[8] = BIG_INTEGER_CHUNKS8(0x6790889, 0x79be73c4, 0x0f7643bb, 0x661c0f71, 0xd9ab80b1, 0x7e230fc0, 0xea2b2797, 0xa3758869);
// y = 18492289853934875728433615546364519615101151049369325523076021792381188176633
const u32 pr_y[8] = BIG_INTEGER_CHUNKS8(0x1c74dd57, 0xe8850b6f, 0x089652a2, 0x98192463, 0x1a12ecaa, 0x3494b476, 0x83832570, 0xaea7b267);

// = p1 + 100 * p2
// x = 5925225539943078052011782854402534874541319543666152803615178215392564222781
const u32 pr100_x[8] = BIG_INTEGER_CHUNKS8(0x26387a79, 0x6b4afb4a, 0x7730093f, 0x18ea31b2, 0x9b9528d4, 0x4fe58295, 0x6194b7b1, 0x7bfd53c3);
// y = 19733743001297787355155826057874421101503607786210974480396698306505458042500
const u32 pr100_y[8] = BIG_INTEGER_CHUNKS8(0x20010096, 0xcf34d5a7, 0xc4a839e6, 0xdfb05fe4, 0x0632f1e6, 0x58f8a021, 0x3e949dcf, 0x53be571e);

// = 128 * p1
// x = 2033540048204564502430637783667346903636294967913472962188082207965010960204
const u32 p1r128_x[8] = BIG_INTEGER_CHUNKS8(0x1dc189b8, 0xd831b545, 0x7fe9b79c, 0x412a9187, 0xfa6dc93c, 0x1fdecb39, 0xdb1c59af, 0xbda40f9c);
// y = 13931799731996164040353389921479055180035863615375143429273578105415422924134
const u32 p1r128_y[8] = BIG_INTEGER_CHUNKS8(0x1ff9a5bd, 0xe2e392fd, 0x26eac79d, 0x37e849ee, 0x857bc9af, 0x0c2c3a75, 0x2bcb0276, 0x436b9128);

PointAffine load_affine(const u32 x_data[8], const u32 y_data[8])
{
  auto x = Element::load(x_data);
  auto y = Element::load(y_data);
  return PointAffine(x, y);
}

// TEST_CASE("Test Element::lt_2m")
// {
//   const u32 e1[8] = BIG_INTEGER_CHUNKS8(0xd2345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678);
//   const u32 e2[8] = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0x70a08b6d, 0x0302b0bb, 0x2f02d522, 0xd0e3951a, 0x7841182d, 0xb0f9fa8e);  // eq
//   const u32 e3[8] = BIG_INTEGER_CHUNKS8(0x60c89ce5, 0xc2634053, 0xffffffff, 0x0302b0bb, 0x2f02d522, 0xd0e3951a, 0x7841182d, 0xb0f9fa8e);

//   const u32 e[8] = BIG_INTEGER_CHUNKS8(0x30644e72, 0xe131a029, 0xb85045b6, 0x8181585d, 0x97816a91, 0x6871ca8d, 0x3c208c16, 0xd87cfd47);

//   REQUIRE(Element::load(e1).lt_2m() == false);
//   REQUIRE(Element::load(e2).lt_2m() == false);
//   REQUIRE(Element::load(e3).lt_2m() == false);

//   REQUIRE(Element::load(e).lt_2m() == true);

//   Point p(Element::load(e1), Element::load(e2), Element::load(e3), Element::load(e));
//   REQUIRE(p.is_elements_lt_2m() == false);
// }

void test_affine_projective_back_and_forth(const u32 x_data[8], const u32 y_data[8])
{
  auto affine = load_affine(x_data, y_data);

  auto projective = launch_kernel1(affine, from_affine);
  auto affine2 = launch_kernel1(projective, to_affine);
  auto projective2 = launch_kernel1(affine2, from_affine);

  REQUIRE(launch_kernel1(projective, is_on_curve));
  REQUIRE(launch_kernel1(projective2, is_on_curve));
  REQUIRE(launch_kernel1(affine2, is_on_curve));

  REQUIRE(launch_kernel2(affine, affine2, eq));
  REQUIRE(launch_kernel2(projective, projective2, eq));
}

void test_affine_projective_back_and_forth_host(const u32 x_data[8], const u32 y_data[8])
{
  auto affine = load_affine(x_data, y_data);
  auto projective = affine.to_point();
  auto affine2 = projective.to_affine();
  auto projective2 = affine2.to_point();

  REQUIRE(affine == affine2);
  REQUIRE(projective == projective2);
}

TEST_CASE("On curve")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto pr = load_affine(pr_x, pr_y);

  REQUIRE(launch_kernel1(p1, is_on_curve));
  REQUIRE(launch_kernel1(p2, is_on_curve));
  REQUIRE(launch_kernel1(pr, is_on_curve));
}

TEST_CASE("On curve (host)")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto pr = load_affine(pr_x, pr_y);

  REQUIRE(p1.is_on_curve());
  REQUIRE(p2.is_on_curve());
  REQUIRE(pr.is_on_curve());
}

TEST_CASE("Affine/Projective back and forth 1")
{
  test_affine_projective_back_and_forth(p1_x, p1_y);
}

TEST_CASE("Affine/Projective back and forth 2")
{
  test_affine_projective_back_and_forth(p2_x, p2_y);
}

TEST_CASE("Affine/Projective back and forth 1 (host)")
{
  test_affine_projective_back_and_forth_host(p1_x, p1_y);
}

TEST_CASE("Affine/Projective back and forth 2 (host)")
{
  test_affine_projective_back_and_forth_host(p2_x, p2_y);
}

TEST_CASE("Point addition commutative")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto p1p = launch_kernel1(p1, from_affine);
  auto p2p = launch_kernel1(p2, from_affine);

  auto sum1 = launch_kernel2(p1p, p2p, add);
  auto sum2 = launch_kernel2(p2p, p1p, add);

  REQUIRE(launch_kernel1(sum1, is_on_curve));
  REQUIRE(launch_kernel1(sum2, is_on_curve));

  REQUIRE(launch_kernel2(sum1, sum2, eq));
}

TEST_CASE("Point addition commutative (host)")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto p1p = p1.to_point();
  auto p2p = p2.to_point();

  auto sum1 = p1p + p2p;
  auto sum2 = p2p + p1p;

  REQUIRE(sum1.is_on_curve());
  REQUIRE(sum2.is_on_curve());

  REQUIRE(sum1 == sum2);
}

TEST_CASE("Point-PointAffine accumulation")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto accu = launch_kernel1(p1, from_affine);

  for (u32 i = 0; i < 100; i ++)
  {
    accu = launch_kernel2(accu, p2, add);
    REQUIRE(elements_lt_2m(accu));
  }
  auto accu_affine = launch_kernel1(accu, to_affine);

  auto pr = load_affine(pr100_x, pr100_y);
  REQUIRE(launch_kernel2(accu_affine, pr, eq));
}

TEST_CASE("Point-PointAffine accumulation (host)")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto accu = p1.to_point();

  for (u32 i = 0; i < 100; i ++)
  {
    accu = accu + p2;
    REQUIRE(elements_lt_2m(accu));
  }
  auto accu_affine = accu.to_affine();

  auto pr = load_affine(pr100_x, pr100_y);
  REQUIRE(accu_affine == pr);
}

TEST_CASE("Point double")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto accu = launch_kernel1(p1, from_affine);
  
  for (u32 i = 0; i < 7; i ++)
  {
    accu = launch_kernel1(accu, self_add);
    REQUIRE(elements_lt_2m(accu));
  }
  auto accu_affine = launch_kernel1(accu, to_affine);
  
  auto pr = load_affine(p1r128_x, p1r128_y);
  REQUIRE(launch_kernel2(accu_affine, pr, eq));
}

TEST_CASE("Point double (host)")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto accu = p1.to_point();

  for (u32 i = 0; i < 7; i ++)
  {
    accu = accu.self_add();
    REQUIRE(elements_lt_2m(accu));
  }
  auto accu_affine = accu.to_affine();

  auto pr = load_affine(p1r128_x, p1r128_y);
  REQUIRE(accu_affine == pr);
}

TEST_CASE("Identity")
{
  auto pi = Point::identity();
  auto p1 = load_affine(p1_x, p1_y);
  auto sum = launch_kernel2(pi, p1, add);
  auto pr = launch_kernel1(sum, to_affine);
  REQUIRE(launch_kernel2(p1, pr, eq));
}

TEST_CASE("Identity (host)")
{
  auto pi = Point::identity();
  auto p1 = load_affine(p1_x, p1_y);
  auto sum = pi + p1;
  auto pr = sum.to_affine();
  REQUIRE(p1 == pr);
}

TEST_CASE("Point accumulation")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto accu = launch_kernel1(p1, from_affine);
  auto p2p = launch_kernel1(p2, from_affine);

  for (u32 i = 0; i < 100; i ++)
  {
    accu = launch_kernel2(accu, p2p, add);
    REQUIRE(elements_lt_2m(accu));
  }
  auto accu_affine = launch_kernel1(accu, to_affine);

  auto pr = load_affine(pr100_x, pr100_y);
  REQUIRE(launch_kernel2(accu_affine, pr, eq));
}

TEST_CASE("Point accumulation (host)")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto accu = p1.to_point();
  auto p2p = p2.to_point();

  for (u32 i = 0; i < 100; i ++)
  {
    accu = accu + p2p;
    REQUIRE(elements_lt_2m(accu));
  }
  auto accu_affine = accu.to_affine();

  auto pr = load_affine(pr100_x, pr100_y);
  REQUIRE(accu_affine == pr);
}

TEST_CASE("Point-Point addition")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto pr = load_affine(pr_x, pr_y);
  auto p1p = launch_kernel1(p1, from_affine);
  auto p2p = launch_kernel1(p2, from_affine);
  auto prp = launch_kernel1(pr, from_affine);

  auto sum1 = launch_kernel2(p1p, p2p, add);
  REQUIRE(elements_lt_2m(sum1));
  REQUIRE(launch_kernel2(sum1, prp, eq));
}

TEST_CASE("Point-Point addition (host)")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto pr = load_affine(pr_x, pr_y);
  auto p1p = p1.to_point();
  auto p2p = p2.to_point();
  auto prp = pr.to_point();

  auto sum1 = p1p + p2p;
  REQUIRE(elements_lt_2m(sum1));
  REQUIRE(sum1 == prp);
}

TEST_CASE("Point-PointAffine addition")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto pr = load_affine(pr_x, pr_y);
  auto p1p = launch_kernel1(p1, from_affine);
  auto prp = launch_kernel1(pr, from_affine);

  auto sum1 = launch_kernel2(p1p, p2, add);
  REQUIRE(elements_lt_2m(sum1));
  REQUIRE(launch_kernel2(sum1, prp, eq));
}

TEST_CASE("Point-PointAffine addition (host)")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto pr = load_affine(pr_x, pr_y);
  auto p1p = p1.to_point();
  auto prp = pr.to_point();

  auto sum1 = p1p + p2;
  REQUIRE(elements_lt_2m(sum1));
  REQUIRE(sum1 == prp);
}

TEST_CASE("Point-PointAffine addition equivalent")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto p1p = launch_kernel1(p1, from_affine);
  auto p2p = launch_kernel1(p2, from_affine);

  auto sum1 = launch_kernel2(p2p, p1, add);
  auto sum2 = launch_kernel2(p2p, p1p, add);

  REQUIRE(launch_kernel1(sum1, is_on_curve));
  REQUIRE(launch_kernel1(sum2, is_on_curve));

  REQUIRE(launch_kernel2(sum1, sum2, eq));
}

TEST_CASE("Doubling equivalent")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p1p = launch_kernel1(p1, from_affine);

  auto sum1 = launch_kernel2(p1p, p1p, add);
  auto sum2 = launch_kernel1(p1p, self_add);

  REQUIRE(launch_kernel1(sum1, is_on_curve));
  REQUIRE(launch_kernel1(sum2, is_on_curve));

  REQUIRE(launch_kernel2(sum1, sum2, eq));
}

TEST_CASE("Reproducible")
{
  auto p1 = load_affine(p1_x, p1_y);
  auto p2 = load_affine(p2_x, p2_y);
  auto p1p = launch_kernel1(p1, from_affine);

  auto sum1 = launch_kernel2(p1p, p2, add);
  auto sum2 = launch_kernel2(p1p, p2, add);

  REQUIRE(launch_kernel1(sum1, is_on_curve));
  REQUIRE(launch_kernel1(sum2, is_on_curve));

  REQUIRE(launch_kernel2(sum1, sum2, eq));

}

