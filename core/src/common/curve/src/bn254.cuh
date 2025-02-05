#ifndef BN254_H
#define BN254_H

#include "../../mont/src/field_impls.cuh"
#include "curve_xyzz.cuh"

namespace bn254
{
  using bn254_fq::Element; // base for curve
  using Number = mont::Number<Element::LIMBS>;
  typedef bn254_fr::Element Field; // field for scalar

  namespace device_constants
  {
    constexpr __device__ Element b = Element(Number(BIG_INTEGER_CHUNKS8(
        0x2a1f6744, 0xce179d8e, 0x334bea4e, 0x696bd284, 0x1f6ac17a, 0xe15521b9, 0x7a17caa9, 0x50ad28d7)));
    constexpr __device__ Element b3 = Element(Number(BIG_INTEGER_CHUNKS8(
        0x1d9598e8, 0xa7e39857, 0x2943337e, 0x3940c6d1, 0x2f3d6f4d, 0xd31bd011, 0xf60647ce, 0x410d7ff7)));
    constexpr __device__ Element a = Element(Number(BIG_INTEGER_CHUNKS8(
        0, 0, 0, 0, 0, 0, 0, 0)));
  }

  namespace host_constants
  {
    constexpr Element b = Element(Number(BIG_INTEGER_CHUNKS8(
        0x2a1f6744, 0xce179d8e, 0x334bea4e, 0x696bd284, 0x1f6ac17a, 0xe15521b9, 0x7a17caa9, 0x50ad28d7)));
    constexpr Element b3 = Element(Number(BIG_INTEGER_CHUNKS8(
        0x1d9598e8, 0xa7e39857, 0x2943337e, 0x3940c6d1, 0x2f3d6f4d, 0xd31bd011, 0xf60647ce, 0x410d7ff7)));
    constexpr Element a = Element(Number(BIG_INTEGER_CHUNKS8(
        0, 0, 0, 0, 0, 0, 0, 0)));
  }

  struct Params
  {

    static constexpr __device__ __host__ __forceinline__
        Element
        a()
    {
#ifdef __CUDA_ARCH__
      return device_constants::a;
#else
      return host_constants::a;
#endif
    }

    static constexpr __device__ __host__ __forceinline__
        Element
        b()
    {
#ifdef __CUDA_ARCH__
      return device_constants::b;
#else
      return host_constants::b;
#endif
    }

    static __device__ __host__ __forceinline__
        Element
        b3()
    {
#ifdef __CUDA_ARCH__
      return device_constants::b3;
#else
      return host_constants::b3;
#endif
    }

    static constexpr __device__ __host__ __forceinline__
        bool
        allow_lazy_modulo()
    {
        return true;
    }
  };

  using Point = curve::EC<Params, Element>::PointXYZZ;
  using PointAffine = curve::EC<Params, Element>::PointAffine;
//   using PointEager = curve::EC<Params<false>, Element>::PointXYZZ;
//   using PointAffineEager = curve::EC<Params<false>, Element>::PointAffine;

//   __host__ __device__ PointEager eager_from_lazy(const Point &lazy)
//   {
//     auto lazy_n = lazy.normalized();
//     return PointEager(lazy_n.x, lazy_n.y, lazy_n.zz, lazy_n.zzz);
//   }

//   __host__ __device__ PointAffineEager eager_from_lazy(const PointAffine &lazy)
//   {
//     return PointAffineEager(lazy.x, lazy.y);
//   }

//   __host__ __device__ Point lazy_from_eager(const PointEager &eager)
//   {
//     return Point(eager.x, eager.y, eager.zz, eager.zzz);
//   }

//   __host__ __device__ PointAffine lazy_from_eager(const PointAffineEager &eager)
//   {
//     return PointAffine(eager.x, eager.y);
//   }
}

#endif