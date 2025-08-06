#pragma once

#include "../../mont/src/bls12381_fq.cuh"
#include "../../mont/src/bls12381_fr.cuh"
#include "curve_xyzz.cuh"

namespace bls12381
{
  using Element = bls12381_fq::Element; // base field for curves
  using BaseNumber = mont::Number<Element::LIMBS>; // number for base field
  using Field = bls12381_fr::Element; // field for scalars
  using Number = mont::Number<Field::LIMBS>; // number for scalar field

  namespace device_constants
  {
    const __device__ Element b = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0x9d64551, 0x3d83de7e, 0x8ec9733b, 0xbf78ab2f, 0xb1d37ebe, 0xe6ba24d7, 0x478fe97a, 0x6b0a807f, 0x53cc0032, 0xfc34000a, 0xaa270000, 0x000cfff3)));
    const __device__ Element b3 = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0x381be09, 0x7f0bb4e1, 0x6140b1fc, 0xfb1e54b7, 0xb10330b7, 0xc0a95bc6, 0x6f7ee9ce, 0x4a6e8b59, 0xdcb8009a, 0x43480020, 0x44760000, 0x0027552e)));
    const __device__ Element a = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
  }

  namespace host_constants
  {
    constexpr Element b = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0x9d64551, 0x3d83de7e, 0x8ec9733b, 0xbf78ab2f, 0xb1d37ebe, 0xe6ba24d7, 0x478fe97a, 0x6b0a807f, 0x53cc0032, 0xfc34000a, 0xaa270000, 0x000cfff3)));
    constexpr Element b3 = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0x381be09, 0x7f0bb4e1, 0x6140b1fc, 0xfb1e54b7, 0xb10330b7, 0xc0a95bc6, 0x6f7ee9ce, 0x4a6e8b59, 0xdcb8009a, 0x43480020, 0x44760000, 0x0027552e)));
    constexpr Element a = Element(BaseNumber(BIG_INTEGER_CHUNKS12(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
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
}
