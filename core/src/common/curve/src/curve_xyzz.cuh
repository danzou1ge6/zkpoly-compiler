/*
The XYZZ representation of the curve
we copied the following formulas from https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
for quick reference, as the website is not always available:

XYZZ coordinates for short Weierstrass curves
An elliptic curve in short Weierstrass form [more information] has parameters a b and coordinates x y satisfying the following equations:
  y^2=x^3+a*x+b
XYZZ coordinates [database entry] represent x y as X Y ZZ ZZZ satisfying the following equations:

  x=X/ZZ
  y=Y/ZZZ
  ZZ3=ZZZ2

Best operation counts
Smallest multiplication counts assuming I=100M, S=1M, *param=0M, add=0M, *const=0M:
14M for addition: 12M+2S.
10M for addition with Z2=1: 8M+2S.
6M for addition with Z1=1 and Z2=1: 4M+2S.
14M for readdition: 12M+2S after 12M+2S.
10M for readdition with Z2=1: 8M+2S after 8M+2S.
6M for readdition with Z1=1 and Z2=1: 4M+2S after 4M+2S.
10M for doubling: 6M+4S.
7M for doubling with Z1=1: 4M+3S.
104M for scaling: 1I+3M+1S.
Smallest multiplication counts assuming I=100M, S=0.8M, *param=0M, add=0M, *const=0M:
13.6M for addition: 12M+2S.
9.6M for addition with Z2=1: 8M+2S.
5.6M for addition with Z1=1 and Z2=1: 4M+2S.
13.6M for readdition: 12M+2S after 12M+2S.
9.6M for readdition with Z2=1: 8M+2S after 8M+2S.
5.6M for readdition with Z1=1 and Z2=1: 4M+2S after 4M+2S.
9.2M for doubling: 6M+4S.
6.4M for doubling with Z1=1: 4M+3S.
103.8M for scaling: 1I+3M+1S.
Smallest multiplication counts assuming I=100M, S=0.67M, *param=0M, add=0M, *const=0M:
13.34M for addition: 12M+2S.
9.34M for addition with Z2=1: 8M+2S.
5.34M for addition with Z1=1 and Z2=1: 4M+2S.
13.34M for readdition: 12M+2S after 12M+2S.
9.34M for readdition with Z2=1: 8M+2S after 8M+2S.
5.34M for readdition with Z1=1 and Z2=1: 4M+2S after 4M+2S.
8.68M for doubling: 6M+4S.
6.01M for doubling with Z1=1: 4M+3S.
103.67M for scaling: 1I+3M+1S.
Summary of all explicit formulas
Operation	Assumptions	Cost	Readdition cost
addition	ZZ1=1 and ZZZ1=1 and ZZ2=1 and ZZZ2=1	4M + 2S	4M + 2S
addition	ZZ2=1 and ZZZ2=1	8M + 2S	8M + 2S
addition		12M + 2S	12M + 2S
doubling	ZZ1=1 and ZZZ1=1	4M + 3S
doubling		6M + 4S + 1*a
scaling		1I + 3M + 1S
Explicit formulas for addition
The "mmadd-2008-s" addition formulas [database entry; Sage verification script; Sage output; three-operand code]:
Assumptions: ZZ1=1 and ZZZ1=1 and ZZ2=1 and ZZZ2=1.
Cost: 4M + 2S + 6add + 1*2.
Source: 2008 Sutherland.
Explicit formulas:
      P = X2-X1
      R = Y2-Y1
      PP = P2
      PPP = P*PP
      Q = X1*PP
      X3 = R2-PPP-2*Q
      Y3 = R*(Q-X3)-Y1*PPP
      ZZ3 = PP
      ZZZ3 = PPP
The "madd-2008-s" addition formulas [database entry; Sage verification script; Sage output; three-operand code]:
Assumptions: ZZ2=1 and ZZZ2=1.
Cost: 8M + 2S + 6add + 1*2.
Source: 2008 Sutherland.
Explicit formulas:
      U2 = X2*ZZ1
      S2 = Y2*ZZZ1
      P = U2-X1
      R = S2-Y1
      PP = P2
      PPP = P*PP
      Q = X1*PP
      X3 = R2-PPP-2*Q
      Y3 = R*(Q-X3)-Y1*PPP
      ZZ3 = ZZ1*PP
      ZZZ3 = ZZZ1*PPP
The "add-2008-s" addition formulas [database entry; Sage verification script; Sage output; three-operand code]:
Cost: 12M + 2S + 6add + 1*2.
Source: 2008 Sutherland.
Explicit formulas:
      U1 = X1*ZZ2
      U2 = X2*ZZ1
      S1 = Y1*ZZZ2
      S2 = Y2*ZZZ1
      P = U2-U1
      R = S2-S1
      PP = P2
      PPP = P*PP
      Q = U1*PP
      X3 = R2-PPP-2*Q
      Y3 = R*(Q-X3)-S1*PPP
      ZZ3 = ZZ1*ZZ2*PP
      ZZZ3 = ZZZ1*ZZZ2*PPP
Explicit formulas for doubling
The "mdbl-2008-s-1" doubling formulas [database entry; Sage verification script; Sage output; three-operand code]:
Assumptions: ZZ1=1 and ZZZ1=1.
Cost: 4M + 3S + 4add + 2*2 + 1*3.
Source: 2008 Sutherland.
Explicit formulas:
      U = 2*Y1
      V = U2
      W = U*V
      S = X1*V
      M = 3*X12+a
      X3 = M2-2*S
      Y3 = M*(S-X3)-W*Y1
      ZZ3 = V
      ZZZ3 = W
The "dbl-2008-s-1" doubling formulas [database entry; Sage verification script; Sage output; three-operand code]:
Cost: 6M + 4S + 1*a + 4add + 2*2 + 1*3.
Source: 2008 Sutherland.
Explicit formulas:
      U = 2*Y1
      V = U2
      W = U*V
      S = X1*V
      M = 3*X12+a*ZZ12
      X3 = M2-2*S
      Y3 = M*(S-X3)-W*Y1
      ZZ3 = V*ZZ1
      ZZZ3 = W*ZZZ1
Explicit formulas for tripling
Explicit formulas for differential addition
Explicit formulas for differential addition and doubling
Explicit formulas for scaling
The "z" scaling formulas [database entry; Sage verification script; Sage output; three-operand code]:
Cost: 1I + 3M + 1S + 0add.
Explicit formulas:
      A = 1/ZZZ1
      B = (ZZ1*A)2
      X3 = X1*B
      Y3 = Y1*A
      ZZ3 = 1
      ZZZ3 = 1
*/

#pragma once

#include "../../mont/src/field.cuh"
#include <iostream>

#ifdef __CUDA_ARCH__
#define likely(x) (__builtin_expect((x), 1))
#define unlikely(x) (__builtin_expect((x), 0))
#else
#define likely(x) (x) [[likely]]
#define unlikely(x) (x) [[unlikely]]
#endif 

namespace curve
{
    using mont::u32;
    using mont::usize;

    template <class Params, class Element>
    struct EC {
        struct PointXYZZ;
        struct PointAffine;

        // we assume that no points in pointaffine are identity
        struct PointAffine {
            static const usize N_WORDS = 2 * Element::LIMBS;

            Element x, y;

            friend std::ostream& operator<<(std::ostream &os, const PointAffine &p) {
                os << "{\n";
                os << "  .x = " << p.x << ",\n";
                os << "  .y = " << p.y << ",\n";
                os << "}";
                return os;
            }

            friend std::istream& operator>>(std::istream &is, PointAffine &p) {
                is >> p.x >> p.y;
                return is;
            }

            __host__ __device__ __forceinline__ PointAffine() {}
            __host__ __device__ __forceinline__ PointAffine(Element x, Element y) : x(x), y(y) {}

            __device__ __host__ __forceinline__ PointAffine neg() const & {
                return PointAffine(x, y.neg());
            }

            static __host__ __device__ __forceinline__ PointAffine load(const u32 *p) {
                auto x = Element::load(p);
                auto y = Element::load(p + Element::LIMBS);
                return PointAffine(x, y);
            }
            __host__ __device__ __forceinline__ void store(u32 *p) {
                x.store(p);
                y.store(p + Element::LIMBS);
            }

            static __device__ __host__ __forceinline__ PointAffine identity() {
                return PointAffine(Element::zero(), Element::zero());
            }

            __device__ __host__ __forceinline__ bool is_identity() const & {
                return y.is_zero();
            }

            __device__ __host__ __forceinline__ bool operator==(const PointAffine &rhs) const & {
                return x == rhs.x && y == rhs.y;
            }

            __device__ __host__ __forceinline__ bool is_on_curve() const & {
                Element t0, t1;
                t0 = x.square();
                if constexpr (!Params::a().is_zero()) t0 = t0 + Params::a();
                t0 = t0 * x;
                t0 = t0 + Params::b();
                t1 = y.square();
                t0 = t1 - t0;
                return t0.is_zero();
            }

            __device__ __host__ __forceinline__ PointXYZZ to_point() const& {
                if unlikely(is_identity()) return PointXYZZ::identity();
                return PointXYZZ(x, y, Element::one(), Element::one());
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
            __device__ __host__ __forceinline__ PointXYZZ add_self() const& {
                auto u = y + y;
                auto v = u.square();
                auto w = u * v;
                auto s = x * v;
                auto x2 = x.square();
                auto m = x2 + x2 + x2;
                if constexpr (!Params::a().is_zero()) m = m + Params::a();
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                return PointXYZZ(x3, y3, v, w);
            }
         };


        //  https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
        //  x=X/ZZ
        //  y=Y/ZZZ
        //  ZZ^3=ZZZ^2
        struct PointXYZZ {
            static const usize N_WORDS = 4 * Element::LIMBS;
            Element x, y, zz, zzz;

            __host__ __device__ __forceinline__ PointXYZZ() {};
            __host__ __device__ __forceinline__ PointXYZZ(Element x, Element y, Element zz, Element zzz) : x(x), y(y), zz(zz), zzz(zzz) {}

            static __host__ __device__ __forceinline__ PointXYZZ load(const u32 *p) {
                auto x = Element::load(p);
                auto y = Element::load(p + Element::LIMBS);
                auto zz = Element::load(p + Element::LIMBS * 2);
                auto zzz = Element::load(p + Element::LIMBS * 3);
                return PointXYZZ(x, y, zz, zzz);
            }
            __host__ __device__ __forceinline__ void store(u32 *p) {
                x.store(p);
                y.store(p + Element::LIMBS);
                zz.store(p + Element::LIMBS * 2);
                zzz.store(p + Element::LIMBS * 3);
            }

            static constexpr __device__ __host__ __forceinline__ PointXYZZ identity() {
                return PointXYZZ(Element::zero(), Element::zero(), Element::zero(), Element::one());
            }

            __device__ __host__ __forceinline__ bool is_identity() const & {
                return zz.is_zero();
            }

            __device__ __host__ __forceinline__ PointXYZZ neg() const & {
                return PointXYZZ(x, y.neg(), zz, zzz);
            }

            __device__ __host__ __forceinline__ bool operator==(const PointXYZZ &rhs) const & {
                if (zz.is_zero() != rhs.zz.is_zero())
                    return false;
                auto x1 = x * rhs.zz;
                auto x2 = rhs.x * zz;
                auto y1 = y * rhs.zzz;
                auto y2 = rhs.y * zzz;
                return x1 == x2 && y1 == y2;
            }

            // x = X/ZZ
            // y = Y/ZZZ
            // ZZ^3 = ZZZ^2
            // y^2 = x^3 + a*x + b
            // Y^2/ZZZ^2 = X^3/ZZ^3 + a*X/ZZ + b
            // Y^2 = X^3 + a*X*ZZ^2 + b*ZZ^3
            __device__ __host__ __forceinline__ bool is_on_curve() const & {
                auto y2 = y.square();
                auto x3 = x.square() * x;
                auto zz2 = zz.square();
                auto zz3 = zz * zz2;
                auto zzz2 = zzz.square();
                if (zz3 != zzz2) return false;
                Element a_x_zz2;
                if constexpr (Params::a().is_zero()) a_x_zz2 = Element::zero();
                else a_x_zz2 = Params::a() * x * zz2;
                auto b_zz3 = Params::b() * zz3;
                return y2 == x3 + a_x_zz2 + b_zz3;
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#scaling-z
            __device__ __host__ __forceinline__ PointAffine to_affine() const & {
                auto A = zzz.invert();
                auto B = (zz * A).square();
                auto X3 = x * B;
                auto Y3 = y * A;
                return PointAffine(X3, Y3);
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
            __device__ __host__ __forceinline__ PointXYZZ operator + (const PointXYZZ &rhs) const & {
                if unlikely(this->is_identity()) return rhs;
                if unlikely(rhs.is_identity()) return *this;
                auto u1 = x * rhs.zz;
                auto u2 = rhs.x * zz;
                auto s1 = y * rhs.zzz;
                auto s2 = rhs.y * zzz;
                auto p = u2 - u1;
                auto r = s2 - s1;
                if unlikely(p.is_zero() && r.is_zero()) {
                    return this->self_add();
                }
                auto pp = p.square();
                auto ppp = p * pp;
                auto q = u1 * pp;
                auto x3 = r.square() - ppp - q - q;
                auto y3 = r * (q - x3) - s1 * ppp;
                auto zz3 = zz * rhs.zz * pp;
                auto zzz3 = zzz * rhs.zzz * ppp;
                return PointXYZZ(x3, y3, zz3, zzz3);
            }

            __device__ __host__ __forceinline__ PointXYZZ operator - (const PointXYZZ &rhs) const & {
                return *this + rhs.neg();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
            __device__ __host__ __forceinline__ PointXYZZ operator + (const PointAffine &rhs) const & {
                if unlikely(this->is_identity()) return rhs.to_point();
                if unlikely(rhs.is_identity()) return *this;
                auto u2 = rhs.x * zz;
                auto s2 = rhs.y * zzz;
                auto p = u2 - x;
                auto r = s2 - y;
                if unlikely(p.is_zero() && r.is_zero()) {
                    return rhs.add_self();
                }
                auto pp = p.square();
                auto ppp = p * pp;
                auto q = x * pp;
                auto x3 = r.square() - ppp - q - q;
                auto y3 = r * (q - x3) - y * ppp;
                auto zz3 = zz * pp;
                auto zzz3 = zzz * ppp;
                return PointXYZZ(x3, y3, zz3, zzz3);
            }

            __device__ __host__ __forceinline__ PointXYZZ operator - (const PointAffine &rhs) const & {
                return *this + rhs.neg();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
            __device__ __host__ __forceinline__ PointXYZZ self_add() const & {
                if unlikely(zz.is_zero()) return *this;
                auto u = y + y;
                auto v = u.square();
                auto w = u * v;
                auto s = x * v;
                auto x2 = x.square();
                auto m = x2 + x2 + x2;
                if constexpr (!Params::a().is_zero()) m = m + (Params::a() * zz.square());
                auto x3 = m.square() - s - s;
                auto y3 = m * (s - x3) - w * y;
                auto zz3 = v * zz;
                auto zzz3 = w * zzz;
                return PointXYZZ(x3, y3, zz3, zzz3);
            }

            static __device__ __host__ __forceinline__ void multiple_iter(const PointXYZZ &p, bool &found_one, PointXYZZ &res, u32 n) {
                for (int i = 31; i >= 0; i--) {
                    if (found_one) res = res.self_add();
                    if ((n >> i) & 1) {
                        found_one = true;
                        res = res + p;
                    }
                }
            }

            __device__ __host__ __forceinline__ PointXYZZ multiple(u32 n) const & {
                auto res = identity();
                bool found_one = false;
                multiple_iter(*this, found_one, res, n);
                return res;
            }

            __device__ __host__ __forceinline__ PointXYZZ shuffle_down(const u32 delta, u32 mask = 0xffffffff) const & {
                PointXYZZ res;
                res.x = x.shuffle_down(delta, mask);
                res.y = y.shuffle_down(delta, mask);
                res.zz = zz.shuffle_down(delta, mask);
                res.zzz = zzz.shuffle_down(delta, mask);
                return res;
            }
        };
    };
}