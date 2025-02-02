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
                is >> p.x.n >> p.y.n;
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

            // static __device__ __host__ __forceinline__ PointAffine identity() {
            //     return PointAffine(Element::zero(), Element::zero());
            // }

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
                // if unlikely(is_identity()) return PointXYZZ::identity();
                auto res = PointXYZZ(x, y, Element::one(), Element::one());
                return res;
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
                auto p = PointXYZZ(x3, y3, v, w);
                return p;
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
                auto zz_ = Params::allow_lazy_modulo ? zz.modulo_m() : zz;
                return zz_.is_zero();
            }

            __device__ __host__ __forceinline__ PointXYZZ neg() const & {
                auto y_ = Params::allow_lazy_modulo ?  y.modulo_m() : y;
                auto r = PointXYZZ(x, y_.neg(), zz, zzz);
                return r;
            }

            __device__ __host__ __forceinline__ bool operator==(const PointXYZZ &rhs_) const & {
                if (zz.is_zero() != rhs_.zz.is_zero())
                    return false;
                auto lhs = normalized();
                auto rhs = rhs_.normalized();
                auto x1 = lhs.x * rhs.zz;
                auto x2 = rhs.x * lhs.zz;
                auto y1 = lhs.y * rhs.zzz;
                auto y2 = rhs.y * lhs.zzz;
                return x1 == x2 && y1 == y2;
            }

            // x = X/ZZ
            // y = Y/ZZZ
            // ZZ^3 = ZZZ^2
            // y^2 = x^3 + a*x + b
            // Y^2/ZZZ^2 = X^3/ZZ^3 + a*X/ZZ + b
            // Y^2 = X^3 + a*X*ZZ^2 + b*ZZ^3
            __device__ __host__ __forceinline__ bool is_on_curve() const & {
                auto self = normalized();
                auto y2 = self.y.square();
                auto x3 = self.x.square() * self.x;
                auto zz2 = self.zz.square();
                auto zz3 = self.zz * zz2;
                auto zzz2 = self.zzz.square();
                if (zz3 != zzz2) return false;
                Element a_x_zz2;
                if constexpr (Params::a().is_zero()) a_x_zz2 = Element::zero();
                else a_x_zz2 = Params::a() * self.x * zz2;
                auto b_zz3 = Params::b() * zz3;
                return y2 == x3 + a_x_zz2 + b_zz3;
            }

            __device__ __host__ __forceinline__ void normalize()
            {
                if (Params::allow_lazy_modulo)
                {
                    x = x.modulo_m();
                    y = y.modulo_m();
                    zz = zz.modulo_m();
                    zzz = zzz.modulo_m();
                }
            }

            __device__ __host__ __forceinline__ PointXYZZ normalized() const &
            {
                PointXYZZ r = *this;
                r.normalize();
                return r;
            }

            __device__ __host__ __forceinline__ bool is_elements_lt_2m() const &
            {
                return x.lt_2m() && y.lt_2m() && zz.lt_2m() && zzz.lt_2m();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#scaling-z
            __device__ __host__ __forceinline__ PointAffine to_affine() const & {
                auto self = normalized();
                auto A = self.zzz.invert();
                auto B = (self.zz * A).square();
                auto X3 = self.x * B;
                auto Y3 = self.y * A;
                return PointAffine(X3, Y3);
            }

            template <bool LAZY_MODULOS>
            __device__ __host__ __forceinline__ PointXYZZ add(const PointXYZZ &rhs) const &
            {
                if (LAZY_MODULOS)
                {
                    if unlikely(this->is_identity()) return rhs;
                    if unlikely(rhs.is_identity()) return *this;
                    auto u1 = x.template mul<false>(rhs.zz);
                    auto u2 = rhs.x.template mul<false>(zz);
                    auto s1 = y.template mul<false>(rhs.zzz);
                    auto s2 = rhs.y.template mul<false>(zzz);
                    auto p = u2.sub_modulo_mm2(u1);
                    auto r = s2.sub_modulo_mm2(s1);
                    p = p.modulo_m();
                    r = r.modulo_m();
                    if unlikely(p.is_zero() && r.is_zero()) {
                        return this->self_add();
                    }
                    auto pp = p.template square<false>();
                    auto ppp = p.template mul<false>(pp);
                    auto q = u1.template mul<false>(pp);
                    auto x3 = r.template square<false>();
                    x3 = x3.sub_modulo_mm2(ppp);
                    x3 = x3.sub_modulo_mm2(q);
                    x3 = x3.sub_modulo_mm2(q);
                    auto y3 = r.template mul<false>(q.sub_modulo_mm2(x3));
                    y3 = y3.sub_modulo_mm2(s1.template mul<false>(ppp));
                    auto zz3 = zz.template mul<false>(rhs.zz);
                    zz3 = zz3.template mul<false>(pp);
                    auto zzz3 = zzz.template mul<false>(rhs.zzz);
                    zzz3 = zzz3.template mul<false>(ppp);
                    return PointXYZZ(x3, y3, zz3, zzz3);

                } else {
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
            }

            __host__ __device__ void device_print() const &
            {
                printf(
                    "{ x = %x %x %x %x %x %x %x %x\n, y = %x %x %x %x %x %x %x %x\n, zz = %x %x %x %x %x %x %x %x\n, zzz = %x %x %x %x %x %x %x %x }\n",
                    x.n.limbs[7], x.n.limbs[6], x.n.limbs[5], x.n.limbs[4], x.n.limbs[3], x.n.limbs[2], x.n.limbs[1], x.n.limbs[0], 
                    y.n.limbs[7], y.n.limbs[6], y.n.limbs[5], y.n.limbs[4], y.n.limbs[3], y.n.limbs[2], y.n.limbs[1], y.n.limbs[0], 
                    zz.n.limbs[7], zz.n.limbs[6], zz.n.limbs[5], zz.n.limbs[4], zz.n.limbs[3], zz.n.limbs[2], zz.n.limbs[1], zz.n.limbs[0], 
                    zzz.n.limbs[7], zzz.n.limbs[6], zzz.n.limbs[5], zzz.n.limbs[4], zzz.n.limbs[3], zzz.n.limbs[2], zzz.n.limbs[1], zzz.n.limbs[0]
                );
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
            __device__ __host__ __forceinline__ PointXYZZ operator + (const PointXYZZ &rhs) const & {
                return add<Params::allow_lazy_modulo()>(rhs);
            }

            __device__ __host__ __forceinline__ PointXYZZ operator - (const PointXYZZ &rhs) const & {
                return *this + rhs.neg();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
            template <bool LAZY_MODULOS>
            __device__ __host__ __forceinline__ PointXYZZ add(const PointAffine &rhs) const & {
                PointXYZZ sum;
                if (LAZY_MODULOS)
                {
                    if unlikely(this->is_identity()) return rhs.to_point();
                    // if unlikely(rhs.is_identity()) return *this;
                    auto u2 = rhs.x.template mul<false>(zz);
                    auto s2 = rhs.y.template mul<false>(zzz);
                    auto p = u2.sub_modulo_mm2(x);
                    auto r = s2.sub_modulo_mm2(y);
                    if unlikely(p.is_zero() && r.is_zero()) {
                        return rhs.add_self();
                    }
                    auto pp = p.template square<false>();
                    auto ppp = p.template mul<false>(pp);
                    auto q = x.template mul<false>(pp);
                    auto x3 = r.template square<false>();
                    x3 = x3.sub_modulo_mm2(ppp);
                    x3 = x3.sub_modulo_mm2(q);
                    x3 = x3.sub_modulo_mm2(q);
                    auto y3 = r.template mul<false>(q.sub_modulo_mm2(x3));
                    y3 = y3.sub_modulo_mm2(y.template mul<false>(ppp));
                    auto zz3 = zz.template mul<false>(pp);
                    auto zzz3 = zzz.template mul<false>(ppp);
                    sum = PointXYZZ(x3, y3, zz3, zzz3);
                } else {
                    if unlikely(this->is_identity()) return rhs.to_point();
                    // if unlikely(rhs.is_identity()) return *this;
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
                    sum = PointXYZZ(x3, y3, zz3, zzz3);
                }
                return sum;
           }

            __device__ __host__ __forceinline__ PointXYZZ operator + (const PointAffine &rhs) const & {
                return add<Params::allow_lazy_modulo()>(rhs);
            }

            __device__ __host__ __forceinline__ PointXYZZ operator - (const PointAffine &rhs) const & {
                return *this + rhs.neg();
            }

            // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
            __device__ __host__ __forceinline__ PointXYZZ self_add() const & {
                PointXYZZ r;
                if (Params::allow_lazy_modulo)
                {
                    if unlikely(zz.is_zero()) return *this;
                    auto u = y.add_modulo_mm2(y);
                    auto v = u.template square<false>();
                    auto w = u.template mul<false>(v);
                    auto s = x.template mul<false>(v);
                    auto x2 = x.template square<false>();
                    auto m = x2.add_modulo_mm2(x2).add_modulo_mm2(x2);
                    if constexpr (!Params::a().is_zero())
                        m = m.add_modulo_mm2(Params::a().template mul<false>(zz.template square<false>()));
                    auto x3 = m.template square<false>();
                    x3 = x3.sub_modulo_mm2(s).sub_modulo_mm2(s);
                    auto y3 = m.template mul<false>(s.sub_modulo_mm2(x3));
                    y3 = y3.sub_modulo_mm2(w.template mul<false>(y));
                    auto zz3 = v.template mul<false>(zz);
                    auto zzz3 = w.template mul<false>(zzz);
                    r = PointXYZZ(x3, y3, zz3, zzz3);
                } else {
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
                    r = PointXYZZ(x3, y3, zz3, zzz3);
                }
                return r;
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

            __device__ __forceinline__ PointXYZZ shuffle_down(const u32 delta) const & {
                PointXYZZ res;
                res.x = x.shuffle_down(delta);
                res.y = y.shuffle_down(delta);
                res.zz = zz.shuffle_down(delta);
                res.zzz = zzz.shuffle_down(delta);
                return res;
            }
        };
    };
}