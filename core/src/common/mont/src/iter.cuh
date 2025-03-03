#pragma once

#include <cstdint>
#include <iterator>
#include "field.cuh"

namespace mont {

template<typename Element>
struct RotatingIterator {
    // STL iterator traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Element;
    using difference_type = std::ptrdiff_t;
    using pointer = Element*;
    using reference = Element&;

    Element* data;
    i64 rotate;
    usize length;

    __device__ __host__ __forceinline__ RotatingIterator(Element* data_, i64 rotate_, usize length_) 
        : data(data_), rotate(rotate_), length(length_) {
            assert(abs(rotate) < length);
        }

    // 允许非const到const的隐式转换
    __device__ __host__ __forceinline__ operator RotatingIterator<const Element>() const {
        return RotatingIterator<const Element>(data, rotate, length);
    }

    __device__ __host__ __forceinline__ Element& operator[](usize i) {
        return data[(i + length - rotate) % length];
    }

    __device__ __host__ __forceinline__ const Element& operator[](usize i) const {
        return data[(i + length - rotate) % length];
    }

    // Random Access Iterator requirements
    __device__ __host__ __forceinline__ Element& operator*() { 
        return data[(length - rotate) % length];
    }

    __device__ __host__ __forceinline__ const Element& operator*() const { 
        return data[(length - rotate) % length];
    }

    __device__ __host__ __forceinline__ RotatingIterator& operator++() {
        rotate = (rotate - 1 + length) % length;
        return *this;
    }

    __device__ __host__ __forceinline__ RotatingIterator operator++(int) {
        RotatingIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    __device__ __host__ __forceinline__ RotatingIterator& operator--() {
        rotate = (rotate + 1) % length;
        return *this;
    }

    __device__ __host__ __forceinline__ RotatingIterator operator--(int) {
        RotatingIterator tmp = *this;
        --(*this);
        return tmp;
    }

    __device__ __host__ __forceinline__ RotatingIterator& operator+=(difference_type n) {
        rotate = (rotate - n + length) % length;
        return *this;
    }

    __device__ __host__ __forceinline__ RotatingIterator operator+(difference_type n) const {
        RotatingIterator tmp = *this;
        return tmp += n;
    }

    __device__ __host__ __forceinline__ friend RotatingIterator operator+(difference_type n, const RotatingIterator& it) {
        return it + n;
    }

    __device__ __host__ __forceinline__ RotatingIterator& operator-=(difference_type n) {
        return operator+=(-n);
    }

    __device__ __host__ __forceinline__ RotatingIterator operator-(difference_type n) const {
        RotatingIterator tmp = *this;
        return tmp -= n;
    }

    __device__ __host__ __forceinline__ difference_type operator-(const RotatingIterator& other) const {
        return (other.rotate - this->rotate + length) % length;
    }

    __device__ __host__ __forceinline__ bool operator==(const RotatingIterator& other) const {
        return data == other.data && rotate == other.rotate && length == other.length;
    }

    __device__ __host__ __forceinline__ bool operator!=(const RotatingIterator& other) const {
        return !(*this == other);
    }

    __device__ __host__ __forceinline__ bool operator<(const RotatingIterator& other) const {
        return rotate > other.rotate;  // 注意：这里是反的，因为rotate越大，实际访问的位置越前
    }

    __device__ __host__ __forceinline__ bool operator>(const RotatingIterator& other) const {
        return other < *this;
    }

    __device__ __host__ __forceinline__ bool operator<=(const RotatingIterator& other) const {
        return !(other < *this);
    }

    __device__ __host__ __forceinline__ bool operator>=(const RotatingIterator& other) const {
        return !(*this < other);
    }
};

// 常量版本特化
template<typename Element>
struct RotatingIterator<const Element> {
    // STL iterator traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Element;
    using difference_type = std::ptrdiff_t;
    using pointer = const Element*;
    using reference = const Element&;

    const Element* data;
    i64 rotate;
    usize length;

    __device__ __host__ __forceinline__ RotatingIterator(const Element* data_, i64 rotate_, usize length_)
        : data(data_), rotate(rotate_), length(length_) {
            assert(abs(rotate) < length);
        }

    __device__ __host__ __forceinline__ const Element& operator[](usize i) const {
        return data[(i + length - rotate) % length];
    }

    __device__ __host__ __forceinline__ const Element& operator*() const { 
        return data[(length - rotate) % length];
    }

    __device__ __host__ __forceinline__ RotatingIterator& operator++() {
        rotate = (rotate - 1 + length) % length;
        return *this;
    }

    __device__ __host__ __forceinline__ RotatingIterator operator++(int) {
        RotatingIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    __device__ __host__ __forceinline__ RotatingIterator& operator--() {
        rotate = (rotate + 1) % length;
        return *this;
    }

    __device__ __host__ __forceinline__ RotatingIterator operator--(int) {
        RotatingIterator tmp = *this;
        --(*this);
        return tmp;
    }

    __device__ __host__ __forceinline__ RotatingIterator& operator+=(difference_type n) {
        rotate = (rotate - n + length) % length;
        return *this;
    }

    __device__ __host__ __forceinline__ RotatingIterator operator+(difference_type n) const {
        RotatingIterator tmp = *this;
        return tmp += n;
    }

    __device__ __host__ __forceinline__ friend RotatingIterator operator+(difference_type n, const RotatingIterator& it) {
        return it + n;
    }

    __device__ __host__ __forceinline__ RotatingIterator& operator-=(difference_type n) {
        return operator+=(-n);
    }

    __device__ __host__ __forceinline__ RotatingIterator operator-(difference_type n) const {
        RotatingIterator tmp = *this;
        return tmp -= n;
    }

    __device__ __host__ __forceinline__ difference_type operator-(const RotatingIterator& other) const {
        return (other.rotate - this->rotate + length) % length;
    }

    __device__ __host__ __forceinline__ bool operator==(const RotatingIterator& other) const {
        return data == other.data && rotate == other.rotate && length == other.length;
    }

    __device__ __host__ __forceinline__ bool operator!=(const RotatingIterator& other) const {
        return !(*this == other);
    }

    __device__ __host__ __forceinline__ bool operator<(const RotatingIterator& other) const {
        return rotate > other.rotate;  // 注意：这里是反的
    }

    __device__ __host__ __forceinline__ bool operator>(const RotatingIterator& other) const {
        return other < *this;
    }

    __device__ __host__ __forceinline__ bool operator<=(const RotatingIterator& other) const {
        return !(other < *this);
    }

    __device__ __host__ __forceinline__ bool operator>=(const RotatingIterator& other) const {
        return !(*this < other);
    }
};

// 工具函数 - 非const版本
template<typename Element>
__device__ __host__ __forceinline__ RotatingIterator<Element> make_rotating_iter(Element* data, i64 rotate, usize length) {
    return RotatingIterator<Element>(data, rotate, length);
}

// 工具函数 - const版本 
template<typename Element>
__device__ __host__ __forceinline__ RotatingIterator<const Element> make_rotating_iter(const Element* data, i64 rotate, usize length) {
    return RotatingIterator<const Element>(data, rotate, length);
}

} // namespace mont

// 特化std::iterator_traits
namespace std {
template<typename Element>
struct iterator_traits<mont::RotatingIterator<Element>> {
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Element;
    using difference_type = std::ptrdiff_t;
    using pointer = Element*;
    using reference = Element&;
};

template<typename Element>
struct iterator_traits<mont::RotatingIterator<const Element>> {
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Element;
    using difference_type = std::ptrdiff_t;
    using pointer = const Element*;
    using reference = const Element&;
};
}
