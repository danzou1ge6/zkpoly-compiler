#pragma once

#include <cstdint>
#include <iterator>
#include <cassert>

using i64 = std::int64_t;
using usize = std::size_t;
using u32 = std::uint32_t;

// collection for poly ptr, used to transport from rust
extern "C" struct PolyPtr {
    u32* ptr;
    usize len;
    usize rotate;
    usize offset;
    usize whole_len;
};

extern "C" struct ConstPolyPtr {
    const u32* ptr;
    usize len;
    usize rotate;
    usize offset;
    usize whole_len;
};

namespace iter {

// 统一的切片迭代器实现，将rotation统一到offset中
template<typename Element>
struct SliceIterator {
    // STL iterator traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Element;
    using difference_type = std::ptrdiff_t;
    using pointer = Element*;
    using reference = Element&;

    Element* data;
    usize length;      // 当前视图长度
    usize offset;      // 当前视图的起始偏移
    usize whole_len;   // 底层数组总长度
    const usize start_offset;  // 起始偏移

    __device__ __host__ __forceinline__ SliceIterator(
        Element* data_,
        usize length_,
        usize offset_ = 0,
        usize whole_len_ = 0
    ) : data(data_), length(length_), offset(offset_), start_offset(offset_) {
        whole_len = whole_len_ == 0 ? length_ : whole_len_;
        assert(length <= whole_len);
        assert(offset < whole_len);
    }

    // 允许非const到const的隐式转换
    __device__ __host__ __forceinline__ operator SliceIterator<const Element>() const {
        return SliceIterator<const Element>(data, length, offset, whole_len);
    }

    __device__ __host__ __forceinline__ Element& operator[](usize i) {
        assert(i < length);
        return data[(offset + i) % whole_len];
    }

    __device__ __host__ __forceinline__ const Element& operator[](usize i) const {
        assert(i < length);
        return data[(offset + i) % whole_len];
    }

    // Random Access Iterator requirements
    __device__ __host__ __forceinline__ Element& operator*() { 
        return data[offset];
    }

    __device__ __host__ __forceinline__ const Element& operator*() const { 
        return data[offset];
    }

    __device__ __host__ __forceinline__ SliceIterator& operator++() {
        offset = (offset + 1) % whole_len;
        return *this;
    }

    __device__ __host__ __forceinline__ SliceIterator operator++(int) {
        SliceIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    __device__ __host__ __forceinline__ SliceIterator& operator--() {
        offset = (offset + whole_len - 1) % whole_len;
        return *this;
    }

    __device__ __host__ __forceinline__ SliceIterator operator--(int) {
        SliceIterator tmp = *this;
        --(*this);
        return tmp;
    }

    __device__ __host__ __forceinline__ SliceIterator& operator+=(difference_type n) {
        offset = (offset + n) % whole_len;
        return *this;
    }

    __device__ __host__ __forceinline__ SliceIterator operator+(difference_type n) const {
        SliceIterator tmp = *this;
        return tmp += n;
    }

    __device__ __host__ __forceinline__ friend SliceIterator operator+(
        difference_type n,
        const SliceIterator& it
    ) {
        return it + n;
    }

    __device__ __host__ __forceinline__ SliceIterator& operator-=(difference_type n) {
        offset = (offset + whole_len - (n % whole_len)) % whole_len;
        return *this;
    }

    __device__ __host__ __forceinline__ SliceIterator operator-(difference_type n) const {
        SliceIterator tmp = *this;
        return tmp -= n;
    }

    __device__ __host__ __forceinline__ difference_type operator-(const SliceIterator& other) const {
        return (whole_len + offset - other.offset) % whole_len;
    }

    __device__ __host__ __forceinline__ bool operator==(const SliceIterator& other) const {
        return data == other.data &&
               length == other.length &&
               offset == other.offset &&
               whole_len == other.whole_len;
    }

    __device__ __host__ __forceinline__ bool operator!=(const SliceIterator& other) const {
        return !(*this == other);
    }

    __device__ __host__ __forceinline__ bool operator<(const SliceIterator& other) const {
        // check two iterators are in the same slice
        assert(whole_len == other.whole_len);
        assert(length == other.length);
        assert(data == other.data);
        assert(start_offset == other.start_offset);

        if (start_offset <= offset && (offset < other.offset || start_offset > other.offset)) {
            return true;
        }

        if (start_offset > offset && start_offset > other.offset && offset < other.offset) {
            return true;
        }

        return false;
    }

    __device__ __host__ __forceinline__ bool operator>(const SliceIterator& other) const {
        return other < *this;
    }

    __device__ __host__ __forceinline__ bool operator<=(const SliceIterator& other) const {
        return !(other < *this);
    }

    __device__ __host__ __forceinline__ bool operator>=(const SliceIterator& other) const {
        return !(*this < other);
    }
};

// 常量版本特化
template<typename Element>
struct SliceIterator<const Element> {
    // STL iterator traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Element;
    using difference_type = std::ptrdiff_t;
    using pointer = const Element*;
    using reference = const Element&;

    const Element* data;
    usize length;      // 当前视图长度
    usize offset;      // 当前视图的起始偏移
    usize whole_len;   // 底层数组总长度
    const usize start_offset;  // 起始偏移
    
    __device__ __host__ __forceinline__ SliceIterator(
        const Element* data_,
        usize length_,
        usize offset_ = 0,
        usize whole_len_ = 0
    ) : data(data_), length(length_), offset(offset_), start_offset(offset_) {
        whole_len = whole_len_ == 0 ? length_ : whole_len_;
        assert(length <= whole_len);
        assert(offset < whole_len);
    }

    __device__ __host__ __forceinline__ const Element& operator[](usize i) const {
        assert(i < length);
        return data[(offset + i) % whole_len];
    }

    __device__ __host__ __forceinline__ const Element& operator*() const { 
        return data[offset];
    }

    __device__ __host__ __forceinline__ SliceIterator& operator++() {
        offset = (offset + 1) % whole_len;
        return *this;
    }

    __device__ __host__ __forceinline__ SliceIterator operator++(int) {
        SliceIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    __device__ __host__ __forceinline__ SliceIterator& operator--() {
        offset = (offset + whole_len - 1) % whole_len;
        return *this;
    }

    __device__ __host__ __forceinline__ SliceIterator operator--(int) {
        SliceIterator tmp = *this;
        --(*this);
        return tmp;
    }

    __device__ __host__ __forceinline__ SliceIterator& operator+=(difference_type n) {
        offset = (offset + n) % whole_len;
        return *this;
    }

    __device__ __host__ __forceinline__ SliceIterator operator+(difference_type n) const {
        SliceIterator tmp = *this;
        return tmp += n;
    }

    __device__ __host__ __forceinline__ friend SliceIterator operator+(
        difference_type n,
        const SliceIterator& it
    ) {
        return it + n;
    }

    __device__ __host__ __forceinline__ SliceIterator& operator-=(difference_type n) {
        offset = (offset + whole_len - (n % whole_len)) % whole_len;
        return *this;
    }

    __device__ __host__ __forceinline__ SliceIterator operator-(difference_type n) const {
        SliceIterator tmp = *this;
        return tmp -= n;
    }

    __device__ __host__ __forceinline__ difference_type operator-(const SliceIterator& other) const {
        return (whole_len + offset - other.offset) % whole_len;
    }

    __device__ __host__ __forceinline__ bool operator==(const SliceIterator& other) const {
        return data == other.data &&
               length == other.length &&
               offset == other.offset &&
               whole_len == other.whole_len;
    }

    __device__ __host__ __forceinline__ bool operator!=(const SliceIterator& other) const {
        return !(*this == other);
    }

    __device__ __host__ __forceinline__ bool operator<(const SliceIterator& other) const {
        // check two iterators are in the same slice
        assert(whole_len == other.whole_len);
        assert(length == other.length);
        assert(data == other.data);
        assert(start_offset == other.start_offset);

        if (start_offset <= offset && (offset < other.offset || start_offset > other.offset)) {
            return true;
        }

        if (start_offset > offset && start_offset > other.offset && offset < other.offset) {
            return true;
        }

        return false;
    }

    __device__ __host__ __forceinline__ bool operator>(const SliceIterator& other) const {
        return other < *this;
    }

    __device__ __host__ __forceinline__ bool operator<=(const SliceIterator& other) const {
        return !(other < *this);
    }

    __device__ __host__ __forceinline__ bool operator>=(const SliceIterator& other) const {
        return !(*this < other);
    }
};

// 工具函数 - 非const版本
template<typename Element>
__device__ __host__ __forceinline__ SliceIterator<Element> make_slice_iter(PolyPtr ptr) {
    if (ptr.rotate != 0) {
        assert(ptr.len == ptr.whole_len);
        assert(ptr.offset == 0);
        ptr.offset = ptr.len - ptr.rotate;
    }
    return SliceIterator<Element>(reinterpret_cast<Element*>(ptr.ptr), ptr.len, ptr.offset, ptr.whole_len);
}

// 工具函数 - const版本
template<typename Element>
__device__ __host__ __forceinline__ SliceIterator<const Element> make_slice_iter(ConstPolyPtr ptr) {
    if (ptr.rotate != 0) {
        assert(ptr.len == ptr.whole_len);
        assert(ptr.offset == 0);
        ptr.offset = ptr.len - ptr.rotate;
    }
    return SliceIterator<const Element>(reinterpret_cast<const Element*>(ptr.ptr), ptr.len, ptr.offset, ptr.whole_len);
}

} // namespace iter

// 特化std::iterator_traits
namespace std {
template<typename Element>
struct iterator_traits<iter::SliceIterator<Element>> {
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Element;
    using difference_type = std::ptrdiff_t;
    using pointer = Element*;
    using reference = Element&;
};

template<typename Element>
struct iterator_traits<iter::SliceIterator<const Element>> {
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Element;
    using difference_type = std::ptrdiff_t;
    using pointer = const Element*;
    using reference = const Element&;
};
}
