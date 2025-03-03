#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../src/iter.cuh"
#include <doctest/doctest.h>
#include <vector>

using namespace iter;

// 测试基本构造和访问
TEST_CASE("SliceIterator: Basic Construction") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter = SliceIterator<u32>(data.data(), 3, 0, 5);
    
    CHECK_EQ(*iter, 1);
    CHECK_EQ(iter[0], 1);
    CHECK_EQ(iter[1], 2);
    CHECK_EQ(iter[2], 3);
}

// 测试偏移和环绕
TEST_CASE("SliceIterator: Offset and Wrap") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter = SliceIterator<u32>(data.data(), 3, 3, 5);
    
    CHECK_EQ(*iter, 4);
    CHECK_EQ(iter[0], 4);
    CHECK_EQ(iter[1], 5);
    CHECK_EQ(iter[2], 1);
}

// 测试迭代器操作
TEST_CASE("SliceIterator: Iterator Operations") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter = SliceIterator<u32>(data.data(), 5, 0, 5);
    
    // 测试++操作符
    CHECK_EQ(*iter, 1);
    ++iter;
    CHECK_EQ(*iter, 2);
    iter++;
    CHECK_EQ(*iter, 3);
    
    // 测试--操作符
    --iter;
    CHECK_EQ(*iter, 2);
    iter--;
    CHECK_EQ(*iter, 1);
    
    // 测试+=和-=操作符
    iter += 2;
    CHECK_EQ(*iter, 3);
    iter -= 1;
    CHECK_EQ(*iter, 2);
}

// 测试比较操作符
TEST_CASE("SliceIterator: Comparison Operators") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter1 = SliceIterator<u32>(data.data(), 5, 0, 5);
    auto iter2 = SliceIterator<u32>(data.data(), 5, 0, 5);
    auto iter3 = SliceIterator<u32>(data.data(), 5, 1, 5);
    
    CHECK(iter1 == iter2);
    CHECK_FALSE(iter1 != iter2);
    CHECK(iter1 < iter3);
    CHECK(iter3 > iter1);
    CHECK(iter1 <= iter2);
    CHECK(iter1 >= iter2);
}

// 测试const转换
TEST_CASE("SliceIterator: Const Conversion") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    SliceIterator<u32> non_const_iter(data.data(), 5, 0, 5);
    SliceIterator<const u32> const_iter = non_const_iter; // 隐式转换
    
    CHECK_EQ(*const_iter, 1);
    CHECK_EQ(const_iter[2], 3);
}

// 测试make_slice_iter函数
TEST_CASE("SliceIterator: Make Slice Iter") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    
    // 测试非const版本
    PolyPtr ptr {
        .ptr = data.data(),
        .len = 5,
        .rotate = 2,
        .offset = 0,
        .whole_len = 5
    };
    
    auto iter = make_slice_iter<u32>(ptr);
    CHECK_EQ(*iter, 4);
    CHECK_EQ(iter[0], 4);
    CHECK_EQ(iter[1], 5);
    CHECK_EQ(iter[2], 1);
    
    // 测试const版本
    ConstPolyPtr const_ptr {
        .ptr = data.data(),
        .len = 5,
        .rotate = 2,
        .offset = 0,
        .whole_len = 5
    };
    
    auto const_iter = make_slice_iter<u32>(const_ptr);
    CHECK_EQ(*const_iter, 4);
    CHECK_EQ(const_iter[0], 4);
    CHECK_EQ(const_iter[1], 5);
    CHECK_EQ(const_iter[2], 1);
}

// 测试迭代器差值运算
TEST_CASE("SliceIterator: Iterator Difference") {
    std::vector<u32> data = {1, 2, 3, 4, 5};
    auto iter1 = SliceIterator<u32>(data.data(), 5, 1, 5);
    auto iter2 = SliceIterator<u32>(data.data(), 5, 3, 5);
    
    CHECK_EQ(iter2 - iter1, 2);
    CHECK_EQ(iter1 - iter2, 3); // 环绕差值
}
