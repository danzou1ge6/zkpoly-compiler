#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/memory_pool.h"
#include <random>

TEST_CASE("test simple alloc and free") {
    memory_pool::slab_manager slab_manager(5, sizeof(int));
    int* ptr = (int*)slab_manager.allocate(5);
    REQUIRE(ptr != nullptr);
    for (int i = 0; i < (1 << 5); i++) {
        ptr[i] = i;
    }
    for (int i = 0; i < (1 << 5); i++) {
        REQUIRE(ptr[i] == i);
    }
    slab_manager.deallocate(ptr);
}

TEST_CASE("test simple alloc and free for multiple times") {
    memory_pool::slab_manager slab_manager(5, sizeof(int));
    int* last_ptr = nullptr;
    for (int j = 0; j < 10; j++) {
        int* ptr = (int*)slab_manager.allocate(5);
        REQUIRE(ptr != nullptr);
        if (last_ptr != nullptr) {
            REQUIRE(ptr == last_ptr);
        }
        last_ptr = ptr;
        for (int i = 0; i < (1 << 5); i++) {
            ptr[i] = i;
        }
        for (int i = 0; i < (1 << 5); i++) {
            REQUIRE(ptr[i] == i);
        }
        slab_manager.deallocate(ptr);
    }
}

TEST_CASE("test simple alloc and free different sizes") {
    memory_pool::slab_manager slab_manager(5, sizeof(int));
    for (int i = 1; i <= 5; i++) {
        int* ptr = (int*)slab_manager.allocate(i);
        REQUIRE(ptr != nullptr);
        for (int j = 0; j < (1 << i); j++) {
            ptr[j] = j;
        }
        for (int j = 0; j < (1 << i); j++) {
            REQUIRE(ptr[j] == j);
        }
    }
}

TEST_CASE("test split and merge") {
    memory_pool::slab_manager slab_manager(5, sizeof(int));
    int* ptr1 = (int*)slab_manager.allocate(4);
    int* ptr2 = (int*)slab_manager.allocate(4);
    slab_manager.deallocate(ptr1);
    slab_manager.deallocate(ptr2);
    int* ptr3 = (int*)slab_manager.allocate(5);
    REQUIRE(ptr1 == ptr3);
}

TEST_CASE("test shrink") {
    memory_pool::slab_manager slab_manager(5, sizeof(int));
    int *ptr[6];
    for (int i = 1; i <= 5; i++) {
        ptr[i] = (int*)slab_manager.allocate(i);
        REQUIRE(ptr[i] != nullptr);
        for (int j = 0; j < (1 << i); j++) {
            ptr[i][j] = j;
        }
    }
    slab_manager.shrink();
    for (int i = 1; i <= 5; i++) {
        for (int j = 0; j < (1 << i); j++) {
            REQUIRE(ptr[i][j] == j);
        }
        slab_manager.deallocate(ptr[i]);
    }
    slab_manager.shrink();
    slab_manager.allocate(5);
}

TEST_CASE("test clear") {
    memory_pool::slab_manager slab_manager(5, sizeof(int));
    for (int i = 1; i <= 5; i++) {
        auto ptr = (int*)slab_manager.allocate(i);
        REQUIRE(ptr != nullptr);
        for (int j = 0; j < (1 << i); j++) {
            ptr[j] = j;
        }
    }
    slab_manager.clear();
    for (int i = 1; i <= 5; i++) {
        int* ptr = (int*)slab_manager.allocate(i);
        REQUIRE(ptr != nullptr);
        for (int j = 0; j < (1 << i); j++) {
            ptr[j] = j;
        }
        for (int j = 0; j < (1 << i); j++) {
            REQUIRE(ptr[j] == j);
        }
    }
}

TEST_CASE("test complex") {
    int large_size = 16;
    int rounds = 1000;
    int items = 1000;
    memory_pool::slab_manager slab_manager(large_size, sizeof(int));
    for (int k = 0; k < rounds; k++) {
        std::vector<int*> ptrs;
        for (int i = 0; i < items; i++) {
            int size = rand() % large_size + 1;
            int* ptr = (int*)slab_manager.allocate(size);
            REQUIRE(ptr != nullptr);
            for (int j = 0; j < (1 << size); j++) {
                ptr[j] = j;
            }
            ptrs.push_back(ptr);
        }
        std::shuffle(ptrs.begin(), ptrs.end(), std::default_random_engine());
        for (int i = 0; i < items; i++) {
            slab_manager.deallocate(ptrs[i]);
        }
    }
    slab_manager.shrink();
    slab_manager.clear();
}