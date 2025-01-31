#include <cuda_runtime.h>
#include "memory_pool.h"
#include <iostream>
#include <cassert>

#define CUDA_CHECK(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    } \
}

namespace memory_pool {

    slab_info::slab_info(void* ptr, uint32_t log_factor, slab_info *parent, bool free)
    : next(nullptr), prev(nullptr), free_next(nullptr), free_prev(nullptr), ptr(ptr),
    log_factor(log_factor), free(free), parent(parent), lchild(nullptr), rchild(nullptr) {}

    slab_list::slab_list() {
        head = new slab_info();
        head->next = head;
        head->prev = head;
        free_head = new slab_info();
        free_head->free_next = free_head;
        free_head->free_prev = free_head;
    }

    slab_list::~slab_list() {
        delete head;
        delete free_head;
    }

    void slab_list::insert(slab_info *slab) {
        slab->next = head;
        slab->prev = head->prev;
        head->prev->next = slab;
        head->prev = slab;
        if (slab->free) {
            insert_free(slab);
        }
    }

    void slab_list::insert_free(slab_info *slab) {
        slab->free_next = free_head;
        slab->free_prev = free_head->free_prev;
        free_head->free_prev->free_next = slab;
        free_head->free_prev = slab;
    }

    void slab_list::remove(slab_info *slab) {
        assert(slab != head && slab != free_head);
        slab->next->prev = slab->prev;
        slab->prev->next = slab->next;
        slab->next = nullptr;
        slab->prev = nullptr;
        if (slab->free) {
            remove_free(slab);
        }
    }

    void slab_list::remove_free(slab_info *slab) {
        assert(slab != free_head);
        slab->free_next->free_prev = slab->free_prev;
        slab->free_prev->free_next = slab->free_next;
        slab->free_next = nullptr;
        slab->free_prev = nullptr;
    }

    slab_info* slab_list::get_free() {
        assert(has_free());
        return free_head->free_next;
    }

    slab_info* slab_list::get_head() {
        return head;
    }

    bool slab_list::has_free() const {
        return free_head->free_next != free_head;
    }

    void slab_list::reset() {
        head->next = head;
        head->prev = head;
        free_head->free_next = free_head;
        free_head->free_prev = free_head;
    }

    slab_manager::slab_manager(uint32_t max_log_factor, size_t base_size)
    : max_log_factor(max_log_factor), max_slab_size(base_size << max_log_factor), slab_lists(max_log_factor + 1), base_size(base_size) {}

    slab_manager::~slab_manager() {
        clear();
    }

    void slab_manager::clear() {
        for (int i = max_log_factor; i >= 0; i--) {
            slab_info *ptr = slab_lists[i].get_head()->next;
            while (ptr != slab_lists[i].get_head()) {
                slab_info *next = ptr->next;
                if (i == max_log_factor) {
                    assert(ptr->ptr != nullptr);
                    CUDA_CHECK(cudaFreeHost(ptr->ptr));
                }
                delete ptr;
                ptr = next;
            }
            slab_lists[i].reset();
        }
        slab_map.clear();
    }

    void* slab_manager::allocate(uint32_t log_factor) {
        assert(log_factor <= max_log_factor);
        if (!slab_lists[log_factor].has_free()) {
            if (log_factor == max_log_factor) {
                void *ptr = allocate_memory();
                assert(ptr != nullptr);
                slab_info *slab = new slab_info(ptr, log_factor);
                slab_lists[log_factor].insert(slab);
            } else {
                // split the slab in upper level
                std::pair<slab_info*, slab_info*> slabs = split(log_factor + 1);
                slab_lists[log_factor].insert(slabs.first);
                slab_lists[log_factor].insert(slabs.second);
            }
        }
        assert(slab_lists[log_factor].has_free());
        slab_info *slab = slab_lists[log_factor].get_free();
        slab->free = false;
        slab_lists[log_factor].remove_free(slab);

        assert(slab_map.find(slab->ptr) == slab_map.end());
        slab_map[slab->ptr] = slab;

        return slab->ptr;
    }

    void slab_manager::deallocate(void *ptr) {
        assert(slab_map.find(ptr) != slab_map.end());
        slab_info *slab = slab_map[ptr];
        slab_map.erase(ptr);
        slab->free = true;
        slab_lists[slab->log_factor].insert_free(slab);

        if (slab->log_factor < max_log_factor) {
            // merge the slab in upper level if possible
            assert(slab->lchild == nullptr && slab->rchild == nullptr);
            assert(slab->parent != nullptr);
            try_merge(slab->parent);
        }
        
    }

    void* slab_manager::allocate_memory() {
        void *ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, max_slab_size);
        if (err == cudaSuccess) return ptr;
        std::cerr << "Failed to allocate memory" << std::endl;
        exit(1);
    }

    std::pair<slab_info*, slab_info*> slab_manager::split(uint32_t log_factor) {
        assert(log_factor <= max_log_factor);
        if (!slab_lists[log_factor].has_free()) {
            if (log_factor == max_log_factor) {
                void *ptr = allocate_memory();
                assert(ptr != nullptr);
                slab_info *slab = new slab_info(ptr, log_factor);
                slab_lists[log_factor].insert(slab);
            } else {
                // split the slab in upper level
                std::pair<slab_info*, slab_info*> slabs = split(log_factor + 1);
                slab_lists[log_factor].insert(slabs.first);
                slab_lists[log_factor].insert(slabs.second);
            }
        }
        assert(slab_lists[log_factor].has_free());
        slab_info *slab = slab_lists[log_factor].get_free();
        slab_lists[log_factor].remove_free(slab);
        slab->free = false;

        void *ptr = slab->ptr;
        slab_info *lchild = new slab_info(ptr, log_factor - 1, slab, true);
        slab_info *rchild = new slab_info((char*)ptr + (base_size << (log_factor - 1)), log_factor - 1, slab, true);
        slab->lchild = lchild;
        slab->rchild = rchild;

        return std::make_pair(lchild, rchild);
    }

    void slab_manager::try_merge(slab_info *slab) {
        assert(slab->lchild != nullptr && slab->rchild != nullptr);
        if (slab->lchild->free && slab->rchild->free) {
            uint32_t log_size = slab->log_factor;

            slab_lists[log_size - 1].remove(slab->lchild);
            slab_lists[log_size - 1].remove(slab->rchild);
            delete slab->lchild;
            delete slab->rchild;
            slab->lchild = nullptr;
            slab->rchild = nullptr;

            slab->free = true;
            slab_lists[log_size].insert_free(slab);
            
            if (slab->log_factor < max_log_factor) {
                assert(slab->parent != nullptr);
                try_merge(slab->parent);
            }
        }
    }

    void slab_manager::shrink() {
        slab_info *ptr = slab_lists[max_log_factor].get_head()->next;
        while (ptr != slab_lists[max_log_factor].get_head()) {
            slab_info *next = ptr->next;
            if (ptr->free) {
                CUDA_CHECK(cudaFreeHost(ptr->ptr));
                slab_lists[max_log_factor].remove(ptr);
                delete ptr;
            }
            ptr = next;
        }
    }
} // namespace memory_pool