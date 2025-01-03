#include "memory_pool_wrapper.h"
#include "../src/memory_pool.h"

SlabMangerHandle create_slab_manager(unsigned int max_log_factor, unsigned long base_size) {
    return reinterpret_cast<SlabMangerHandle>(new memory_pool::slab_manager(max_log_factor, base_size));
}

void destroy_slab_manager(SlabMangerHandle handle) {
    delete reinterpret_cast<memory_pool::slab_manager*>(handle);
}

void* allocate(SlabMangerHandle handle, unsigned int log_factor) {
    return reinterpret_cast<memory_pool::slab_manager*>(handle)->allocate(log_factor);
}

void deallocate(SlabMangerHandle handle, void* ptr) {
    reinterpret_cast<memory_pool::slab_manager*>(handle)->deallocate(ptr);
}

void clear(SlabMangerHandle handle) {
    reinterpret_cast<memory_pool::slab_manager*>(handle)->clear();
}

void shrink(SlabMangerHandle handle) {
    reinterpret_cast<memory_pool::slab_manager*>(handle)->shrink();
}