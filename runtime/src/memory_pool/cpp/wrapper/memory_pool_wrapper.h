#pragma once

#ifdef __cplusplus
extern "C" {
#endif
    typedef void* SlabMangerHandle;

    SlabMangerHandle create_slab_manager(unsigned int max_log_factor, unsigned long base_size);

    void destroy_slab_manager(SlabMangerHandle handle);

    void* allocate(SlabMangerHandle handle, unsigned int log_factor);

    void deallocate(SlabMangerHandle handle, void* ptr);

    void clear(SlabMangerHandle handle);

    void shrink(SlabMangerHandle handle);

#ifdef __cplusplus
}
#endif