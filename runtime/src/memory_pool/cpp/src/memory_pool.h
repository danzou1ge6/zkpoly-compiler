#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace memory_pool {
    /*
    slabs are organized in a binary tree structure with two linked lists in each layer
    the first linked list is the list of all slabs in the layer
    the second linked list is the list of all free slabs in the layer
    e.g.
    layer10: head <-> slab0 <-> slab1 <-> head
    free_layer10: free_head <-> slab0 <-> free_head
    slab1 has two children slab2 and slab3 in layer9
    */
    struct slab_info {
        void* ptr; // pointer to the slab
        bool free; // whether the slab is free
        uint32_t log_factor; // log factor of the size of the slab

        // class slab_list will manage the linked list
        slab_info *next, *prev; // linked list for all slabs in the layer
        slab_info *free_next, *free_prev; // linked list for free slabs in the layer

        // class slab_manager will manage the binary tree
        slab_info *parent, *lchild, *rchild; // pointers to the parent, left child, and right child slabs

        // constructor
        slab_info(void* ptr = nullptr, uint32_t log_factor = 0, slab_info *parent = nullptr, bool free = true);
    };


    // slab_list class for linked list management
    // to follow the rule of raii, the class won't free or allocate nodes, it only manages the linked list heads
    class slab_list {
        slab_info *head, *free_head;
        public:
        slab_list();
        ~slab_list();

        // insert a slab to the list
        void insert(slab_info* slab);

        // remove a slab from the list
        void remove(slab_info* slab);

        // insert a free slab to the free list
        void insert_free(slab_info* slab);

        // remove a free slab from the free list
        void remove_free(slab_info* slab);

        // get the first free slab in the free list
        slab_info* get_free();

        slab_info* get_head();

        // whether the list has free slabs
        bool has_free() const;

        // reset the head and free_head pointers
        void reset();
    };

    // slab_manager class for pinned memory allocation (only work for size 2^k)
    class slab_manager {
        private:
        uint32_t max_log_factor;
        size_t max_slab_size, base_size;
        std::vector<slab_list> slab_lists;

        // map from the pointer to the slab_info
        // if needed, this can be replaced by adding metadata to the allocated memory
        // but for simplicity, we use a map here
        std::unordered_map<void*, slab_info*> slab_map;


        std::pair<slab_info*, slab_info*> split(uint32_t layer_id);


        void try_merge(slab_info* slab);

        // allocate memory with size max_slab_size
        void *allocate_memory();

        public:

        slab_manager(uint32_t max_log_factor, size_t base_size);

        ~slab_manager();

        void* allocate(uint32_t log_factor);

        void deallocate(void* ptr);

        void clear();

        // try to shrink the memory pool to the minimum size
        void shrink();
    };

} // namespace memory_pool