# Memory Pool

A high-performance memory allocation pool for CUDA pinned memory, implemented similarly to Linux's slab allocator. This library provides efficient memory management for frequently allocated and deallocated memory blocks in CUDA applications.

## Overview

The Memory Pool library provides a slab-based memory allocation system optimized for CUDA pinned memory. It offers both C++ and Rust interfaces, allowing seamless integration into applications written in either language.

Key features:
- Efficient allocation and deallocation of memory blocks
- Reduction of memory fragmentation
- Fast reuse of previously allocated memory blocks
- Configurable block sizes and memory hierarchy
- Support for both C++ and Rust applications

## Implementation Principles

### Slab Allocation Mechanism

The memory pool employs a slab allocation strategy similar to the Linux kernel's slab allocator. The core principles include:

1. **Hierarchical Block Organization**: Memory is organized in layers of different-sized blocks. The largest blocks are at the top level, with each level below containing blocks half the size of the level above.

2. **Binary Tree Structure**: Blocks are managed in a binary tree structure, where each parent node can be split into two child nodes of half the size.

3. **On-Demand Splitting**: When memory is requested, if no free block of the appropriate size is available, a larger block is split recursively until a suitable-sized block is available.

4. **Automatic Merging**: When blocks are freed, adjacent free blocks of the same size are automatically merged into larger blocks, reducing fragmentation.

5. **Linked List Management**: Each layer maintains two linked lists: one for all slabs in the layer and another for free slabs, allowing for efficient allocation and deallocation.

## C++ Interface

### `slab_manager` Class

The main class that manages memory allocation and deallocation.

```cpp
namespace memory_pool {
    class slab_manager {
    public:
        // Constructor: initializes a slab manager with specified parameters
        // - max_log_factor: Maximum power of 2 for block size (e.g., 10 means largest block is 2^10 * base_size)
        // - base_size: Base unit size for memory blocks (in bytes)
        slab_manager(uint32_t max_log_factor, size_t base_size);
        
        // Destructor: cleans up all allocated memory
        ~slab_manager();
        
        // Allocates a block of memory of size 2^log_factor * base_size
        // Returns pointer to the allocated memory block
        void* allocate(uint32_t log_factor);
        
        // Deallocates a previously allocated memory block
        void deallocate(void* ptr);
        
        // Releases all allocated memory
        void clear();
        
        // Attempts to release unused large memory blocks
        void shrink();
    };
}
```

### Usage Example (C++)

```cpp
#include "memory_pool.h"

int main() {
    // Create a slab manager with max block size 2^5 * sizeof(int)
    memory_pool::slab_manager slab_manager(5, sizeof(int));
    
    // Allocate memory block of size 2^3 * sizeof(int)
    int* buffer = (int*)slab_manager.allocate(3);
    
    // Use the allocated memory
    for (int i = 0; i < (1 << 3); i++) {
        buffer[i] = i;
    }
    
    // Deallocate the memory when done
    slab_manager.deallocate(buffer);
    
    return 0;
}
```

## Rust Interface

### `PinnedMemoryPool` Struct

The Rust interface wraps the C++ implementation through FFI.

```rust
pub struct PinnedMemoryPool {
    // Internal fields (not meant to be accessed directly)
}

impl PinnedMemoryPool {
    // Creates a new memory pool with specified parameters
    // - max_log_factor: Maximum power of 2 for block size
    // - base_size: Base unit size for memory blocks (in bytes)
    pub fn new(max_log_factor: u32, base_size: usize) -> Self;
    
    // Allocates memory for an array of type T with specified length
    // Returns pointer to the allocated memory
    pub fn allocate<T: Sized>(&self, len: usize) -> *mut T;
    
    // Frees previously allocated memory
    pub fn free<T: Sized>(&self, ptr: *mut T);
    
    // Releases all allocated memory
    pub fn clear(&self);
    
    // Attempts to release unused large memory blocks
    pub fn shrink(&self);
}

// Automatically destroys the memory pool when it goes out of scope
impl Drop for PinnedMemoryPool { /* ... */ }
```

### Usage Example (Rust)

```rust
use memory_pool::PinnedMemoryPool;

fn main() {
    // Create a memory pool with max block size 2^16 * sizeof(u32)
    let pool = PinnedMemoryPool::new(16, std::mem::size_of::<u32>());
    
    // Allocate memory for 100 u32 values
    let ptr = pool.allocate::<u32>(100);
    
    unsafe {
        // Use the allocated memory
        let slice = std::slice::from_raw_parts_mut(ptr, 100);
        for (i, val) in slice.iter_mut().enumerate() {
            *val = i as u32;
        }
    }
    
    // Free the memory when done
    pool.free(ptr);
    
    // The pool will be automatically cleaned up when it goes out of scope
}
```

## Implementation Details

### Memory Organization

The memory pool organizes blocks in a hierarchical structure:

1. The largest blocks (of size `2^max_log_factor * base_size`) are allocated directly from the system using CUDA pinned memory.

2. When a smaller block is needed, the system finds the smallest available block that can accommodate the request. If the block is larger than needed, it is split into two equal-sized blocks recursively until an appropriate size is reached.

3. When blocks are freed, the system checks if the buddy block (the other half of the parent) is also free. If so, they are merged back into a larger block, and this process continues up the tree.

### Binary Tree Structure

The implementation uses a binary tree to manage the block hierarchy:

- Each node in the tree represents a memory block.
- Inner nodes have two children that represent the two halves of the parent block.
- Leaf nodes represent blocks that are currently allocated or ready to be allocated.

### Linked List Management

For each layer (block size), two doubly-linked lists are maintained:

1. A list of all blocks in that layer, regardless of their allocation status.
2. A list of only the free blocks in that layer, for quick access during allocation.

## Performance Considerations

- The memory pool is most effective for applications that frequently allocate and deallocate memory of similar sizes.
- It reduces the overhead of system calls to allocate/deallocate CUDA pinned memory.
- The hierarchical nature of the allocator might lead to internal fragmentation, but this is typically negligible compared to the performance benefits.

## Limitations

- Only power-of-2 sized allocations are efficient. Non-power-of-2 sizes will be rounded up to the next power of 2.
- The maximum block size is determined at initialization and cannot be changed during runtime.

## Future Improvements

Potential areas for enhancement:

- Support for non-power-of-2 allocations
- Cache-aware alignment options
- Thread-safe operations for multi-threaded applications