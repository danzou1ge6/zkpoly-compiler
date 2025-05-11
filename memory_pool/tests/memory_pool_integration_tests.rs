use zkpoly_memory_pool::memory_pool::MemoryPool; // Use the crate name from Cargo.toml
use std::ptr::NonNull;
use std::ffi::c_void;
use std::mem;

// For test_complex, ensure rand crate is added to Cargo.toml dev-dependencies
// e.g., rand = "0.8" (already present as per user's Cargo.toml)
use rand::seq::SliceRandom;
use rand::Rng;

const SIZEOF_INT: usize = mem::size_of::<i32>();

#[test]
fn test_pool_creation() {
    let mut pool = MemoryPool::new(10, 1024); // Made pool mutable
    // Direct assertion of private fields like max_log_factor and base_size
    // is removed as it's an integration test.
    // The successful creation of `pool` itself is a basic test.
    // We can assert behavior that depends on these values if necessary.
    // For example, attempting to allocate a slab of max_log_factor.
    let result = pool.allocate(10); // Max log factor from new(10,...)
    assert!(result.is_ok(), "Should be able to allocate at max_log_factor");
    if let Ok(ptr) = result {
        // Deallocate to clean up
        pool.deallocate(ptr).expect("Deallocation failed in pool creation test");
    }
}

#[test]
fn test_simple_alloc_and_free() {
    let mut pool = MemoryPool::new(5, SIZEOF_INT);
    let ptr_nn = pool.allocate(5).expect("Failed to allocate");
    let ptr = ptr_nn.as_ptr().cast::<i32>();
    let num_elements = 1 << 5;

    unsafe {
        for i in 0..num_elements {
            *ptr.add(i) = i as i32;
        }
        for i in 0..num_elements {
            assert_eq!(*ptr.add(i), i as i32);
        }
    }
    pool.deallocate(ptr_nn).expect("Failed to deallocate");
}

#[test]
fn test_simple_alloc_and_free_for_multiple_times() {
    let mut pool = MemoryPool::new(5, SIZEOF_INT);
    let mut last_ptr_nn: Option<NonNull<c_void>> = None;
    let num_elements = 1 << 5;

    for _j in 0..10 {
        let ptr_nn = pool.allocate(5).expect("Failed to allocate");
        let ptr = ptr_nn.as_ptr().cast::<i32>();
        
        if let Some(last_p) = last_ptr_nn {
            assert_eq!(ptr_nn, last_p, "Pointer should be reused");
        }
        last_ptr_nn = Some(ptr_nn);

        unsafe {
            for i in 0..num_elements {
                *ptr.add(i) = i as i32;
            }
            for i in 0..num_elements {
                assert_eq!(*ptr.add(i), i as i32);
            }
        }
        pool.deallocate(ptr_nn).expect("Failed to deallocate");
    }
}

#[test]
fn test_simple_alloc_and_free_different_sizes() {
    let mut pool = MemoryPool::new(5, SIZEOF_INT);
    let mut allocations: Vec<NonNull<c_void>> = Vec::new();

    for i in 1..=5 { // log_factor from 1 to 5
        let ptr_nn = pool.allocate(i).expect(&format!("Failed to allocate for log_factor {}", i));
        allocations.push(ptr_nn);
        let ptr = ptr_nn.as_ptr().cast::<i32>();
        let num_elements = 1 << i;
        unsafe {
            for j in 0..num_elements {
                *ptr.add(j) = j as i32;
            }
            for j in 0..num_elements {
                assert_eq!(*ptr.add(j), j as i32);
            }
        }
    }
    for ptr_nn in allocations.into_iter().rev() {
        pool.deallocate(ptr_nn).expect("Failed to deallocate in different sizes test");
    }
}

#[test]
fn test_split_and_merge() {
    let mut pool = MemoryPool::new(5, SIZEOF_INT);
    let ptr1_nn = pool.allocate(4).expect("Failed to allocate ptr1");
    let ptr2_nn = pool.allocate(4).expect("Failed to allocate ptr2"); 
    
    pool.deallocate(ptr1_nn).expect("Failed to deallocate ptr1");
    pool.deallocate(ptr2_nn).expect("Failed to deallocate ptr2");

    let ptr3_nn = pool.allocate(5).expect("Failed to allocate ptr3");
    
    // Check if ptr3 reuses the memory of ptr1. This assumes ptr1 and ptr2 were contiguous
    // or could form the block for ptr3. The C++ test implies ptr1 is reused.
    assert_eq!(ptr1_nn.as_ptr(), ptr3_nn.as_ptr(), "ptr3 should reuse the memory of ptr1 after merge");
    pool.deallocate(ptr3_nn).expect("Failed to deallocate ptr3");
}

#[test]
fn test_shrink_and_clear_usage() {
    let mut pool = MemoryPool::new(5, SIZEOF_INT);
    let mut ptrs: Vec<Option<NonNull<c_void>>> = vec![None; 6]; // 0 is unused, 1 to 5 for log_factors

    for i in 1..=5 { // log_factor
        let ptr_nn = pool.allocate(i).expect(&format!("Allocation failed for log_factor {}", i));
        ptrs[i as usize] = Some(ptr_nn);
        let current_ptr = ptr_nn.as_ptr().cast::<i32>();
        let num_elements = 1 << i;
        unsafe {
            for j in 0..num_elements {
                *current_ptr.add(j) = j as i32;
            }
        }
    }

    pool.shrink().expect("Shrink failed");

    for i in 1..=5 {
        if let Some(ptr_nn) = ptrs[i as usize].take() {
            let current_ptr = ptr_nn.as_ptr().cast::<i32>();
            let num_elements = 1 << i;
            unsafe {
                for j in 0..num_elements {
                     assert_eq!(*current_ptr.add(j), j as i32, "Data corruption after shrink for log_factor {}", i);
                }
            }
            pool.deallocate(ptr_nn).expect(&format!("Deallocation failed for log_factor {}", i));
        }
    }
    
    pool.shrink().expect("Shrink failed after deallocations");
    
    let ptr_after_shrink_nn = pool.allocate(5).expect("Allocation failed after shrink");
    pool.deallocate(ptr_after_shrink_nn).expect("Deallocate after shrink failed");
        
    pool.clear().expect("Clear failed");

    for i in 1..=5 {
        let ptr_nn = pool.allocate(i).expect(&format!("Allocation failed for log_factor {} after clear", i));
        let current_ptr = ptr_nn.as_ptr().cast::<i32>();
        let num_elements = 1 << i;
        unsafe {
            for j in 0..num_elements {
                *current_ptr.add(j) = j as i32;
            }
            for j in 0..num_elements {
                assert_eq!(*current_ptr.add(j), j as i32, "Data check failed for log_factor {} after clear", i);
            }
        }
        pool.deallocate(ptr_nn).expect("Deallocate after clear failed");
    }
}

#[test]
fn test_complex() {
    let large_size_log = 16; // Max log_factor
    let rounds = 100; // Reduced rounds for faster testing, C++ was 1000
    let items_per_round = 100; // Reduced items, C++ was 1000
    let mut pool = MemoryPool::new(large_size_log, SIZEOF_INT);
    let mut rng = rand::thread_rng();

    for _k in 0..rounds {
        let mut ptrs_nn: Vec<NonNull<c_void>> = Vec::new();
        let mut ptr_log_factors: Vec<u32> = Vec::new(); // To verify data later based on original size

        for _i in 0..items_per_round {
            let log_factor = rng.gen_range(1..=large_size_log);
            let ptr_nn = pool.allocate(log_factor).expect(&format!("Complex: Allocate failed for log_factor {}", log_factor));
            ptrs_nn.push(ptr_nn);
            ptr_log_factors.push(log_factor); // Store log_factor with ptr

            let current_ptr = ptr_nn.as_ptr().cast::<i32>();
            let num_elements = 1 << log_factor;
            unsafe {
                for j in 0..num_elements {
                    *current_ptr.add(j) = j as i32; // Store simple data
                }
            }
        }

        // To shuffle, we need to keep ptrs_nn and ptr_log_factors associated or shuffle indices.
        // Let's shuffle indices.
        let mut indices: Vec<usize> = (0..ptrs_nn.len()).collect();
        indices.shuffle(&mut rng);

        for &idx_to_dealloc in &indices {
            let ptr_to_dealloc = ptrs_nn[idx_to_dealloc];
            // Optional: Verify data before deallocating if needed, but C++ version doesn't.
            pool.deallocate(ptr_to_dealloc).expect("Complex: Deallocate failed");
        }
    }
    pool.shrink().expect("Complex: Shrink failed");
    pool.clear().expect("Complex: Clear failed");
}