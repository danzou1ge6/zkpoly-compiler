use zkpoly_memory_pool::{CpuMemoryPool, BuddyDiskPool};

use std::mem;

// Helper to get system alignment.
fn get_test_system_alignment() -> usize {
    let dummy_capacity = 4096 * 4; 
    
    match BuddyDiskPool::new(dummy_capacity, None) {
        Ok(pool) => pool.system_alignment(),
        Err(e) => {
            eprintln!("Warning: Could not create dummy BuddyDiskPool to get system alignment: {:?}. Falling back to 4096.", e);
            4096 
        }
    }
}

#[test]
fn test_cpu_direct_to_disk_swap_and_verify() {
    let system_alignment = get_test_system_alignment();
    println!("Test using system_alignment: {}", system_alignment);

    // 1. Initialize Pools
    // Ensure CpuMemoryPool's base_size is a multiple of system_alignment or that allocations naturally align.
    // cudaMallocHost and mmap typically return page-aligned (e.g., 4096 bytes).
    // If system_alignment is <= page_size and page_size is a multiple of system_alignment, then cpu_ptr should be fine.
    let cpu_base_size = system_alignment.max(64); // Ensure base_size is at least system_alignment for easier reasoning, or a common multiple.
    let cpu_max_log_factor = 10; // Max CPU block: cpu_base_size * 2^10
    let mut cpu_pool = CpuMemoryPool::new(cpu_max_log_factor, cpu_base_size);

    let disk_min_block_size = system_alignment; 
    let disk_capacity = disk_min_block_size * 256; 
    let mut disk_pool = BuddyDiskPool::new(disk_capacity, None)
        .expect("Failed to create BuddyDiskPool");

    assert_eq!(disk_pool.system_alignment(), system_alignment, "Disk pool system alignment mismatch");

    // 2. Allocate on CPU for data source
    // We need the I/O length to be a multiple of system_alignment.
    // And the CPU allocation must be at least this I/O length.
    let num_elements_for_io: usize = (system_alignment * 3) / mem::size_of::<u32>(); // Ensure IO is multiple of sys_align
    let io_len_bytes = num_elements_for_io * mem::size_of::<u32>();
    assert_eq!(io_len_bytes % system_alignment, 0, "Chosen io_len_bytes must be multiple of system_alignment");
    
    // Allocate slightly more on CPU if needed, or ensure allocate<T> rounds up to a suitable block.
    // CpuMemoryPool allocates base_size * 2^log_factor.
    // Let's request exactly io_len_bytes for simplicity, assuming CpuMemoryPool handles it.
    let cpu_ptr_src = cpu_pool.allocate::<u8>(io_len_bytes); // Allocate as u8 for direct byte slice
    assert!(!cpu_ptr_src.is_null(), "CPU source allocation returned null pointer");
    assert_eq!((cpu_ptr_src as usize) % system_alignment, 0, 
        "CPU source pointer (0x{:x}) is not aligned to system_alignment ({})", cpu_ptr_src as usize, system_alignment);

    // Fill CPU source buffer
    let original_cpu_slice_u8 = unsafe { std::slice::from_raw_parts_mut(cpu_ptr_src, io_len_bytes) };
    for i in 0..io_len_bytes {
        original_cpu_slice_u8[i] = (i as u8).wrapping_add(0xAA); 
    }

    // 3. Allocate on Disk
    let disk_offset = disk_pool.allocate(io_len_bytes) // Allocate space for io_len_bytes
        .expect("Disk allocation failed");

    // 4. Write directly from CPU buffer to Disk
    // The slice for write must have length `io_len_bytes` (which is aligned)
    // and its pointer `cpu_ptr_src` must be aligned.
    disk_pool.write(disk_offset, original_cpu_slice_u8)
        .expect("Disk write operation failed");

    // 5. Allocate on CPU for data destination (read buffer)
    let cpu_ptr_dst = cpu_pool.allocate::<u8>(io_len_bytes);
    assert!(!cpu_ptr_dst.is_null(), "CPU destination allocation returned null pointer");
    assert_eq!((cpu_ptr_dst as usize) % system_alignment, 0, 
        "CPU destination pointer (0x{:x}) is not aligned to system_alignment ({})", cpu_ptr_dst as usize, system_alignment);
    
    let read_target_slice_u8 = unsafe { std::slice::from_raw_parts_mut(cpu_ptr_dst, io_len_bytes) };
    // Optionally zero out the read buffer first
    read_target_slice_u8.fill(0);


    // 6. Read back from Disk into the second CPU buffer
    disk_pool.read(disk_offset, read_target_slice_u8)
        .expect("Disk read operation failed");

    // 7. Verify data
    assert_eq!(read_target_slice_u8, original_cpu_slice_u8, "Data mismatch after reading back from disk to CPU buffer");

    // 8. Deallocate resources
    cpu_pool.free(cpu_ptr_src);
    cpu_pool.free(cpu_ptr_dst);
    disk_pool.deallocate(disk_offset) // Use the size passed to disk_pool.allocate
        .expect("Disk deallocation failed");

    println!("Integration test 'test_cpu_direct_to_disk_swap_and_verify' passed.");
}


// // Helper for aligned_vec, might still be useful for other tests or if direct CPU alloc is tricky.
// fn aligned_vec(size: usize, alignment: usize, fill_val: u8) -> Vec<u8> {
//     if size == 0 { return Vec::new(); }
//     if alignment == 0 { panic!("Alignment cannot be zero for aligned_vec"); }
//     let layout = Layout::from_size_align(size, alignment).expect("Failed to create layout for aligned_vec");
//     let ptr = unsafe { alloc(layout) };
//     if ptr.is_null() { panic!("Failed to allocate aligned memory for aligned_vec"); }
//     let mut vec = unsafe { Vec::from_raw_parts(ptr, size, size) };
//     vec.fill(fill_val);
//     vec
// }

#[test]
fn test_multiple_direct_swaps() {
    let system_alignment = get_test_system_alignment();
    println!("Test multiple_direct_swaps using system_alignment: {}", system_alignment);

    let mut cpu_pool = CpuMemoryPool::new(10, system_alignment.max(128));
    let disk_min_block = system_alignment;
    let disk_cap = disk_min_block * 512; 
    let mut disk_pool = BuddyDiskPool::new(disk_cap, None)
        .expect("Failed to create BuddyDiskPool for multi-direct-swap test");

    struct AllocationInfoDirect {
        cpu_ptr: *mut u8,
        disk_offset: usize,
        io_len: usize, // This is the aligned length used for I/O and CPU allocation
        original_data_snapshot: Vec<u8>,
    }
    let mut allocations: Vec<AllocationInfoDirect> = Vec::new();

    for i in 0..15 { 
        if i % 4 == 0 && !allocations.is_empty() {
            let idx_to_remove = rand::random::<usize>() % allocations.len();
            let alloc_info = allocations.remove(idx_to_remove);
            
            cpu_pool.free(alloc_info.cpu_ptr);
            disk_pool.deallocate(alloc_info.disk_offset) // Use io_len for dealloc
                .expect(&format!("Disk dealloc failed for offset {} size {}", alloc_info.disk_offset, alloc_info.io_len));
        } else {
            let requested_data_size = (rand::random::<usize>() % (disk_min_block * 3)) + disk_min_block / 2;
            let io_len = (requested_data_size + system_alignment - 1) / system_alignment * system_alignment;
            if io_len == 0 { continue; } // Avoid zero-length I/O

            let cpu_ptr_u8 = cpu_pool.allocate::<u8>(io_len);
            assert!(!cpu_ptr_u8.is_null());
            assert_eq!((cpu_ptr_u8 as usize) % system_alignment, 0, "CPU ptr not sys_aligned in multi_direct_swap");

            let mut current_data_snapshot = Vec::with_capacity(io_len);
            let cpu_slice_u8_mut = unsafe { std::slice::from_raw_parts_mut(cpu_ptr_u8, io_len) };
            for byte_idx in 0..io_len {
                let val = (byte_idx as u8).wrapping_add(i as u8).wrapping_add(0x30);
                cpu_slice_u8_mut[byte_idx] = val;
                current_data_snapshot.push(val);
            }
            
            let disk_offset = disk_pool.allocate(io_len).expect("Disk alloc in multi_direct_swap failed");
            
            disk_pool.write(disk_offset, cpu_slice_u8_mut).expect("Disk write in multi_direct_swap failed");

            allocations.push(AllocationInfoDirect {
                cpu_ptr: cpu_ptr_u8,
                disk_offset,
                io_len,
                original_data_snapshot: current_data_snapshot,
            });
        }
    }

    // Verify all remaining allocations
    for alloc_info in &allocations {
        let read_target_cpu_buf_ptr = cpu_pool.allocate::<u8>(alloc_info.io_len);
        assert!(!read_target_cpu_buf_ptr.is_null());
        assert_eq!((read_target_cpu_buf_ptr as usize) % system_alignment, 0, "Read target CPU ptr not sys_aligned");
        let read_target_slice = unsafe{std::slice::from_raw_parts_mut(read_target_cpu_buf_ptr, alloc_info.io_len)};
        
        disk_pool.read(alloc_info.disk_offset, read_target_slice).expect("Disk read in verification failed");
        assert_eq!(read_target_slice, alloc_info.original_data_snapshot.as_slice(), "Data mismatch during final verification (multi_direct_swap)");
        cpu_pool.free(read_target_cpu_buf_ptr);
    }

    // Cleanup remaining
    for alloc_info in allocations {
        cpu_pool.free(alloc_info.cpu_ptr);
        disk_pool.deallocate(alloc_info.disk_offset).expect("Final disk dealloc failed (multi_direct_swap)");
    }
    println!("Integration test 'test_multiple_direct_swaps' passed.");
}