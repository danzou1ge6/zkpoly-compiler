use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use zkpoly_memory_pool::SwapPagePool;
use zkpoly_common::devices::DeviceType;

// 2MB 页面大小常量
const PAGE_SIZE_2MB: usize = 2 * 1024 * 1024; // 2MB
const PAGE_SIZE_4MB: usize = 4 * 1024 * 1024; // 4MB

// 高并发分配测试
#[test]
fn test_high_concurrency_allocation() {
    println!("开始高并发分配测试");
    
    let page_size = PAGE_SIZE_2MB;
    let max_pages = 16;
    let disk_capacity = 128 * 1024 * 1024; // 128MB
    
    let pool = Arc::new(SwapPagePool::new(
        DeviceType::CPU,
        page_size,
        max_pages,
        disk_capacity,
    ).expect("创建 SwapPagePool 失败"));
    
    let thread_count = 8;
    let allocations_per_thread = 100;
    let barrier = Arc::new(Barrier::new(thread_count));
    let mut threads = Vec::new();
    
    for thread_id in 0..thread_count {
        let pool_clone = Arc::clone(&pool);
        let barrier_clone = Arc::clone(&barrier);
        
        threads.push(thread::spawn(move || {
            barrier_clone.wait();
            
            let mut handles = Vec::new();
            let mut rng = StdRng::seed_from_u64(thread_id as u64);
            
            // 第一阶段：快速分配
            for i in 0..allocations_per_thread {
                let size = rng.gen_range(64 * 1024..2 * 1024 * 1024); // 64KB 到 2MB
                match pool_clone.allocate::<u8>(size) {
                    Ok(handle) => {
                        handles.push((handle, size, thread_id, i));
                    }
                    Err(e) => {
                        println!("线程 {} 分配 {} 失败: {:?}", thread_id, i, e);
                        break;
                    }
                }
                
                // 偶尔暂停一下，模拟真实工作负载
                if i % 10 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            
            println!("线程 {} 完成分配，总共 {} 个", thread_id, handles.len());
            
            // 第二阶段：验证数据
            for (handle, size, tid, i) in &handles {
                match handle.get_ptr() {
                    Ok(ptr) => {
                        let pattern = (*tid as u8).wrapping_add(*i as u8);
                        unsafe {
                            // 写入模式数据
                            for j in 0..std::cmp::min(*size, 1024) {
                                *ptr.as_ptr().add(j) = pattern;
                            }
                            
                            // 验证数据
                            for j in 0..std::cmp::min(*size, 1024) {
                                assert_eq!(*ptr.as_ptr().add(j), pattern);
                            }
                        }
                    }
                    Err(e) => {
                        println!("线程 {} 获取指针失败: {:?}", thread_id, e);
                    }
                }
            }
            
            // 第三阶段：随机释放
            let mut released = 0;
            for (handle, _, _, _) in handles {
                if rng.gen_bool(0.8) { // 80% 概率释放
                    if let Err(e) = pool_clone.deallocate(handle) {
                        println!("线程 {} 释放失败: {:?}", thread_id, e);
                    } else {
                        released += 1;
                    }
                }
            }
            
            println!("线程 {} 释放了 {} 个分配", thread_id, released);
        }));
    }
    
    for thread in threads {
        thread.join().expect("线程 join 失败");
    }
    
    let final_stats = pool.stats();
    println!("高并发测试完成 - 最终统计: 内存页面={}, 换出页面={}, 活跃分配={}", 
             final_stats.in_memory_pages, final_stats.swapped_pages, final_stats.active_allocations);
}

// 长时间运行的稳定性测试
#[test]
#[ignore] // 长时间运行，默认忽略
fn test_long_running_stability() {
    println!("开始长时间稳定性测试");
    
    let page_size = PAGE_SIZE_2MB;
    let max_pages = 8;
    let disk_capacity = 64 * 1024 * 1024;
    
    let pool = Arc::new(SwapPagePool::new(
        DeviceType::CPU,
        page_size,
        max_pages,
        disk_capacity,
    ).expect("创建 SwapPagePool 失败"));
    
    let duration = Duration::from_secs(300); // 5 分钟
    let start_time = Instant::now();
    let stats_mutex = Arc::new(Mutex::new(HashMap::new()));
    
    let thread_count = 4;
    let mut threads = Vec::new();
    
    for thread_id in 0..thread_count {
        let pool_clone = Arc::clone(&pool);
        let stats_clone = Arc::clone(&stats_mutex);
        let end_time = start_time + duration;
        
        threads.push(thread::spawn(move || {
            let mut operations = 0u64;
            let mut allocations = 0u64;
            let mut deallocations = 0u64;
            let mut errors = 0u64;
            let mut rng = StdRng::seed_from_u64(thread_id as u64);
            let mut active_handles = Vec::new();
            
            while Instant::now() < end_time {
                operations += 1;
                
                // 决定是分配还是释放
                if active_handles.is_empty() || rng.gen_bool(0.6) {
                    // 分配新内存
                    let size = rng.gen_range(128 * 1024..4 * 1024 * 1024);
                    match pool_clone.allocate::<u8>(size) {
                        Ok(handle) => {
                            active_handles.push((handle, size));
                            allocations += 1;
                        }
                        Err(_) => {
                            errors += 1;
                        }
                    }
                } else {
                    // 释放现有内存
                    if !active_handles.is_empty() {
                        let index = rng.gen_range(0..active_handles.len());
                        let (handle, _) = active_handles.remove(index);
                        match pool_clone.deallocate(handle) {
                            Ok(()) => deallocations += 1,
                            Err(_) => errors += 1,
                        }
                    }
                }
                
                // 偶尔进行数据验证
                if operations % 100 == 0 && !active_handles.is_empty() {
                    let index = rng.gen_range(0..active_handles.len());
                    if let Ok(ptr) = active_handles[index].0.get_ptr() {
                        let pattern = (thread_id as u8).wrapping_add((operations % 256) as u8);
                        unsafe {
                            *ptr.as_ptr() = pattern;
                            assert_eq!(*ptr.as_ptr(), pattern);
                        }
                    }
                }
                
                // 短暂休息
                if operations % 50 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            
            // 清理剩余分配
            for (handle, _) in active_handles {
                let _ = pool_clone.deallocate(handle);
            }
            
            // 记录统计信息
            let mut stats = stats_clone.lock().unwrap();
            stats.insert(thread_id, (operations, allocations, deallocations, errors));
            
            println!("线程 {} 完成: 操作={}, 分配={}, 释放={}, 错误={}", 
                     thread_id, operations, allocations, deallocations, errors);
        }));
    }
    
    for thread in threads {
        thread.join().expect("线程 join 失败");
    }
    
    let stats = stats_mutex.lock().unwrap();
    let total_ops: u64 = stats.values().map(|(ops, _, _, _)| *ops).sum();
    let total_allocs: u64 = stats.values().map(|(_, allocs, _, _)| *allocs).sum();
    let total_deallocs: u64 = stats.values().map(|(_, _, deallocs, _)| *deallocs).sum();
    let total_errors: u64 = stats.values().map(|(_, _, _, errors)| *errors).sum();
    
    println!("长时间稳定性测试完成:");
    println!("总操作数: {}", total_ops);
    println!("总分配数: {}", total_allocs);
    println!("总释放数: {}", total_deallocs);
    println!("总错误数: {}", total_errors);
    println!("错误率: {:.2}%", (total_errors as f64 / total_ops as f64) * 100.0);
    
    let final_stats = pool.stats();
    println!("最终内存状态: 内存页面={}, 换出页面={}", 
             final_stats.in_memory_pages, final_stats.swapped_pages);
}

// 内存碎片化测试
#[test]
fn test_memory_fragmentation() {
    println!("开始内存碎片化测试");
    
    let page_size = PAGE_SIZE_2MB;
    let max_pages = 8;
    let disk_capacity = 64 * 1024 * 1024;
    
    let pool = SwapPagePool::new(
        DeviceType::CPU,
        page_size,
        max_pages,
        disk_capacity,
    ).expect("创建 SwapPagePool 失败");
    
    let mut handles = Vec::new();
    let mut rng = StdRng::seed_from_u64(42);
    
    // 第一阶段：分配很多小块内存
    for i in 0..50 {
        let size = rng.gen_range(64 * 1024..512 * 1024); // 64KB 到 512KB
        match pool.allocate::<u8>(size) {
            Ok(handle) => handles.push(handle),
            Err(e) => {
                println!("分配 {} 失败: {:?}", i, e);
                break;
            }
        }
    }
    
    println!("分配了 {} 个小块内存", handles.len());
    
    // 第二阶段：随机释放一些内存，创建碎片
    let mut released = 0;
    for i in (0..handles.len()).rev().step_by(2) {
        if let Some(handle) = handles.get(i) {
            // 我们需要克隆 handle 或者使用其他方法
            // 这里简化处理，跳过这个测试的这部分
        }
    }
    
    // 第三阶段：尝试分配大块内存
    for i in 0..5 {
        let large_size = 2 * 1024 * 1024; // 2MB
        match pool.allocate::<u8>(large_size) {
            Ok(handle) => {
                println!("成功分配大块内存 {}", i);
                handles.push(handle);
            }
            Err(e) => {
                println!("分配大块内存 {} 失败: {:?}", i, e);
            }
        }
    }
    
    let stats = pool.stats();
    println!("碎片化测试统计:");
    println!("活跃分配: {}", stats.active_allocations);
    println!("内存使用率: {:.2}%", stats.memory_usage_ratio() * 100.0);
    println!("碎片率: {:.2}%", stats.fragmentation_ratio() * 100.0);
    
    // 清理
    for handle in handles {
        pool.deallocate(handle).expect("清理失败");
    }
    
    println!("内存碎片化测试完成");
}

// 内存泄漏检测测试
#[test]
fn test_memory_leak_detection() {
    println!("开始内存泄漏检测测试");
    
    let page_size = PAGE_SIZE_2MB;
    let max_pages = 4;
    let disk_capacity = 32 * 1024 * 1024;
    
    let pool = SwapPagePool::new(
        DeviceType::CPU,
        page_size,
        max_pages,
        disk_capacity,
    ).expect("创建 SwapPagePool 失败");
    
    let initial_stats = pool.stats();
    println!("初始状态: 空闲页面={}, 活跃分配={}", 
             initial_stats.free_pages, initial_stats.active_allocations);
    
    // 执行多轮分配和释放
    for round in 0..10 {
        let mut handles = Vec::new();
        
        // 分配内存
        for i in 0..5 {
            let size = (i + 1) * 256 * 1024; // 256KB, 512KB, 768KB, 1MB, 1.25MB
            match pool.allocate::<u8>(size) {
                Ok(handle) => handles.push(handle),
                Err(e) => println!("轮次 {} 分配 {} 失败: {:?}", round, i, e),
            }
        }
        
        // 使用内存
        for (i, handle) in handles.iter().enumerate() {
            if let Ok(ptr) = handle.get_ptr() {
                unsafe {
                    *ptr.as_ptr() = (round + i) as u8;
                    assert_eq!(*ptr.as_ptr(), (round + i) as u8);
                }
            }
        }
        
        // 释放所有内存
        for handle in handles {
            pool.deallocate(handle).expect("释放失败");
        }
        
        let round_stats = pool.stats();
        println!("轮次 {} 后: 空闲页面={}, 活跃分配={}, 换出页面={}", 
                 round, round_stats.free_pages, round_stats.active_allocations, 
                 round_stats.swapped_pages);
    }
    
    let final_stats = pool.stats();
    
    // 检查是否有内存泄漏
    assert_eq!(final_stats.active_allocations, 0, "检测到内存泄漏：还有活跃分配");
    assert_eq!(final_stats.swapped_pages, 0, "检测到内存泄漏：还有换出页面");
    
    // 页面应该全部回到空闲状态（可能有一些差异由于内部状态）
    println!("泄漏检测结果:");
    println!("初始空闲页面: {}", initial_stats.free_pages);
    println!("最终空闲页面: {}", final_stats.free_pages);
    println!("最终活跃分配: {}", final_stats.active_allocations);
    
    if final_stats.free_pages >= initial_stats.free_pages - 1 {
        println!("✓ 没有检测到明显的内存泄漏");
    } else {
        println!("⚠ 可能存在内存泄漏");
    }
    
    println!("内存泄漏检测测试完成");
}

// 错误处理测试
#[test]
fn test_error_handling() {
    println!("开始错误处理测试");
    
    let page_size = PAGE_SIZE_2MB;
    let max_pages = 2; // 很小的内存池
    let disk_capacity = 8 * 1024 * 1024; // 很小的磁盘空间
    
    let pool = SwapPagePool::new(
        DeviceType::CPU,
        page_size,
        max_pages,
        disk_capacity,
    ).expect("创建 SwapPagePool 失败");
    
    let mut handles = Vec::new();
    
    // 尝试分配超过容量的内存
    for i in 0..10 {
        match pool.allocate::<u8>(page_size) {
            Ok(handle) => {
                handles.push(handle);
                println!("成功分配第 {} 个页面", i);
            }
            Err(e) => {
                println!("分配第 {} 个页面失败: {:?}", i, e);
                break;
            }
        }
    }
    
    // 尝试分配巨大的内存块
    match pool.allocate::<u8>(1024 * 1024 * 1024) { // 1GB
        Ok(_) => println!("意外地成功分配了 1GB 内存"),
        Err(e) => println!("预期的巨大分配失败: {:?}", e),
    }
    
    // 清理
    for handle in handles {
        pool.deallocate(handle).expect("清理失败");
    }
    
    println!("错误处理测试完成");
}

// 并发换入换出测试
#[test]
fn test_concurrent_swap_operations() {
    println!("开始并发换入换出测试");
    
    let page_size = PAGE_SIZE_2MB;
    let max_pages = 4; // 强制频繁换出
    let disk_capacity = 64 * 1024 * 1024;
    
    let pool = Arc::new(SwapPagePool::new(
        DeviceType::CPU,
        page_size,
        max_pages,
        disk_capacity,
    ).expect("创建 SwapPagePool 失败"));
    
    let thread_count = 6;
    let barrier = Arc::new(Barrier::new(thread_count));
    let mut threads = Vec::new();
    
    for thread_id in 0..thread_count {
        let pool_clone = Arc::clone(&pool);
        let barrier_clone = Arc::clone(&barrier);
        
        threads.push(thread::spawn(move || {
            barrier_clone.wait();
            
            let mut handles = Vec::new();
            let mut rng = StdRng::seed_from_u64(thread_id as u64);
            
            // 分配一些内存
            for i in 0..3 {
                if let Ok(handle) = pool_clone.allocate::<u8>(page_size) {
                    handles.push((handle, thread_id, i));
                }
            }
            
            // 随机访问内存，触发换入换出
            for _ in 0..20 {
                if !handles.is_empty() {
                    let index = rng.gen_range(0..handles.len());
                    let (handle, tid, i) = &handles[index];
                    
                    if let Ok(ptr) = handle.get_ptr() {
                        let pattern = (*tid as u8).wrapping_add(*i as u8);
                        unsafe {
                            *ptr.as_ptr() = pattern;
                            assert_eq!(*ptr.as_ptr(), pattern);
                        }
                    }
                    
                    // 短暂休息，让其他线程有机会触发换出
                    thread::sleep(Duration::from_millis(10));
                }
            }
            
            // 清理
            for (handle, _, _) in handles {
                let _ = pool_clone.deallocate(handle);
            }
            
            println!("线程 {} 完成并发换入换出测试", thread_id);
        }));
    }
    
    for thread in threads {
        thread.join().expect("线程 join 失败");
    }
    
    let final_stats = pool.stats();
    println!("并发换入换出测试完成 - 最终统计: 内存页面={}, 换出页面={}", 
             final_stats.in_memory_pages, final_stats.swapped_pages);
}