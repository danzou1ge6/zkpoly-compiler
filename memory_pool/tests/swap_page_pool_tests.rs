use rand::Rng;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use zkpoly_common::devices::DeviceType;
use zkpoly_memory_pool::SwapPagePool;

// 2MB 页面大小常量
const PAGE_SIZE_2MB: usize = 2 * 1024 * 1024; // 2MB

// 基本分配和释放测试
#[test]
fn test_basic_allocation_and_deallocation() {
    println!("测试基本分配和释放功能");

    // 创建一个小的内存池用于测试
    let page_size = PAGE_SIZE_2MB; // 2MB 页面
    let max_pages = 4; // 最多 4 个页面在内存中 (8MB)
    let disk_capacity = 32 * 1024 * 1024; // 32MB 磁盘空间

    let pool = SwapPagePool::new(DeviceType::CPU, page_size, max_pages, disk_capacity)
        .expect("创建 SwapPagePool 失败");

    // 测试小分配 (1MB, 小于页面大小)
    let handle1 = pool.allocate::<u8>(1024 * 1024).expect("分配 1MB 失败");
    let handle2 = pool.allocate::<u8>(1536 * 1024).expect("分配 1.5MB 失败");

    // 获取指针并写入数据
    {
        let ptr1 = handle1.get_ptr().expect("获取 ptr1 失败");
        let ptr2 = handle2.get_ptr().expect("获取 ptr2 失败");

        unsafe {
            // 写入测试数据
            for i in 0..(1024 * 1024) {
                *ptr1.as_ptr().add(i) = (i % 256) as u8;
            }
            for i in 0..(1536 * 1024) {
                *ptr2.as_ptr().add(i) = ((i + 128) % 256) as u8;
            }

            // 验证数据
            for i in 0..(1024 * 1024) {
                assert_eq!(*ptr1.as_ptr().add(i), (i % 256) as u8);
            }
            for i in 0..(1536 * 1024) {
                assert_eq!(*ptr2.as_ptr().add(i), ((i + 128) % 256) as u8);
            }
        }

        // ptr1 和 ptr2 在这里会被自动释放
    }

    // 释放分配
    pool.deallocate(handle1).expect("释放 handle1 失败");
    pool.deallocate(handle2).expect("释放 handle2 失败");

    println!("基本分配和释放测试通过");
}

// 测试大分配 (跨多个页面)
#[test]
fn test_large_allocation() {
    println!("测试大分配功能");

    let page_size = PAGE_SIZE_2MB;
    let max_pages = 6;
    let disk_capacity = 64 * 1024 * 1024; // 64MB

    let pool = SwapPagePool::new(DeviceType::CPU, page_size, max_pages, disk_capacity)
        .expect("创建 SwapPagePool 失败");

    // 分配一个需要多个页面的大内存块 (5MB)
    let large_size = 5 * 1024 * 1024;
    let handle = pool.allocate::<u32>(large_size).expect("分配 5MB 失败");

    {
        let ptr = handle.get_ptr().expect("获取大内存指针失败");
        let element_count = large_size / 4; // u32 是 4 字节

        unsafe {
            // 写入递增的数据
            for i in 0..element_count {
                *ptr.as_ptr().add(i) = i as u32;
            }

            // 验证数据
            for i in 0..element_count {
                assert_eq!(*ptr.as_ptr().add(i), i as u32);
            }
        }
    }

    pool.deallocate(handle).expect("释放大内存失败");

    println!("大分配测试通过");
}

// 测试内存压力和换出机制
#[test]
fn test_memory_pressure_and_swapping() {
    println!("测试内存压力和换出机制");

    let page_size = PAGE_SIZE_2MB;
    let max_pages = 3; // 只允许 3 个页面在内存中 (6MB)
    let disk_capacity = 32 * 1024 * 1024; // 32MB 磁盘空间

    let pool = SwapPagePool::new(DeviceType::CPU, page_size, max_pages, disk_capacity)
        .expect("创建 SwapPagePool 失败");

    let mut handles = Vec::new();
    let alloc_size = page_size; // 每次分配一个页面大小

    // 分配超过内存限制的页面，强制发生换出
    for i in 0..6 {
        let handle = pool
            .allocate::<u8>(alloc_size)
            .expect(&format!("分配第 {} 个页面失败", i));

        // 写入数据到每个分配
        {
            let ptr = handle.get_ptr().expect(&format!("获取第 {} 个指针失败", i));
            unsafe {
                for j in 0..alloc_size {
                    *ptr.as_ptr().add(j) = ((i * 100 + j) % 256) as u8;
                }
            }
        }

        handles.push(handle);

        // 检查统计信息
        let stats = pool.stats();
        println!(
            "分配第 {} 个页面后: 内存页面={}, 换出页面={}, 空闲页面={}, 活跃分配={}",
            i,
            stats.in_memory_pages,
            stats.swapped_pages,
            stats.free_pages,
            stats.active_allocations
        );
    }

    // 验证所有数据仍然正确 (这会触发换回操作)
    for (i, handle) in handles.iter().enumerate() {
        let ptr = handle
            .get_ptr()
            .expect(&format!("重新获取第 {} 个指针失败", i));
        unsafe {
            for j in 0..alloc_size {
                let expected = ((i * 100 + j) % 256) as u8;
                let actual = *ptr.as_ptr().add(j);
                assert_eq!(
                    actual, expected,
                    "第 {} 个分配的第 {} 个字节数据不匹配",
                    i, j
                );
            }
        }
    }

    // 释放所有分配
    for (i, handle) in handles.into_iter().enumerate() {
        pool.deallocate(handle)
            .expect(&format!("释放第 {} 个分配失败", i));
    }

    println!("内存压力和换出测试通过");
}

// 测试 LRU 更新机制
#[test]
fn test_lru_update() {
    println!("测试 LRU 更新机制");

    let page_size = PAGE_SIZE_2MB;
    let max_pages = 2;
    let disk_capacity = 16 * 1024 * 1024;

    let pool = SwapPagePool::new(DeviceType::CPU, page_size, max_pages, disk_capacity)
        .expect("创建 SwapPagePool 失败");

    // 分配三个页面
    let handle1 = pool.allocate::<u8>(page_size).expect("分配页面1失败");
    let handle2 = pool.allocate::<u8>(page_size).expect("分配页面2失败");
    let handle3 = pool.allocate::<u8>(page_size).expect("分配页面3失败");

    // 访问第一个页面使其成为最近使用的
    {
        let ptr1 = handle1.get_ptr().expect("获取 ptr1 失败");
        unsafe {
            *ptr1.as_ptr() = 42;
        }
    }

    // 分配第四个页面，这应该会换出最少使用的页面
    let handle4 = pool.allocate::<u8>(page_size).expect("分配页面4失败");

    // 验证数据仍然可以访问
    {
        let ptr1 = handle1.get_ptr().expect("重新获取 ptr1 失败");
        unsafe {
            assert_eq!(*ptr1.as_ptr(), 42);
        }
    }

    // 清理
    pool.deallocate(handle1).expect("释放 handle1 失败");
    pool.deallocate(handle2).expect("释放 handle2 失败");
    pool.deallocate(handle3).expect("释放 handle3 失败");
    pool.deallocate(handle4).expect("释放 handle4 失败");

    println!("LRU 更新测试通过");
}

// 多线程测试
#[test]
fn test_multithreaded_access() {
    println!("测试多线程访问");

    let page_size = PAGE_SIZE_2MB;
    let max_pages = 4;
    let disk_capacity = 32 * 1024 * 1024;

    let pool = Arc::new(
        SwapPagePool::new(DeviceType::CPU, page_size, max_pages, disk_capacity)
            .expect("创建 SwapPagePool 失败"),
    );

    let barrier = Arc::new(Barrier::new(4));
    let mut threads = Vec::new();

    for thread_id in 0..4 {
        let pool_clone = Arc::clone(&pool);
        let barrier_clone = Arc::clone(&barrier);

        threads.push(thread::spawn(move || {
            barrier_clone.wait();

            // 每个线程分配和使用不同的内存
            let handle = pool_clone
                .allocate::<u64>(1024 * 1024)
                .expect(&format!("线程 {} 分配失败", thread_id));

            {
                let ptr = handle
                    .get_ptr()
                    .expect(&format!("线程 {} 获取指针失败", thread_id));
                let element_count = (1024 * 1024) / 8; // u64 是 8 字节

                unsafe {
                    // 写入线程 ID 模式的数据
                    for i in 0..element_count {
                        *ptr.as_ptr().add(i) = (thread_id as u64) << 32 | (i as u64);
                    }

                    // 验证数据
                    for i in 0..element_count {
                        let expected = (thread_id as u64) << 32 | (i as u64);
                        assert_eq!(*ptr.as_ptr().add(i), expected);
                    }
                }
            }

            // 模拟一些工作
            thread::sleep(Duration::from_millis(10));

            pool_clone
                .deallocate(handle)
                .expect(&format!("线程 {} 释放失败", thread_id));
        }));
    }

    for thread in threads {
        thread.join().expect("线程 join 失败");
    }

    println!("多线程访问测试通过");
}

// 压力测试
#[test]
fn test_stress() {
    println!("开始压力测试");

    let page_size = PAGE_SIZE_2MB;
    let max_pages = 8;
    let disk_capacity = 128 * 1024 * 1024; // 128MB

    let pool = SwapPagePool::new(DeviceType::CPU, page_size, max_pages, disk_capacity)
        .expect("创建 SwapPagePool 失败");

    let mut rng = rand::thread_rng();
    let rounds = 50;
    let allocations_per_round = 20;

    for round in 0..rounds {
        let mut handles = Vec::new();

        // 随机大小的分配
        for i in 0..allocations_per_round {
            let size = rng.gen_range(64 * 1024..4 * 1024 * 1024); // 64KB 到 4MB
            let handle = pool
                .allocate::<u8>(size)
                .expect(&format!("轮次 {} 分配 {} 失败", round, i));
            let pattern = (round * allocations_per_round + i) as u8;

            // 写入和验证数据
            {
                let ptr = handle
                    .get_ptr()
                    .expect(&format!("轮次 {} 获取指针 {} 失败", round, i));
                unsafe {
                    for j in 0..size {
                        *ptr.as_ptr().add(j) = pattern;
                    }

                    // 验证一部分数据
                    for j in 0..std::cmp::min(size, 1024) {
                        assert_eq!(*ptr.as_ptr().add(j), pattern);
                    }
                }
            }

            handles.push((handle, size, pattern));
            let stats = pool.stats();
            println!(
                "轮次 {}: 内存页面={}, 换出页面={}, 内存使用率={:.2}%",
                round,
                stats.in_memory_pages,
                stats.swapped_pages,
                stats.memory_usage_ratio() * 100.0
            );
        }

        // 验证剩余的分配
        for (handle, size, pattern) in &handles {
            let ptr = handle.get_ptr().expect("重新获取指针失败");
            unsafe {
                for j in 0..std::cmp::min(*size, 1024) {
                    assert_eq!(*ptr.as_ptr().add(j), *pattern);
                }
            }
        }

        let stats = pool.stats();
        println!(
            "轮次 {}: 内存页面={}, 换出页面={}, 内存使用率={:.2}%",
            round,
            stats.in_memory_pages,
            stats.swapped_pages,
            stats.memory_usage_ratio() * 100.0
        );

        // 释放分配
        for (handle, _, _) in handles {
            println!("releasing handle{:?}", handle);
            pool.deallocate(handle).expect("释放分配失败");
        }

        let stats = pool.stats();
        println!(
            "轮次 {}: 内存页面={}, 换出页面={}, 内存使用率={:.2}%",
            round,
            stats.in_memory_pages,
            stats.swapped_pages,
            stats.memory_usage_ratio() * 100.0
        );
    }

    println!("压力测试完成");
}

// 测试统计信息
#[test]
fn test_statistics() {
    println!("测试统计信息");

    let page_size = PAGE_SIZE_2MB;
    let max_pages = 4;
    let disk_capacity = 32 * 1024 * 1024;

    let pool = SwapPagePool::new(DeviceType::CPU, page_size, max_pages, disk_capacity)
        .expect("创建 SwapPagePool 失败");

    // 初始统计
    let initial_stats = pool.stats();
    assert_eq!(initial_stats.total_pages, max_pages);
    assert_eq!(initial_stats.free_pages, max_pages);
    assert_eq!(initial_stats.in_memory_pages, 0);
    assert_eq!(initial_stats.swapped_pages, 0);
    assert_eq!(initial_stats.active_allocations, 0);
    assert_eq!(initial_stats.page_size, page_size);

    // 分配一些内存
    let handle1 = pool.allocate::<u8>(page_size).expect("分配失败");
    let handle2 = pool.allocate::<u8>(page_size).expect("分配失败");

    let stats_after_alloc = pool.stats();
    assert!(stats_after_alloc.active_allocations >= 2);
    assert!(stats_after_alloc.in_memory_pages >= 2);
    assert!(stats_after_alloc.free_pages <= initial_stats.free_pages);

    // 测试内存使用率计算
    assert!(stats_after_alloc.memory_usage_ratio() > 0.0);
    assert!(stats_after_alloc.memory_usage_ratio() <= 1.0);

    // 清理
    pool.deallocate(handle1).expect("释放失败");
    pool.deallocate(handle2).expect("释放失败");

    println!("统计信息测试通过");
}

// 测试强制换出
#[test]
fn test_force_eviction() {
    println!("测试强制换出");

    let page_size = PAGE_SIZE_2MB;
    let max_pages = 4;
    let disk_capacity = 32 * 1024 * 1024;

    let pool = SwapPagePool::new(DeviceType::CPU, page_size, max_pages, disk_capacity)
        .expect("创建 SwapPagePool 失败");

    // 填满内存
    let mut handles = Vec::new();
    for i in 0..max_pages {
        let handle = pool
            .allocate::<u8>(page_size)
            .expect(&format!("分配 {} 失败", i));
        handles.push(handle);
    }

    // 确保所有页面都被标记为可交换（通过释放指针引用）
    // 这里我们需要确保没有活跃的 MemoryPtr

    // 强制换出 2 个页面
    let evicted = pool.force_evict(2).expect("强制换出失败");
    println!("成功换出 {} 个页面", evicted);

    let stats = pool.stats();
    assert!(stats.swapped_pages >= evicted);

    // 清理
    for handle in handles {
        pool.deallocate(handle).expect("释放失败");
    }

    println!("强制换出测试通过");
}

// 性能基准测试
#[test]
#[ignore] // 标记为 ignore，只在需要时运行
fn benchmark_allocation_performance() {
    println!("开始性能基准测试");

    let page_size = PAGE_SIZE_2MB;
    let max_pages = 16;
    let disk_capacity = 256 * 1024 * 1024;

    let pool = SwapPagePool::new(DeviceType::CPU, page_size, max_pages, disk_capacity)
        .expect("创建 SwapPagePool 失败");

    let iterations = 1000;
    let allocation_size = 1024 * 1024; // 1MB

    let start = Instant::now();

    for i in 0..iterations {
        let handle = pool
            .allocate::<u8>(allocation_size)
            .expect(&format!("分配 {} 失败", i));

        // 简单的写入和读取
        {
            let ptr = handle.get_ptr().expect("获取指针失败");
            unsafe {
                *ptr.as_ptr() = (i % 256) as u8;
                assert_eq!(*ptr.as_ptr(), (i % 256) as u8);
            }
        }

        pool.deallocate(handle).expect(&format!("释放 {} 失败", i));

        if i % 100 == 0 {
            println!("完成 {} 次迭代", i);
        }
    }

    let duration = start.elapsed();
    let ops_per_second = iterations as f64 / duration.as_secs_f64();

    println!("性能基准测试完成:");
    println!("总时间: {:?}", duration);
    println!("每秒操作数: {:.2}", ops_per_second);
    println!("平均每次操作时间: {:?}", duration / iterations);
}
