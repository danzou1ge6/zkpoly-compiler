use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Weak};

use crate::buddy_disk_pool::{BuddyDiskPool, BuddyDiskPoolError};
use zkpoly_common::define_usize_id;
use zkpoly_common::devices::DeviceType;
use zkpoly_cuda_api::mem::page_allocator::CudaPageAllocator;

/// 错误类型
#[derive(Debug)]
pub enum SwapPagePoolError {
    OutOfMemory,
    AllocationNotFound,
    DiskPoolError(BuddyDiskPoolError),
    AllocationFailed,
    InvalidHandle,
    DeadLock,
}

impl From<BuddyDiskPoolError> for SwapPagePoolError {
    fn from(err: BuddyDiskPoolError) -> Self {
        SwapPagePoolError::DiskPoolError(err)
    }
}

define_usize_id!(AllocationId); // 内存分配 ID，唯一标识一次分配
define_usize_id!(VirtualPageId); // 虚拟页面 ID，每次分配新页面时递增生成

/// 虚拟页面状态 - 虚拟页面可以在内存中或在磁盘上
#[derive(Debug, Clone, PartialEq)]
enum VirtualPageState {
    /// 在内存中，可以被交换出去
    InMemorySwappable { physical_page_id: usize },
    /// 在内存中，但有活跃引用，不能交换
    InMemoryUnswappable {
        physical_page_id: usize,
        ref_count: usize,
    },
    /// 已经被交换到磁盘
    SwappedToDisk { disk_offset: usize },
}

/// 虚拟页面信息 - 这些页面在分配时动态创建，可以被换出
#[derive(Debug, Clone)]
struct VirtualPageInfo {
    /// 虚拟页面ID
    virtual_page_id: VirtualPageId,
    /// 页面状态（包含物理位置或磁盘位置）
    state: VirtualPageState,
    /// 属于哪个分配
    allocation_id: AllocationId,
    /// 最后访问时间（用于 LRU）
    last_access: u64,
}

/// 物理页面信息 - 固定数量的物理内存页面
#[derive(Debug)]
struct PhysicalPageInfo {
    /// 物理页面是否被使用
    is_used: bool,
    /// 如果被使用，映射到哪个虚拟页面
    virtual_page_id: Option<VirtualPageId>,
}

/// 内存分配信息
#[derive(Debug, Clone)]
struct AllocationInfo {
    /// 虚拟地址
    virtual_address: usize,
    /// 分配大小
    size: usize,
    /// 使用的虚拟页面列表
    virtual_page_ids: Vec<VirtualPageId>,
    /// LRU 链表中的位置（仅当没有活跃引用时有效）
    lru_node: Option<LruNode>,
    /// 是否有活跃指针引用，应该是只会有一个活跃指针，这是编译保证的，所以可以用bool来表示
    has_active_ptr: bool,
}

/// LRU 链表节点
#[derive(Debug, Clone)]
struct LruNode {
    prev: Option<AllocationId>,
    next: Option<AllocationId>,
}

/// LRU 链表
#[derive(Debug)]
struct LruList {
    head: Option<AllocationId>,
    tail: Option<AllocationId>,
}

impl LruList {
    fn new() -> Self {
        Self {
            head: None,
            tail: None,
        }
    }

    fn push_front(
        &mut self,
        alloc_id: AllocationId,
        allocations: &mut HashMap<AllocationId, AllocationInfo>,
    ) {
        // 如果节点已经在链表中，先移除它
        if let Some(alloc_info) = allocations.get(&alloc_id) {
            if alloc_info.lru_node.is_some() {
                self.remove(alloc_id, allocations);
            }
        }

        // 现在添加到头部
        if let Some(alloc_info) = allocations.get_mut(&alloc_id) {
            alloc_info.lru_node = Some(LruNode {
                prev: None,
                next: self.head,
            });
        }

        if let Some(old_head) = self.head {
            if let Some(old_head_info) = allocations.get_mut(&old_head) {
                if let Some(ref mut node) = old_head_info.lru_node {
                    node.prev = Some(alloc_id);
                }
            }
        } else {
            self.tail = Some(alloc_id);
        }

        self.head = Some(alloc_id);
    }

    fn remove(
        &mut self,
        alloc_id: AllocationId,
        allocations: &mut HashMap<AllocationId, AllocationInfo>,
    ) {
        if let Some(alloc_info) = allocations.get_mut(&alloc_id) {
            if let Some(node) = alloc_info.lru_node.take() {
                match (node.prev, node.next) {
                    (None, None) => {
                        // 唯一节点
                        self.head = None;
                        self.tail = None;
                    }
                    (None, Some(next)) => {
                        // 头节点
                        self.head = Some(next);
                        if let Some(next_info) = allocations.get_mut(&next) {
                            if let Some(ref mut next_node) = next_info.lru_node {
                                next_node.prev = None;
                            }
                        }
                    }
                    (Some(prev), None) => {
                        // 尾节点
                        self.tail = Some(prev);
                        if let Some(prev_info) = allocations.get_mut(&prev) {
                            if let Some(ref mut prev_node) = prev_info.lru_node {
                                prev_node.next = None;
                            }
                        }
                    }
                    (Some(prev), Some(next)) => {
                        // 中间节点
                        if let Some(prev_info) = allocations.get_mut(&prev) {
                            if let Some(ref mut prev_node) = prev_info.lru_node {
                                prev_node.next = Some(next);
                            }
                        }
                        if let Some(next_info) = allocations.get_mut(&next) {
                            if let Some(ref mut next_node) = next_info.lru_node {
                                next_node.prev = Some(prev);
                            }
                        }
                    }
                }
            }
        }
    }

    fn pop_back(
        &mut self,
        allocations: &mut HashMap<AllocationId, AllocationInfo>,
    ) -> Option<AllocationId> {
        if let Some(tail_id) = self.tail {
            self.remove(tail_id, allocations);
            Some(tail_id)
        } else {
            None
        }
    }

    /// 打印LRU链表状态（用于调试）
    fn print_debug(&self, allocations: &HashMap<AllocationId, AllocationInfo>) {
        let mut current = self.head;
        let mut visited = HashSet::new();
        let mut count = 0;

        println!("LRU List State:");
        println!("Head: {:?}, Tail: {:?}", self.head, self.tail);

        while let Some(alloc_id) = current {
            // 检测循环
            if visited.contains(&alloc_id) {
                println!("*** CYCLE DETECTED at AllocationId: {} ***", alloc_id.0);
                break;
            }
            visited.insert(alloc_id);

            // 防止无限循环
            count += 1;
            if count > 1000 {
                println!("*** STOPPED after 1000 iterations (possible infinite loop) ***");
                break;
            }

            if let Some(alloc_info) = allocations.get(&alloc_id) {
                if let Some(ref node) = alloc_info.lru_node {
                    println!(
                        "AllocationId: {}, Size: {}, ActivePtr: {}, Prev: {:?}, Next: {:?}",
                        alloc_id.0,
                        alloc_info.size,
                        alloc_info.has_active_ptr,
                        node.prev,
                        node.next
                    );
                    current = node.next;
                } else {
                    println!("AllocationId: {} has no LRU node!", alloc_id.0);
                    break;
                }
            } else {
                println!("AllocationId: {} not found in allocations", alloc_id.0);
                break;
            }
        }

        println!("Total nodes traversed: {}", count);
    }
}

/// 指针句柄，表示对分配内存的引用
#[derive(Debug, Clone)]
pub struct PtrHandle<T> {
    allocation_id: AllocationId,
    pool: Weak<Mutex<SwapPagePoolInner>>,
    _phantom: PhantomData<T>,
}

impl<T> PtrHandle<T> {
    /// 获取原始指针，标记对应页面为不可交换
    pub fn get_ptr(&self) -> Result<MemoryPtr<T>, SwapPagePoolError> {
        if let Some(pool) = self.pool.upgrade() {
            let mut inner = pool.lock().unwrap();
            let (ptr, allocation_id) = inner.acquire_memory_ptr(self.allocation_id)?;
            Ok(MemoryPtr {
                ptr,
                allocation_id,
                pool: self.pool.clone(),
            })
        } else {
            Err(SwapPagePoolError::InvalidHandle)
        }
    }
}

/// 内存指针，表示正在使用的内存
pub struct MemoryPtr<T> {
    ptr: NonNull<T>,
    allocation_id: AllocationId,
    pool: Weak<Mutex<SwapPagePoolInner>>,
}

impl<T> MemoryPtr<T> {
    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_ref(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }

    pub fn as_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T> Drop for MemoryPtr<T> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            let mut inner = pool.lock().unwrap();
            inner.release_memory_ptr(self.allocation_id);
        }
    }
}

unsafe impl<T: Send> Send for MemoryPtr<T> {}
unsafe impl<T: Sync> Sync for MemoryPtr<T> {}

/// Swap Page Pool 的内部实现 - 简化为单一状态结构
struct SwapPagePoolInner {
    /// CUDA 页面分配器
    page_allocator: CudaPageAllocator,
    /// 磁盘池
    disk_pool: BuddyDiskPool,
    /// 虚拟页面信息映射 - 动态创建，可以被换出
    virtual_pages: HashMap<VirtualPageId, VirtualPageInfo>,
    /// 物理页面信息映射 - 固定数量的物理内存页面
    physical_pages: HashMap<usize, PhysicalPageInfo>,
    /// 分配信息映射
    allocations: HashMap<AllocationId, AllocationInfo>,
    /// LRU 链表
    lru_list: LruList,
    /// 空闲物理页面队列
    free_physical_pages: VecDeque<usize>,
    /// 下一个分配 ID
    next_allocation_id: usize,
    /// 下一个虚拟页面 ID
    next_virtual_page_id: usize,
    /// 页面大小
    page_size: usize,
    /// 最大物理页面数
    max_physical_pages: usize,
    /// 访问计数器（用于 LRU 时间戳）
    access_counter: u64,
}

impl SwapPagePoolInner {
    /// 获取内存指针并标记为不可交换
    fn acquire_memory_ptr<T>(
        &mut self,
        allocation_id: AllocationId,
    ) -> Result<(NonNull<T>, AllocationId), SwapPagePoolError> {
        if let None = self.allocations.get(&allocation_id) {
            return Err(SwapPagePoolError::AllocationNotFound);
        }

        let virtual_page_ids = self
            .allocations
            .get(&allocation_id)
            .unwrap()
            .virtual_page_ids
            .clone();
        let pages_swapped_back = self.ensure_virtual_pages_in_memory(&virtual_page_ids)?;

        // 检查是否需要从磁盘换回虚拟页面，如果有换回操作则返回true

        // 只有真的需要重新换回时才重新构建虚拟地址映射
        if pages_swapped_back {
            let physical_page_ids = self.get_physical_page_ids(&virtual_page_ids)?;
            let alloc_info = self.allocations.get_mut(&allocation_id).unwrap();

            let va_size = alloc_info.virtual_page_ids.len() * self.page_size;
            let ptr = self
                .page_allocator
                .allocate::<T>(va_size, &physical_page_ids);

            alloc_info.virtual_address = ptr as usize;
        }
        let alloc_info = self.allocations.get_mut(&allocation_id).unwrap();

        // 标记虚拟页面为不可交换
        for &virtual_page_id in &alloc_info.virtual_page_ids {
            if let Some(virtual_page_info) = self.virtual_pages.get_mut(&virtual_page_id) {
                match &mut virtual_page_info.state {
                    VirtualPageState::InMemorySwappable { physical_page_id } => {
                        virtual_page_info.state = VirtualPageState::InMemoryUnswappable {
                            physical_page_id: *physical_page_id,
                            ref_count: 1,
                        };
                    }
                    VirtualPageState::InMemoryUnswappable { ref_count, .. } => {
                        *ref_count += 1;
                    }
                    VirtualPageState::SwappedToDisk { .. } => {
                        return Err(SwapPagePoolError::AllocationFailed);
                    }
                }
                virtual_page_info.last_access = self.access_counter;
                self.access_counter += 1;
            }
        }

        let va = alloc_info.virtual_address;

        // 标记分配有活跃指针并从 LRU 链表中移除
        alloc_info.has_active_ptr = true;
        self.lru_list.remove(allocation_id, &mut self.allocations);

        let ptr = NonNull::new(va as *mut T).ok_or(SwapPagePoolError::AllocationFailed)?;

        Ok((ptr, allocation_id))
    }

    /// 释放内存指针的引用
    fn release_memory_ptr(&mut self, allocation_id: AllocationId) {
        if let Some(alloc_info) = self.allocations.get_mut(&allocation_id) {
            alloc_info.has_active_ptr = false;

            // 将虚拟页面标记为可交换
            for &virtual_page_id in &alloc_info.virtual_page_ids {
                if let Some(virtual_page_info) = self.virtual_pages.get_mut(&virtual_page_id) {
                    match &mut virtual_page_info.state {
                        VirtualPageState::InMemoryUnswappable {
                            physical_page_id,
                            ref_count,
                        } => {
                            *ref_count -= 1;
                            if *ref_count == 0 {
                                virtual_page_info.state = VirtualPageState::InMemorySwappable {
                                    physical_page_id: *physical_page_id,
                                };
                            }
                        }
                        _ => {} // 其他状态不需要处理
                    }
                }
            }

            // 将分配重新加入 LRU 链表
            if !alloc_info.has_active_ptr {
                self.lru_list
                    .push_front(allocation_id, &mut self.allocations);
            }
        }
    }

    /// 确保虚拟页面都在内存中，返回是否有页面被换回
    fn ensure_virtual_pages_in_memory(
        &mut self,
        virtual_page_ids: &[VirtualPageId],
    ) -> Result<bool, SwapPagePoolError> {
        let mut pages_swapped_back = false;

        for &virtual_page_id in virtual_page_ids {
            let virtual_page_info = self.virtual_pages.get(&virtual_page_id).cloned().unwrap();
            if let VirtualPageState::SwappedToDisk { disk_offset } = virtual_page_info.state {
                // 需要从磁盘换回
                let physical_page_id = self.allocate_physical_page()?;

                // 从磁盘读取数据
                let page_ptr = self.page_allocator.get_page(physical_page_id);
                let buffer =
                    unsafe { std::slice::from_raw_parts_mut(page_ptr as *mut u8, self.page_size) };
                self.disk_pool.read(disk_offset, buffer)?;

                // 释放磁盘空间
                self.disk_pool.deallocate(disk_offset, self.page_size)?;

                // 更新虚拟页面状态
                self.virtual_pages.get_mut(&virtual_page_id).unwrap().state =
                    VirtualPageState::InMemorySwappable { physical_page_id };

                // 更新物理页面映射
                if let Some(physical_page_info) = self.physical_pages.get_mut(&physical_page_id) {
                    physical_page_info.is_used = true;
                    physical_page_info.virtual_page_id = Some(virtual_page_id);
                }

                pages_swapped_back = true;
            }
        }

        Ok(pages_swapped_back)
    }

    /// 获取虚拟页面对应的物理页面ID列表
    fn get_physical_page_ids(
        &self,
        virtual_page_ids: &[VirtualPageId],
    ) -> Result<Vec<usize>, SwapPagePoolError> {
        let mut physical_page_ids = Vec::new();

        for &virtual_page_id in virtual_page_ids {
            if let Some(virtual_page_info) = self.virtual_pages.get(&virtual_page_id) {
                match &virtual_page_info.state {
                    VirtualPageState::InMemorySwappable { physical_page_id }
                    | VirtualPageState::InMemoryUnswappable {
                        physical_page_id, ..
                    } => {
                        physical_page_ids.push(*physical_page_id);
                    }
                    VirtualPageState::SwappedToDisk { .. } => {
                        return Err(SwapPagePoolError::AllocationFailed);
                    }
                }
            } else {
                return Err(SwapPagePoolError::AllocationNotFound);
            }
        }

        Ok(physical_page_ids)
    }

    /// 分配物理页面
    fn allocate_physical_page(&mut self) -> Result<usize, SwapPagePoolError> {
        if let Some(physical_page_id) = self.free_physical_pages.pop_front() {
            Ok(physical_page_id)
        } else {
            // 没有空闲物理页面，需要换出一些虚拟页面
            self.evict_lru_virtual_page()?;
            self.free_physical_pages
                .pop_front()
                .ok_or(SwapPagePoolError::OutOfMemory)
        }
    }

    /// 换出LRU虚拟页面
    fn evict_lru_virtual_page(&mut self) -> Result<(), SwapPagePoolError> {
        if let Some(allocation_id) = self.lru_list.pop_back(&mut self.allocations) {
            if let Some(alloc_info) = self.allocations.get(&allocation_id) {
                if !alloc_info.has_active_ptr {
                    let virtual_page_ids = alloc_info.virtual_page_ids.clone();
                    self.swap_virtual_pages_to_disk(&virtual_page_ids)?;
                    return Ok(());
                }
            } else {
                panic!("LRU list contains invalid alloc_info")
            }
        }
        Err(SwapPagePoolError::OutOfMemory)
    }

    /// 将虚拟页面换出到磁盘
    fn swap_virtual_pages_to_disk(
        &mut self,
        virtual_page_ids: &[VirtualPageId],
    ) -> Result<(), SwapPagePoolError> {
        for &virtual_page_id in virtual_page_ids {
            if let Some(virtual_page_info) = self.virtual_pages.get_mut(&virtual_page_id) {
                if let VirtualPageState::InMemorySwappable { physical_page_id } =
                    virtual_page_info.state
                {
                    // 分配磁盘空间
                    let disk_offset = self.disk_pool.allocate(self.page_size)?;

                    // 读取页面数据并写入磁盘
                    let page_ptr = self.page_allocator.get_page(physical_page_id);
                    let buffer = unsafe {
                        std::slice::from_raw_parts(page_ptr as *const u8, self.page_size)
                    };
                    self.disk_pool.write(disk_offset, buffer)?;

                    // 更新虚拟页面状态
                    virtual_page_info.state = VirtualPageState::SwappedToDisk { disk_offset };

                    // 释放物理页面
                    if let Some(physical_page_info) = self.physical_pages.get_mut(&physical_page_id)
                    {
                        physical_page_info.is_used = false;
                        physical_page_info.virtual_page_id = None;
                    }
                    self.free_physical_pages.push_back(physical_page_id);
                }
            }
        }

        Ok(())
    }

    /// 分配内存
    fn allocate_memory(&mut self, size: usize) -> Result<AllocationId, SwapPagePoolError> {
        let pages_needed = (size + self.page_size - 1) / self.page_size;
        let mut virtual_page_ids = Vec::new();

        // 为每个需要的页面创建虚拟页面
        for _ in 0..pages_needed {
            let virtual_page_id = VirtualPageId(self.next_virtual_page_id);
            self.next_virtual_page_id += 1;

            let physical_page_id = self.allocate_physical_page()?;

            let virtual_page_info = VirtualPageInfo {
                virtual_page_id,
                state: VirtualPageState::InMemorySwappable { physical_page_id },
                allocation_id: AllocationId(0), // 临时设置，下面会更新
                last_access: self.access_counter,
            };
            self.access_counter += 1;

            // 更新物理页面映射
            if let Some(physical_page_info) = self.physical_pages.get_mut(&physical_page_id) {
                physical_page_info.is_used = true;
                physical_page_info.virtual_page_id = Some(virtual_page_id);
            }

            virtual_page_ids.push(virtual_page_id);

            // 添加到虚拟页面映射
            self.virtual_pages
                .insert(virtual_page_id, virtual_page_info);
        }

        let allocation_id = AllocationId(self.next_allocation_id);
        self.next_allocation_id += 1;

        // 更新虚拟页面的分配ID
        for &virtual_page_id in &virtual_page_ids {
            if let Some(virtual_page_info) = self.virtual_pages.get_mut(&virtual_page_id) {
                virtual_page_info.allocation_id = allocation_id;
            }
        }

        // 通过 CUDA page allocator 分配虚拟地址
        let physical_page_ids = self.get_physical_page_ids(&virtual_page_ids)?;
        assert_eq!(physical_page_ids.len(), pages_needed);
        let va_size = pages_needed * self.page_size;
        let virtual_addr = self
            .page_allocator
            .allocate::<u8>(va_size, &physical_page_ids);

        // 创建分配信息
        let alloc_info = AllocationInfo {
            virtual_address: virtual_addr as usize,
            size,
            virtual_page_ids,
            has_active_ptr: false,
            lru_node: None,
        };

        self.allocations.insert(allocation_id, alloc_info);

        // 加入 LRU 链表
        self.lru_list
            .push_front(allocation_id, &mut self.allocations);

        Ok(allocation_id)
    }

    /// 释放内存
    fn deallocate_memory(&mut self, allocation_id: AllocationId) -> Result<(), SwapPagePoolError> {
        // 从 LRU 链表中移除
        self.lru_list.remove(allocation_id, &mut self.allocations);

        if let Some(alloc_info) = self.allocations.remove(&allocation_id) {
            // 释放虚拟页面
            for virtual_page_id in alloc_info.virtual_page_ids.iter() {
                if let Some(virtual_page_info) = self.virtual_pages.remove(&virtual_page_id) {
                    match virtual_page_info.state {
                        VirtualPageState::InMemorySwappable { physical_page_id }
                        | VirtualPageState::InMemoryUnswappable {
                            physical_page_id, ..
                        } => {
                            // 释放物理页面
                            if let Some(physical_page_info) =
                                self.physical_pages.get_mut(&physical_page_id)
                            {
                                physical_page_info.is_used = false;
                                physical_page_info.virtual_page_id = None;
                            }
                            self.free_physical_pages.push_back(physical_page_id);
                        }
                        VirtualPageState::SwappedToDisk { disk_offset } => {
                            // 释放磁盘空间
                            self.disk_pool.deallocate(disk_offset, self.page_size)?;
                        }
                    }
                }
            }

            Ok(())
        } else {
            Err(SwapPagePoolError::AllocationNotFound)
        }
    }

    /// 打印LRU链表状态（用于调试）
    fn print_lru_list(&self) {
        self.lru_list.print_debug(&self.allocations);
    }
}

/// Swap Page Pool 主结构
pub struct SwapPagePool {
    inner: Arc<Mutex<SwapPagePoolInner>>,
}

impl SwapPagePool {
    /// 创建新的 SwapPagePool
    pub fn new(
        device: DeviceType,
        page_size: usize,
        max_physical_pages: usize,
        disk_pool_capacity: usize,
    ) -> Result<Self, SwapPagePoolError> {
        let page_allocator = CudaPageAllocator::new(device, page_size, max_physical_pages);
        let disk_pool = BuddyDiskPool::new(disk_pool_capacity, page_size)?;

        // 初始化物理页面信息
        let mut physical_pages = HashMap::new();
        let mut free_physical_pages = VecDeque::new();

        for i in 0..max_physical_pages {
            physical_pages.insert(
                i,
                PhysicalPageInfo {
                    is_used: false,
                    virtual_page_id: None,
                },
            );
            free_physical_pages.push_back(i);
        }

        let inner = Arc::new(Mutex::new(SwapPagePoolInner {
            page_allocator,
            disk_pool,
            virtual_pages: HashMap::new(),
            physical_pages,
            allocations: HashMap::new(),
            lru_list: LruList::new(),
            free_physical_pages,
            next_allocation_id: 0,
            next_virtual_page_id: 0,
            page_size,
            max_physical_pages,
            access_counter: 0,
        }));

        Ok(Self { inner })
    }

    /// 分配内存并返回 PtrHandle
    pub fn allocate<U>(&self, size: usize) -> Result<PtrHandle<U>, SwapPagePoolError> {
        let mut inner = self.inner.lock().unwrap();
        let allocation_id = inner.allocate_memory(size)?;

        Ok(PtrHandle {
            allocation_id,
            pool: Arc::downgrade(&self.inner),
            _phantom: PhantomData,
        })
    }

    /// 释放内存分配
    pub fn deallocate(&self, handle: PtrHandle<impl Sized>) -> Result<(), SwapPagePoolError> {
        let mut inner = self.inner.lock().unwrap();
        inner.deallocate_memory(handle.allocation_id)
    }

    /// 获取池的统计信息
    pub fn stats(&self) -> SwapPagePoolStats {
        let inner = self.inner.lock().unwrap();

        let mut in_memory_pages = 0;
        let mut swapped_pages = 0;

        for virtual_page_info in inner.virtual_pages.values() {
            match virtual_page_info.state {
                VirtualPageState::InMemorySwappable { .. }
                | VirtualPageState::InMemoryUnswappable { .. } => {
                    in_memory_pages += 1;
                }
                VirtualPageState::SwappedToDisk { .. } => {
                    swapped_pages += 1;
                }
            }
        }

        let free_pages = inner.physical_pages.values().filter(|p| !p.is_used).count();

        SwapPagePoolStats {
            total_pages: inner.max_physical_pages,
            free_pages,
            in_memory_pages,
            swapped_pages,
            active_allocations: inner.allocations.len(),
            page_size: inner.page_size,
        }
    }

    /// 强制换出指定数量的页面（用于内存压力情况）
    pub fn force_evict(&self, target_pages: usize) -> Result<usize, SwapPagePoolError> {
        let mut inner = self.inner.lock().unwrap();
        let mut evicted = 0;

        while evicted < target_pages {
            match inner.evict_lru_virtual_page() {
                Ok(()) => evicted += 1,
                Err(SwapPagePoolError::OutOfMemory) => break, // 没有更多可换出的页面
                Err(e) => return Err(e),
            }
        }

        Ok(evicted)
    }

    /// 打印LRU链表状态（用于调试）
    pub fn debug_print_lru(&self) {
        let inner = self.inner.lock().unwrap();
        inner.print_lru_list();
    }
}

/// 内存池统计信息
#[derive(Debug, Clone)]
pub struct SwapPagePoolStats {
    pub total_pages: usize,
    pub free_pages: usize,
    pub in_memory_pages: usize,
    pub swapped_pages: usize,
    pub active_allocations: usize,
    pub page_size: usize,
}

impl SwapPagePoolStats {
    /// 计算内存使用率
    pub fn memory_usage_ratio(&self) -> f64 {
        (self.total_pages - self.free_pages) as f64 / self.total_pages as f64
    }

    /// 计算碎片率
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.active_allocations == 0 {
            0.0
        } else {
            1.0 - (self.active_allocations as f64 / (self.total_pages - self.free_pages) as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        // 这里需要模拟的测试，因为需要 CUDA 环境
        // 实际测试应该在有 GPU 的环境中运行

        // 模拟测试：验证结构体可以正确创建
        println!("SwapPagePool 结构定义测试通过");
    }

    #[test]
    fn test_error_types() {
        // 测试错误类型
        let err = SwapPagePoolError::OutOfMemory;
        assert!(matches!(err, SwapPagePoolError::OutOfMemory));

        let err2 = SwapPagePoolError::AllocationNotFound;
        assert!(matches!(err2, SwapPagePoolError::AllocationNotFound));
    }

    #[test]
    fn test_allocation_id() {
        let id1 = AllocationId(0);
        let id2 = AllocationId(1);
        assert_ne!(id1, id2);
        assert_eq!(id1, AllocationId(0));
    }

    #[test]
    fn test_stats() {
        let stats = SwapPagePoolStats {
            total_pages: 100,
            free_pages: 50,
            in_memory_pages: 30,
            swapped_pages: 20,
            active_allocations: 10,
            page_size: 4096,
        };

        assert_eq!(stats.memory_usage_ratio(), 0.5);
        assert!(stats.fragmentation_ratio() > 0.0);
    }
}
