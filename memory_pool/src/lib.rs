#![feature(btree_cursors)]
pub mod cpu_pool;
pub mod memory_pool;
pub mod buddy_disk_pool;
pub use cpu_pool::CpuMemoryPool;
pub use buddy_disk_pool::BuddyDiskPool;
pub mod static_allocator;