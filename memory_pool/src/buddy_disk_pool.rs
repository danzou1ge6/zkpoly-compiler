#![allow(dead_code, unused_imports, unused_variables)] // TODO: Remove as implementation progresses

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::Mutex; // Using Mutex for interior mutability of free_lists etc.

use libc;
use nix::fcntl::{self, OFlag};
use nix::sys::{statfs, uio};
use nix::unistd;

#[derive(Debug)]
pub enum BuddyDiskPoolError {
    IoError(io::Error),
    NixError(nix::Error),
    OutOfMemory,
    AlignmentError(String),
    InvalidOffset(String),
    InvalidSize(String),
    FileOpenError(String),
    FallocateError(String),
    InvalidCapacity(String),
    InvalidMinBlockSize(String),
    BlockNotFound(String),
    InternalError(String),
}

impl From<io::Error> for BuddyDiskPoolError {
    fn from(err: io::Error) -> BuddyDiskPoolError {
        BuddyDiskPoolError::IoError(err)
    }
}

impl From<nix::Error> for BuddyDiskPoolError {
    fn from(err: nix::Error) -> BuddyDiskPoolError {
        BuddyDiskPoolError::NixError(err)
    }
}

/// Metadata for a disk-based slab.
#[derive(Debug, Clone)]
struct DiskSlabInfo {
    offset: usize, // Offset within the disk file
    log_factor: u32,
    free: bool,
    parent: Option<usize>, // Index in BuddyDiskPool::slabs
    lchild: Option<usize>,
    rchild: Option<usize>,
    next_in_layer: Option<usize>,
    prev_in_layer: Option<usize>,
    next_free_in_layer: Option<usize>,
    prev_free_in_layer: Option<usize>,
}

impl DiskSlabInfo {
    fn new(offset: usize, log_factor: u32, parent: Option<usize>) -> Self {
        DiskSlabInfo {
            offset,
            log_factor,
            free: true,
            parent,
            lchild: None,
            rchild: None,
            next_in_layer: None,
            prev_in_layer: None,
            next_free_in_layer: None,
            prev_free_in_layer: None,
        }
    }
}

/// Information for each layer of slabs, primarily head pointers for linked lists.
#[derive(Debug, Default, Clone, Copy)]
struct SlabLayerInfo {
    head_all_idx: Option<usize>,  // Index in BuddyDiskPool::slabs
    head_free_idx: Option<usize>, // Index in BuddyDiskPool::slabs
}

pub struct BuddyDiskPool {
    file: File,
    file_path: PathBuf, // To aid in debugging or potential re-opening
    _temp_dir_handle: tempfile::TempDir,

    slabs: Vec<DiskSlabInfo>,
    free_slab_info_indices: Vec<usize>,
    slab_layers: Vec<SlabLayerInfo>,
    
    // Maps an allocated offset to its DiskSlabInfo index in slabs.
    active_allocations: HashMap<usize, usize>, 

    capacity: usize,         // Total usable capacity, aligned to system_alignment
    system_alignment: usize, // Alignment required by O_DIRECT and filesystem
    min_block_size: usize,   // Smallest allocatable block size, aligned to system_alignment
    max_log_factor: u32,     // Max log_factor relative to min_block_size
                             // num_levels = max_log_factor + 1
}

// Wrap mutable parts in a Mutex if BuddyDiskPool needs to be Sync
// For now, assuming &mut self for main operations. If methods take &self, then Mutex is needed.
// Let's make the core mutable structures (slabs, free_slab_info_indices, slab_layers, active_allocations)
// part of an inner struct guarded by a Mutex if we want &self methods for allocation/deallocation.
// Or, keep them as direct fields and use &mut self. The original MemoryPool uses &mut self.

impl BuddyDiskPool {
    pub fn new(total_capacity_request: usize, min_block_size_request: usize) -> Result<Self, BuddyDiskPoolError> {
        if total_capacity_request == 0 {
            return Err(BuddyDiskPoolError::InvalidCapacity("Total capacity request cannot be zero.".to_string()));
        }
        if min_block_size_request == 0 {
            return Err(BuddyDiskPoolError::InvalidMinBlockSize("Min block size request cannot be zero.".to_string()));
        }

        let temp_dir = tempfile::Builder::new().prefix("buddy_disk_pool_").tempdir()?;
        let file_path = temp_dir.path().join("buddy_pool.dat");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(&file_path)
            .map_err(|e| BuddyDiskPoolError::FileOpenError(format!("Failed to open {:?}: {}", file_path, e)))?;

        let system_alignment = match statfs::fstatfs(&file) {
            Ok(fs_stat) => fs_stat.block_size() as usize,
            Err(e) => return Err(BuddyDiskPoolError::NixError(e)),
        };

        if system_alignment == 0 || !system_alignment.is_power_of_two() {
            return Err(BuddyDiskPoolError::AlignmentError(format!("Invalid system alignment {}", system_alignment)));
        }

        let min_block_size = (min_block_size_request + system_alignment - 1) / system_alignment * system_alignment;
        if min_block_size == 0 {
             return Err(BuddyDiskPoolError::InvalidMinBlockSize(format!("Requested min_block_size {} is too small to be aligned to {}", min_block_size_request, system_alignment)));
        }
        if min_block_size > total_capacity_request && total_capacity_request > 0 { // allow total_capacity_request = 0 for now, though earlier check catches it
             return Err(BuddyDiskPoolError::InvalidMinBlockSize(format!("Min block size {} cannot be greater than total capacity request {}", min_block_size, total_capacity_request)));
        }
        
        // Capacity must be a multiple of min_block_size for the buddy system to perfectly cover it.
        // And also a multiple of system_alignment for fallocate.
        let num_min_blocks_in_capacity = total_capacity_request / min_block_size;
        let capacity = num_min_blocks_in_capacity * min_block_size;
        // Ensure this capacity is also aligned for fallocate, though min_block_size is already system_aligned.
        // So, capacity will be system_aligned.

        if capacity == 0 && total_capacity_request > 0 {
             return Err(BuddyDiskPoolError::InvalidCapacity(format!("Adjusted capacity is zero. Original request: {}, MinBlock: {}. Num min blocks: {}", total_capacity_request, min_block_size, num_min_blocks_in_capacity)));
        }

        if capacity > 0 {
            fcntl::fallocate(&file, fcntl::FallocateFlags::empty(), 0, capacity as libc::off_t)
                .map_err(|e| BuddyDiskPoolError::FallocateError(format!("fallocate failed for size {}: {}", capacity, e)))?;
        }
        
        let max_log_factor = if capacity >= min_block_size {
            (capacity / min_block_size).checked_ilog2().unwrap_or(0)
        } else {
            0 // Should be caught by capacity == 0 check if min_block_size > 0
        };
        
        let num_layers = (max_log_factor + 1) as usize;

        let mut pool = Self {
            file,
            file_path,
            _temp_dir_handle: temp_dir,
            slabs: Vec::new(),
            free_slab_info_indices: Vec::new(),
            slab_layers: vec![SlabLayerInfo::default(); num_layers],
            active_allocations: HashMap::new(),
            capacity,
            system_alignment,
            min_block_size,
            max_log_factor,
        };

        // Initialize free list with top-level blocks covering the capacity
        if capacity > 0 {
            let mut remaining_cap = capacity;
            let mut current_offset = 0;
            for lf in (0..=max_log_factor).rev() {
                let block_size_at_lf = pool.min_block_size * (1 << lf);
                if block_size_at_lf == 0 { continue; } // Should not happen if min_block_size > 0

                while remaining_cap >= block_size_at_lf {
                    let slab_idx = pool.obtain_slab_info_index(current_offset, lf, None);
                    // pool.slabs[slab_idx].free is true by default from new()
                    pool.layer_insert_all(lf, slab_idx);
                    pool.layer_insert_free(lf, slab_idx);
                    
                    current_offset += block_size_at_lf;
                    remaining_cap -= block_size_at_lf;
                }
            }
        }
        Ok(pool)
    }

    fn obtain_slab_info_index(&mut self, offset: usize, log_factor: u32, parent_idx: Option<usize>) -> usize {
        if let Some(idx) = self.free_slab_info_indices.pop() {
            self.slabs[idx] = DiskSlabInfo::new(offset, log_factor, parent_idx);
            idx
        } else {
            self.slabs.push(DiskSlabInfo::new(offset, log_factor, parent_idx));
            self.slabs.len() - 1
        }
    }

    fn release_slab_info_index(&mut self, slab_idx: usize) {
        // Could clear self.slabs[slab_idx] for safety, though obtain_slab_info_index overwrites.
        self.free_slab_info_indices.push(slab_idx);
    }

    // --- Linked list helpers (adapted from MemoryPool) ---
    fn layer_insert_all(&mut self, log_factor: u32, slab_idx: usize) {
        let layer_info = &mut self.slab_layers[log_factor as usize];
        match layer_info.head_all_idx {
            Some(head_idx) => {
                let head_prev_idx = self.slabs[head_idx].prev_in_layer.expect("Head of 'all' list must have a prev pointer");
                self.slabs[slab_idx].next_in_layer = Some(head_idx);
                self.slabs[slab_idx].prev_in_layer = Some(head_prev_idx);
                self.slabs[head_idx].prev_in_layer = Some(slab_idx);
                self.slabs[head_prev_idx].next_in_layer = Some(slab_idx);
            }
            None => {
                self.slabs[slab_idx].next_in_layer = Some(slab_idx);
                self.slabs[slab_idx].prev_in_layer = Some(slab_idx);
                layer_info.head_all_idx = Some(slab_idx);
            }
        }
    }

    fn layer_remove_all(&mut self, log_factor: u32, slab_idx: usize) {
        let (slab_next_opt, slab_prev_opt) = {
            let slab_to_remove = &self.slabs[slab_idx];
            (slab_to_remove.next_in_layer, slab_to_remove.prev_in_layer)
        };
        if slab_next_opt == Some(slab_idx) { // Only element
            self.slab_layers[log_factor as usize].head_all_idx = None;
        } else {
            let next_s_idx = slab_next_opt.expect("Slab in list must have next pointer");
            let prev_s_idx = slab_prev_opt.expect("Slab in list must have prev pointer");
            self.slabs[next_s_idx].prev_in_layer = Some(prev_s_idx);
            self.slabs[prev_s_idx].next_in_layer = Some(next_s_idx);
            if self.slab_layers[log_factor as usize].head_all_idx == Some(slab_idx) {
                self.slab_layers[log_factor as usize].head_all_idx = Some(next_s_idx);
            }
        }
        self.slabs[slab_idx].next_in_layer = None;
        self.slabs[slab_idx].prev_in_layer = None;
    }

    fn layer_insert_free(&mut self, log_factor: u32, slab_idx: usize) {
        // Ensure slab is marked free before adding
        if !self.slabs[slab_idx].free {
            // This would be an internal error
            // For now, let's assume caller ensures this or we mark it free here.
            // self.slabs[slab_idx].free = true;
        }
        let layer_info = &mut self.slab_layers[log_factor as usize];
        match layer_info.head_free_idx {
            Some(head_idx) => {
                let head_prev_idx = self.slabs[head_idx].prev_free_in_layer.expect("Head of 'free' list must have a prev pointer");
                self.slabs[slab_idx].next_free_in_layer = Some(head_idx);
                self.slabs[slab_idx].prev_free_in_layer = Some(head_prev_idx);
                self.slabs[head_idx].prev_free_in_layer = Some(slab_idx);
                self.slabs[head_prev_idx].next_free_in_layer = Some(slab_idx);
            }
            None => {
                self.slabs[slab_idx].next_free_in_layer = Some(slab_idx);
                self.slabs[slab_idx].prev_free_in_layer = Some(slab_idx);
                layer_info.head_free_idx = Some(slab_idx);
            }
        }
    }

    fn layer_remove_free(&mut self, log_factor: u32, slab_idx: usize) {
        let (slab_next_opt, slab_prev_opt) = {
            let slab_to_remove = &self.slabs[slab_idx];
            // assert!(slab_to_remove.free, "Slab being removed from free list should be free");
            (slab_to_remove.next_free_in_layer, slab_to_remove.prev_free_in_layer)
        };
         if slab_next_opt.is_none() && slab_prev_opt.is_none() && self.slab_layers[log_factor as usize].head_free_idx != Some(slab_idx) {
            // Not in the free list, or list is corrupted. This can happen if already removed.
            // This function should be idempotent or ensure it's only called once.
            return; 
        }

        if slab_next_opt == Some(slab_idx) { // Only element
            self.slab_layers[log_factor as usize].head_free_idx = None;
        } else {
            // These expects will panic if the slab was not actually in a list.
            let next_s_idx = slab_next_opt.ok_or_else(|| BuddyDiskPoolError::InternalError(format!("Slab {} (lf {}) not in free list (no next_free_in_layer)", slab_idx, log_factor))).unwrap(); // TODO: Return Result
            let prev_s_idx = slab_prev_opt.ok_or_else(|| BuddyDiskPoolError::InternalError(format!("Slab {} (lf {}) not in free list (no prev_free_in_layer)", slab_idx, log_factor))).unwrap(); // TODO: Return Result

            self.slabs[next_s_idx].prev_free_in_layer = Some(prev_s_idx);
            self.slabs[prev_s_idx].next_free_in_layer = Some(next_s_idx);
            if self.slab_layers[log_factor as usize].head_free_idx == Some(slab_idx) {
                self.slab_layers[log_factor as usize].head_free_idx = Some(next_s_idx);
            }
        }
        self.slabs[slab_idx].next_free_in_layer = None;
        self.slabs[slab_idx].prev_free_in_layer = None;
    }
    
    fn get_first_free_slab_in_layer(&self, log_factor: u32) -> Option<usize> {
        if (log_factor as usize) < self.slab_layers.len() {
            self.slab_layers[log_factor as usize].head_free_idx
        } else {
            None
        }
    }

    /// Splits a parent slab to create children at `child_log_factor`.
    fn split_slab(&mut self, parent_log_factor: u32) -> Result<(usize, usize), BuddyDiskPoolError> {
        if parent_log_factor == 0 {
            return Err(BuddyDiskPoolError::InternalError("Cannot split slab of log_factor 0".to_string()));
        }
        if parent_log_factor > self.max_log_factor {
             return Err(BuddyDiskPoolError::InternalError(format!("Cannot split from log_factor {} > max_log_factor {}", parent_log_factor, self.max_log_factor)));
        }

        let parent_slab_idx = self.get_first_free_slab_in_layer(parent_log_factor)
            .ok_or_else(|| BuddyDiskPoolError::InternalError(format!("No free slab to split at log_factor {}", parent_log_factor)))?;

        self.slabs[parent_slab_idx].free = false;
        self.layer_remove_free(parent_log_factor, parent_slab_idx);

        let child_log_factor = parent_log_factor - 1;
        let parent_offset = self.slabs[parent_slab_idx].offset;
        let child_actual_size = self.min_block_size * (1 << child_log_factor);

        let lchild_offset = parent_offset;
        let rchild_offset = parent_offset + child_actual_size;

        let lchild_idx = self.obtain_slab_info_index(lchild_offset, child_log_factor, Some(parent_slab_idx));
        let rchild_idx = self.obtain_slab_info_index(rchild_offset, child_log_factor, Some(parent_slab_idx));
        
        self.slabs[parent_slab_idx].lchild = Some(lchild_idx);
        self.slabs[parent_slab_idx].rchild = Some(rchild_idx);

        // Children are initially free
        self.slabs[lchild_idx].free = true;
        self.slabs[rchild_idx].free = true;

        self.layer_insert_all(child_log_factor, lchild_idx);
        self.layer_insert_all(child_log_factor, rchild_idx);
        self.layer_insert_free(child_log_factor, lchild_idx);
        self.layer_insert_free(child_log_factor, rchild_idx);
        
        Ok((lchild_idx, rchild_idx))
    }

    fn ensure_free_slab_exists(&mut self, log_factor: u32) -> Result<usize, BuddyDiskPoolError> {
        if self.get_first_free_slab_in_layer(log_factor).is_some() {
            return Ok(self.get_first_free_slab_in_layer(log_factor).unwrap());
        }

        if log_factor == self.max_log_factor {
            // This case should ideally be handled by initial pool setup if capacity allows.
            // If we reach here, it means the largest blocks were already allocated and split,
            // or initial capacity didn't perfectly align to create a block of this max_log_factor.
            // For a disk pool with pre-fallocated space, we don't "allocate new physical memory"
            // in the same way. The "physical memory" is the file.
            // If no free slab at max_log_factor, it means all space is used or fragmented into smaller blocks.
            return Err(BuddyDiskPoolError::OutOfMemory); // No more blocks of this size can be formed/found
        }

        // Recursively ensure a parent exists
        self.ensure_free_slab_exists(log_factor + 1)?;
        // Now, a free slab at log_factor + 1 must exist. Split it.
        let (lchild_idx, _rchild_idx) = self.split_slab(log_factor + 1)?;
        
        // The split ensures lchild_idx is at `log_factor` and is free.
        // `split_slab` adds children to free list.
        Ok(lchild_idx) // Or rchild_idx, or whichever one is desired. Usually lchild.
    }
    
    /// Calculates the required log_factor for a given size.
    fn get_target_log_factor(&self, size_bytes: usize) -> Result<u32, BuddyDiskPoolError> {
        if size_bytes == 0 {
            return Err(BuddyDiskPoolError::InvalidSize("Allocation size cannot be zero.".to_string()));
        }
        if size_bytes > self.capacity { // Check against total pool capacity
             return Err(BuddyDiskPoolError::OutOfMemory);
        }
        // Smallest block that can contain size_bytes
        let num_min_chunks = (size_bytes + self.min_block_size - 1) / self.min_block_size;
        let log_factor = num_min_chunks.next_power_of_two().trailing_zeros();

        if log_factor > self.max_log_factor {
            Err(BuddyDiskPoolError::InvalidSize(format!(
                "Requested size {} (needs log_factor {}) exceeds max pool block size (max_log_factor {}). Min block size: {}",
                size_bytes, log_factor, self.max_log_factor, self.min_block_size
            )))
        } else {
            Ok(log_factor)
        }
    }

    pub fn allocate(&mut self, size_bytes: usize) -> Result<usize, BuddyDiskPoolError> {
        if size_bytes == 0 {
            // Or return a special offset like 0 with size 0, if that's meaningful.
            // For now, error on zero size.
            return Err(BuddyDiskPoolError::InvalidSize("Cannot allocate zero bytes.".to_string()));
        }
        // User does not specify alignment, pool uses its system_alignment internally for all O_DIRECT ops.
        // The size_bytes itself doesn't need to be pre-aligned by user, but the allocated block will be.
        // The block chosen will have a size of min_block_size * 2^log_factor, which is inherently aligned
        // if min_block_size is system_aligned.

        let log_factor = self.get_target_log_factor(size_bytes)?;
        let slab_idx = self.ensure_free_slab_exists(log_factor)?;

        self.slabs[slab_idx].free = false;
        self.layer_remove_free(log_factor, slab_idx);

        let offset = self.slabs[slab_idx].offset;
        self.active_allocations.insert(offset, slab_idx);
        Ok(offset)
    }

    fn try_merge(&mut self, slab_idx: usize) -> Result<(), BuddyDiskPoolError> {
        let parent_idx_opt = self.slabs[slab_idx].parent;
        let current_log_factor = self.slabs[slab_idx].log_factor;

        if current_log_factor == self.max_log_factor {
            return Ok(()); // Cannot merge beyond max_log_factor
        }
        
        if let Some(parent_idx) = parent_idx_opt {
            let (lchild_idx_opt, rchild_idx_opt) = {
                let parent_slab = &self.slabs[parent_idx];
                (parent_slab.lchild, parent_slab.rchild)
            };

            if let (Some(lidx), Some(ridx)) = (lchild_idx_opt, rchild_idx_opt) {
                if self.slabs[lidx].free && self.slabs[ridx].free {
                    // Both children are free, merge them into parent.
                    let child_lf = self.slabs[lidx].log_factor; // Should be same for rchild

                    self.layer_remove_free(child_lf, lidx);
                    self.layer_remove_all(child_lf, lidx);
                    self.release_slab_info_index(lidx);

                    self.layer_remove_free(child_lf, ridx);
                    self.layer_remove_all(child_lf, ridx);
                    self.release_slab_info_index(ridx);
                    
                    let parent_mut_slab = &mut self.slabs[parent_idx];
                    parent_mut_slab.lchild = None;
                    parent_mut_slab.rchild = None;
                    parent_mut_slab.free = true;
                    let parent_log_factor = parent_mut_slab.log_factor;
                    self.layer_insert_free(parent_log_factor, parent_idx);

                    // Recursively try to merge the parent
                    return self.try_merge(parent_idx);
                }
            } else {
                return Err(BuddyDiskPoolError::InternalError(format!("Parent slab {} is missing child information during merge attempt for slab {}.", parent_idx, slab_idx)));
            }
        }
        Ok(())
    }

    pub fn deallocate(&mut self, offset: usize, size_bytes: usize) -> Result<(), BuddyDiskPoolError> {
        if size_bytes == 0 {
             return Err(BuddyDiskPoolError::InvalidSize("Cannot deallocate zero bytes.".to_string()));
        }
        let slab_idx = self.active_allocations.remove(&offset).ok_or_else(|| {
            BuddyDiskPoolError::BlockNotFound(format!("Offset {} not found in active allocations or already deallocated.", offset))
        })?;

        // Verify size matches allocated block's effective size (optional, but good for safety)
        let allocated_log_factor = self.slabs[slab_idx].log_factor;
        let allocated_block_size = self.min_block_size * (1 << allocated_log_factor);
        // The provided size_bytes is the user's original request. The actual block might be larger.
        // We deallocate the entire buddy block.
        // No, the user should provide the size they thought they allocated, or we derive it from log_factor.
        // Let's assume the `size_bytes` is for validation or finding the correct log_factor if not using active_allocations map.
        // Since we have slab_idx from active_allocations, we know its log_factor.

        self.slabs[slab_idx].free = true;
        self.layer_insert_free(allocated_log_factor, slab_idx);
        
        self.try_merge(slab_idx)?; // Start merge attempt from the deallocated slab itself (it will look at its parent)

        Ok(())
    }

    /// Reads data from the disk pool into the provided buffer.
    /// Offset must be from a previous `allocate` call.
    /// Buffer length must be <= allocated size for that offset, and meet O_DIRECT alignment.
    /// Buffer address must also meet O_DIRECT alignment.
    pub fn read(&self, offset: usize, buffer: &mut [u8]) -> Result<(), BuddyDiskPoolError> {
        if offset % self.system_alignment != 0 {
            return Err(BuddyDiskPoolError::AlignmentError(format!("Read offset {} is not aligned to system alignment {}", offset, self.system_alignment)));
        }
        if buffer.len() == 0 {
            return Ok(()); // Reading zero bytes is a no-op
        }
        if buffer.len() % self.system_alignment != 0 {
            return Err(BuddyDiskPoolError::AlignmentError(format!("Read buffer length {} is not a multiple of system alignment {}", buffer.len(), self.system_alignment)));
        }
        if buffer.as_ptr() as usize % self.system_alignment != 0 {
             return Err(BuddyDiskPoolError::AlignmentError(format!("Read buffer address {:p} is not aligned to system alignment {}", buffer.as_ptr(), self.system_alignment)));
        }
        // Check if offset + buffer.len() is within file capacity
        if offset + buffer.len() > self.capacity {
            return Err(BuddyDiskPoolError::InvalidOffset(format!("Read request (offset {} + len {}) exceeds pool capacity {}", offset, buffer.len(), self.capacity)));
        }

        match uio::pread(&self.file, buffer, offset as libc::off_t) {
            Ok(bytes_read) if bytes_read == buffer.len() => Ok(()),
            Ok(bytes_read) => Err(BuddyDiskPoolError::IoError(io::Error::new(io::ErrorKind::Other, format!("Partial read: expected {}, got {}", buffer.len(), bytes_read)))),
            Err(e) => Err(BuddyDiskPoolError::NixError(e)),
        }
    }

    /// Writes data from the provided buffer to the disk pool.
    /// Offset must be from a previous `allocate` call.
    /// Buffer length must be <= allocated size for that offset, and meet O_DIRECT alignment.
    /// Buffer address must also meet O_DIRECT alignment.
    pub fn write(&self, offset: usize, buffer: &[u8]) -> Result<(), BuddyDiskPoolError> {
        if offset % self.system_alignment != 0 {
            return Err(BuddyDiskPoolError::AlignmentError(format!("Write offset {} is not aligned to system alignment {}", offset, self.system_alignment)));
        }
        if buffer.len() == 0 {
            return Ok(()); // Writing zero bytes is a no-op
        }
        if buffer.len() % self.system_alignment != 0 {
            return Err(BuddyDiskPoolError::AlignmentError(format!("Write buffer length {} is not a multiple of system alignment {}", buffer.len(), self.system_alignment)));
        }
        if buffer.as_ptr() as usize % self.system_alignment != 0 {
             return Err(BuddyDiskPoolError::AlignmentError(format!("Write buffer address {:p} is not aligned to system alignment {}", buffer.as_ptr(), self.system_alignment)));
        }
        if offset + buffer.len() > self.capacity {
            return Err(BuddyDiskPoolError::InvalidOffset(format!("Write request (offset {} + len {}) exceeds pool capacity {}", offset, buffer.len(), self.capacity)));
        }
        
        let mutable_buffer = unsafe { std::slice::from_raw_parts_mut(buffer.as_ptr() as *mut u8, buffer.len()) };
        match uio::pwrite(&self.file, mutable_buffer, offset as libc::off_t) {
            Ok(bytes_written) if bytes_written == buffer.len() => Ok(()),
            Ok(bytes_written) => Err(BuddyDiskPoolError::IoError(io::Error::new(io::ErrorKind::Other, format!("Partial write: expected {}, got {}", buffer.len(), bytes_written)))),
            Err(e) => Err(BuddyDiskPoolError::NixError(e)),
        }
    }
    
    // --- Getter methods for testing or info ---
    pub fn system_alignment(&self) -> usize { self.system_alignment }
    pub fn capacity(&self) -> usize { self.capacity }
    pub fn min_block_size(&self) -> usize { self.min_block_size }
    pub fn max_log_factor(&self) -> u32 { self.max_log_factor }
}

impl Drop for BuddyDiskPool {
    fn drop(&mut self) {
        // The file is closed automatically when `self.file` (File object) is dropped.
        // The temporary directory `self._temp_dir_handle` (TempDir object)
        // and its contents (including our pool file) are removed when it's dropped.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    fn get_system_alignment_for_test() -> usize {
        let temp_dir = tempfile::Builder::new().prefix("align_test_").tempdir().unwrap();
        let file_path = temp_dir.path().join("align_test.dat");
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).custom_flags(libc::O_DIRECT).open(&file_path).unwrap();
        let fs_stat = statfs::fstatfs(&file).unwrap();
        fs_stat.block_size() as usize
    }
    
    fn aligned_vec(size: usize, alignment: usize, fill_val: u8) -> Vec<u8> {
        let layout = Layout::from_size_align(size, alignment).expect("Failed to create layout for aligned_vec");
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            panic!("Failed to allocate aligned memory for aligned_vec");
        }
        let mut vec = unsafe { Vec::from_raw_parts(ptr, size, size) };
        vec.iter_mut().for_each(|x| *x = fill_val);
        // Important: To prevent double-free, Vec needs to "forget" it owns this memory if we dealloc manually,
        // or ensure Layout matches when it's dropped. Here, we let Vec manage it.
        vec
    }


    #[test]
    fn test_buddy_pool_new() {
        let sys_align = get_system_alignment_for_test();
        let min_req = sys_align / 2; // Test rounding up min_block_size
        let cap_req = sys_align * 8 - (sys_align / 2); // Test rounding down capacity

        let pool = BuddyDiskPool::new(cap_req, min_req).unwrap();

        assert_eq!(pool.system_alignment(), sys_align);
        assert_eq!(pool.min_block_size(), sys_align); // min_req rounded up
        assert_eq!(pool.capacity(), sys_align * 7);  // cap_req rounded down to multiple of new min_block_size (sys_align)
                                                     // (sys_align * 8 - sys_align/2) / sys_align = 7 (integer div)
                                                     // capacity = 7 * sys_align
        
        let expected_max_log_factor = (pool.capacity() / pool.min_block_size()).checked_ilog2().unwrap_or(0);
        assert_eq!(pool.max_log_factor, expected_max_log_factor);

        // Check initial free list state
        let mut total_free_slab_space = 0;
        let slabs_vec = &pool.slabs; // Avoid repeated locking if not using Mutex for slabs
        for layer_info in &pool.slab_layers {
            if let Some(mut head_idx) = layer_info.head_free_idx {
                let start_head_idx = head_idx;
                loop {
                    let slab = &slabs_vec[head_idx];
                    assert!(slab.free);
                    total_free_slab_space += pool.min_block_size() * (1 << slab.log_factor);
                    head_idx = slab.next_free_in_layer.unwrap();
                    if head_idx == start_head_idx { break; }
                }
            }
        }
        assert_eq!(total_free_slab_space, pool.capacity());
    }

    #[test]
    fn test_buddy_alloc_dealloc_simple() {
        let sys_align = get_system_alignment_for_test();
        let min_b = sys_align;
        let cap = min_b * 16; // 16 min_blocks, max_log_factor = 4
        let mut pool = BuddyDiskPool::new(cap, min_b).unwrap();

        // Allocate a block of min_block_size
        let offset1 = pool.allocate(min_b).unwrap();
        assert_eq!(offset1 % sys_align, 0);

        // Allocate another
        let offset2 = pool.allocate(min_b * 2).unwrap(); // Needs log_factor 1
        assert_eq!(offset2 % sys_align, 0);
        assert_ne!(offset1, offset2);

        pool.deallocate(offset1, min_b).unwrap();
        pool.deallocate(offset2, min_b * 2).unwrap();

        // Try to allocate the whole thing again
        let offset_full = pool.allocate(cap).unwrap();
        assert_eq!(offset_full, 0);
        pool.deallocate(offset_full, cap).unwrap();
    }

    #[test]
    fn test_buddy_alloc_split_and_merge() {
        let sys_align = get_system_alignment_for_test();
        let min_b = sys_align;
        let cap = min_b * 4; // max_log_factor = 2 (blocks of size min_b, 2*min_b, 4*min_b)
        let mut pool = BuddyDiskPool::new(cap, min_b).unwrap();
        
        // Initial state: one block of 4*min_b at log_factor 2, offset 0
        // lf 0: []
        // lf 1: []
        // lf 2: [idx_A(off=0, sz=4*min_b)]

        let off1 = pool.allocate(min_b).unwrap(); // Needs lf 0
        // Split A (lf 2) -> B(lf 1, off=0), C(lf 1, off=2*min_b)
        // B is used for further split. C is free.
        // Split B (lf 1) -> D(lf 0, off=0), E(lf 0, off=min_b)
        // D is allocated (off1). E is free.
        // State:
        // lf 0: [idx_E(off=min_b)]
        // lf 1: [idx_C(off=2*min_b)]
        // lf 2: []
        // Active: idx_D(off=0)
        assert_eq!(off1, 0);

        let off2 = pool.allocate(min_b).unwrap(); // Needs lf 0. Takes E.
        // State:
        // lf 0: []
        // lf 1: [idx_C(off=2*min_b)]
        // lf 2: []
        // Active: idx_D(off=0), idx_E(off=min_b)
        assert_eq!(off2, min_b);

        pool.deallocate(off1, min_b).unwrap(); // Free D (lf 0, off=0)
        // D becomes free. Try merge D and E. Parent is B.
        // E is not free (it's off2). So no merge of D+E.
        // State:
        // lf 0: [idx_D(off=0)] (D is now free)
        // lf 1: [idx_C(off=2*min_b)]
        // Active: idx_E(off=min_b)

        pool.deallocate(off2, min_b).unwrap(); // Free E (lf 0, off=min_b)
        // E becomes free. Try merge D and E. Parent is B. Both D,E free.
        // Merge D,E into B (lf 1, off=0). D,E slabinfo released. B becomes free.
        // Try merge B and C. Parent is A.
        // B (lf 1, off=0) is free. C (lf 1, off=2*min_b) is free.
        // Merge B,C into A (lf 2, off=0). B,C slabinfo released. A becomes free.
        // State:
        // lf 0: []
        // lf 1: []
        // lf 2: [idx_A(off=0)]
        // Active: []

        // Allocate the whole thing
        let off_final = pool.allocate(cap).unwrap();
        assert_eq!(off_final, 0);
        pool.deallocate(off_final, cap).unwrap();
    }

    #[test]
    fn test_buddy_read_write() {
        let sys_align = get_system_alignment_for_test();
        let min_b = sys_align;
        let cap = min_b * 8;
        let mut pool = BuddyDiskPool::new(cap, min_b).unwrap();

        let alloc_size = min_b * 2; // lf 1
        let offset = pool.allocate(alloc_size).unwrap();

        let write_buf = aligned_vec(alloc_size, sys_align, 0xAB);
        let mut read_buf = aligned_vec(alloc_size, sys_align, 0);
        
        pool.write(offset, &write_buf).unwrap();
        pool.read(offset, &mut read_buf).unwrap();
        assert_eq!(write_buf, read_buf, "Full block read/write mismatch");

        // Partial read/write if block is larger than one system_alignment unit
        if alloc_size > sys_align {
            let part_size = sys_align;
            let write_buf_part = aligned_vec(part_size, sys_align, 0xCD);
            let mut read_buf_part = aligned_vec(part_size, sys_align, 0);

            // Write to second sys_align chunk
            pool.write(offset + sys_align, &write_buf_part).unwrap();
            // Read from second sys_align chunk
            pool.read(offset + sys_align, &mut read_buf_part).unwrap();
            assert_eq!(write_buf_part, read_buf_part, "Partial block read/write mismatch");

            // Verify first part is untouched by the partial write
            let mut first_part_check_buf = aligned_vec(sys_align, sys_align, 0);
            pool.read(offset, &mut first_part_check_buf).unwrap();
            let expected_first_part_data: Vec<u8> = write_buf.iter().take(sys_align).cloned().collect();
            assert_eq!(first_part_check_buf, expected_first_part_data, "First part of block was modified unexpectedly");
        }
        pool.deallocate(offset, alloc_size).unwrap();
    }
    
    #[test]
    fn test_out_of_memory_buddy() {
        let sys_align = get_system_alignment_for_test();
        let min_b = sys_align;
        let cap = min_b * 2; // Small pool
        let mut pool = BuddyDiskPool::new(cap, min_b).unwrap();

        let _off1 = pool.allocate(min_b * 2).unwrap(); // Allocate everything

        match pool.allocate(min_b) {
            Err(BuddyDiskPoolError::OutOfMemory) => {}, // Expected
            Ok(_) => panic!("Should be out of memory"),
            Err(e) => panic!("Unexpected error type for OOM: {:?}", e),
        }
    }
}