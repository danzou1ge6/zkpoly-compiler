#![allow(dead_code, unused_imports, unused_variables)] // TODO: Remove as implementation progresses

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::raw::c_void;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::Mutex; // Using Mutex for interior mutability of free_lists etc.

use libc;
use nix::fcntl::{self, OFlag};
use nix::sys::{statfs, uio};
use nix::unistd;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use zkpoly_cuda_api::bindings::{
    cuFileHandleDeregister, cuFileRead, cuFileWrite, cudaSetDevice, CUfileDescr_t, CUfileHandle_t,
};
use zkpoly_cuda_api::cuda_check;
use zkpoly_cuda_api::file::register_cufile;

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

pub type DiskMemoryPool = Vec<BuddyDiskPool>;

pub struct BuddyDiskPool {
    file: File,
    file_path: PathBuf, // To aid in debugging or potential re-opening
    temp_dir_handle: Option<tempfile::TempDir>,
    _cu_file_descr: CUfileDescr_t, // cuFile descriptor for GPU direct I/O
    cu_file_handle: CUfileHandle_t, // cuFile handle for GPU direct I/O

    slabs: Vec<DiskSlabInfo>,
    free_slab_info_indices: Vec<usize>,
    slab_layers: Vec<SlabLayerInfo>,

    // Maps an allocated offset to its DiskSlabInfo index in slabs.
    active_allocations: HashMap<usize, usize>,

    capacity: usize,         // Current total usable capacity, aligned to system_alignment
    system_alignment: usize, // Alignment required by O_DIRECT and filesystem
    min_block_size: usize,   // Smallest allocatable block size, aligned to system_alignment
    max_log_factor: u32,     // Max log_factor relative to min_block_size
                            // num_levels = max_log_factor + 1
    max_block_size: usize,   // Size of the largest allocatable block
}

#[derive(Debug, Clone)]
pub struct DiskAllocInfo {
    pub offset: usize, // offset in the file where the allocation starts
    pub fd: i32,       // file descriptor of the disk pool file
    pub cu_file_handle: CUfileHandle_t, // cuFile handle for GPU direct I/O
}

// Safety: cuFile api is thread-safe
unsafe impl Send for DiskAllocInfo {}
unsafe impl Sync for DiskAllocInfo {}

impl DiskAllocInfo {
    pub fn new(offset: usize, disk_pool: &BuddyDiskPool) -> Self {
        DiskAllocInfo {
            offset,
            fd: disk_pool.get_fd(),
            cu_file_handle: disk_pool.cu_file_handle,
        }
    }
}

// Wrap mutable parts in a Mutex if BuddyDiskPool needs to be Sync
// For now, assuming &mut self for main operations. If methods take &self, then Mutex is needed.
// Let's make the core mutable structures (slabs, free_slab_info_indices, slab_layers, active_allocations)
// part of an inner struct guarded by a Mutex if we want &self methods for allocation/deallocation.
// Or, keep them as direct fields and use &mut self. The original MemoryPool uses &mut self.

impl BuddyDiskPool {
    pub fn get_alignment(&self) -> usize {
        self.system_alignment
    }

    pub fn get_block_size(&self, log_factor: u32) -> usize {
        self.min_block_size * (1 << log_factor)
    }

    /// keep the temporary directory and its contents (including the pool file)
    pub fn keep(&mut self) -> PathBuf {
        self.temp_dir_handle.take().unwrap().keep()
    }

    pub fn new(
        max_block_size: usize,
        tmp_dir: Option<PathBuf>, // Temporary directory for the pool file
    ) -> Result<Self, BuddyDiskPoolError> {
        if max_block_size == 0 {
            return Err(BuddyDiskPoolError::InvalidSize(
                "Max block size cannot be zero.".to_string(),
            ));
        }

        let temp_dir = if tmp_dir.is_some() {
            tempfile::Builder::new()
                .prefix("buddy_disk_pool_")
                .tempdir_in(tmp_dir.unwrap())?
        } else {
            tempfile::Builder::new()
                .prefix("buddy_disk_pool_")
                .tempdir()?
        };
        let file_path = temp_dir.path().join("buddy_pool.dat");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(&file_path)
            .map_err(|e| {
                BuddyDiskPoolError::FileOpenError(format!("Failed to open {:?}: {}", file_path, e))
            })?;

        // Get system alignment
        let system_alignment = match statfs::fstatfs(&file) {
            Ok(fs_stat) => fs_stat.block_size() as usize,
            Err(e) => return Err(BuddyDiskPoolError::NixError(e)),
        };

        if system_alignment == 0 || !system_alignment.is_power_of_two() {
            return Err(BuddyDiskPoolError::AlignmentError(format!(
                "Invalid system alignment {}",
                system_alignment
            )));
        }
        
        // Align max_block_size to system_alignment
        let max_block_size = (max_block_size + system_alignment - 1) / system_alignment * system_alignment;
        
        // Calculate min_block_size and max_log_factor from max_block_size
        let min_block_size = system_alignment;
        let max_log_factor = (max_block_size / min_block_size).ilog2();

        // Initially we have just one max-sized block
        let initial_capacity = max_block_size;
        
        // Allocate initial file space
        fcntl::fallocate(
            &file,
            fcntl::FallocateFlags::empty(),
            0,
            initial_capacity as libc::off_t,
        )
        .map_err(|e| {
            BuddyDiskPoolError::FallocateError(format!(
                "fallocate failed for size {}: {}",
                initial_capacity, e
            ))
        })?;

        let num_layers = (max_log_factor + 1) as usize;

        // now we need to register the file to cuFile for GPU direct I/O
        let (cu_file_descr, cu_file_handle) = register_cufile(&file);

        let mut pool = Self {
            file,
            file_path,
            temp_dir_handle: Some(temp_dir),
            slabs: Vec::new(),
            free_slab_info_indices: Vec::new(),
            slab_layers: vec![SlabLayerInfo::default(); num_layers],
            active_allocations: HashMap::new(),
            capacity: initial_capacity,
            system_alignment,
            min_block_size,
            max_log_factor,
            max_block_size,
            _cu_file_descr: cu_file_descr,
            cu_file_handle,
        };

        // Initialize the first max-sized block
        let slab_idx = pool.obtain_slab_info_index(0, max_log_factor, None);
        pool.layer_insert_all(max_log_factor, slab_idx);
        pool.layer_insert_free(max_log_factor, slab_idx);

        if !pool.slabs[slab_idx].free {
            return Err(BuddyDiskPoolError::InternalError(
                "Initial block not marked as free".to_string(),
            ));
        }
        Ok(pool)
    }

    fn obtain_slab_info_index(
        &mut self,
        offset: usize,
        log_factor: u32,
        parent_idx: Option<usize>,
    ) -> usize {
        if let Some(idx) = self.free_slab_info_indices.pop() {
            self.slabs[idx] = DiskSlabInfo::new(offset, log_factor, parent_idx);
            idx
        } else {
            self.slabs
                .push(DiskSlabInfo::new(offset, log_factor, parent_idx));
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
                let head_prev_idx = self.slabs[head_idx]
                    .prev_in_layer
                    .expect("Head of 'all' list must have a prev pointer");
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
        if slab_next_opt == Some(slab_idx) {
            // Only element
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
                let head_prev_idx = self.slabs[head_idx]
                    .prev_free_in_layer
                    .expect("Head of 'free' list must have a prev pointer");
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
            (
                slab_to_remove.next_free_in_layer,
                slab_to_remove.prev_free_in_layer,
            )
        };
        if slab_next_opt.is_none()
            && slab_prev_opt.is_none()
            && self.slab_layers[log_factor as usize].head_free_idx != Some(slab_idx)
        {
            // Not in the free list, or list is corrupted. This can happen if already removed.
            // This function should be idempotent or ensure it's only called once.
            return;
        }

        if slab_next_opt == Some(slab_idx) {
            // Only element
            self.slab_layers[log_factor as usize].head_free_idx = None;
        } else {
            // These expects will panic if the slab was not actually in a list.
            let next_s_idx = slab_next_opt
                .ok_or_else(|| {
                    BuddyDiskPoolError::InternalError(format!(
                        "Slab {} (lf {}) not in free list (no next_free_in_layer)",
                        slab_idx, log_factor
                    ))
                })
                .unwrap(); // TODO: Return Result
            let prev_s_idx = slab_prev_opt
                .ok_or_else(|| {
                    BuddyDiskPoolError::InternalError(format!(
                        "Slab {} (lf {}) not in free list (no prev_free_in_layer)",
                        slab_idx, log_factor
                    ))
                })
                .unwrap(); // TODO: Return Result

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
            return Err(BuddyDiskPoolError::InternalError(
                "Cannot split slab of log_factor 0".to_string(),
            ));
        }
        if parent_log_factor > self.max_log_factor {
            return Err(BuddyDiskPoolError::InternalError(format!(
                "Cannot split from log_factor {} > max_log_factor {}",
                parent_log_factor, self.max_log_factor
            )));
        }

        let parent_slab_idx = self
            .get_first_free_slab_in_layer(parent_log_factor)
            .ok_or_else(|| {
                BuddyDiskPoolError::InternalError(format!(
                    "No free slab to split at log_factor {}",
                    parent_log_factor
                ))
            })?;

        self.slabs[parent_slab_idx].free = false;
        self.layer_remove_free(parent_log_factor, parent_slab_idx);

        let child_log_factor = parent_log_factor - 1;
        let parent_offset = self.slabs[parent_slab_idx].offset;
        let child_actual_size = self.min_block_size * (1 << child_log_factor);

        let lchild_offset = parent_offset;
        let rchild_offset = parent_offset + child_actual_size;

        let lchild_idx =
            self.obtain_slab_info_index(lchild_offset, child_log_factor, Some(parent_slab_idx));
        let rchild_idx =
            self.obtain_slab_info_index(rchild_offset, child_log_factor, Some(parent_slab_idx));

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

    /// Ensures a free slab exists at the given `log_factor`.
    /// If no free slab exists at this level:
    /// - For max_log_factor, it will try to expand the file by allocating a new max-sized block
    /// - For other levels, it will split a larger block from a higher level
    /// Returns the index of a free slab at this `log_factor`.
    fn ensure_free_slab_exists(&mut self, log_factor: u32) -> Result<usize, BuddyDiskPoolError> {
        if self.get_first_free_slab_in_layer(log_factor).is_some() {
            return Ok(self.get_first_free_slab_in_layer(log_factor).unwrap());
        }

        if log_factor == self.max_log_factor {
            // Try expanding the file by one max block size
            let new_capacity = self.capacity + self.max_block_size;
            
            // Extend the file
            fcntl::fallocate(
                &self.file,
                fcntl::FallocateFlags::empty(),
                self.capacity as i64,
                self.max_block_size as i64,
            ).map_err(|e| {
                BuddyDiskPoolError::FallocateError(format!(
                    "fallocate failed for expansion size {}: {}",
                    self.max_block_size, e
                ))
            })?;

            // Create a new slab for the expanded space
            let new_slab_idx = self.obtain_slab_info_index(self.capacity, self.max_log_factor, None);
            
            // Add to the appropriate lists
            self.layer_insert_all(self.max_log_factor, new_slab_idx);
            self.layer_insert_free(self.max_log_factor, new_slab_idx);
            
            // Update capacity after successful slab creation
            self.capacity = new_capacity;
            
            return Ok(new_slab_idx);
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
            return Err(BuddyDiskPoolError::InvalidSize(
                "Allocation size cannot be zero.".to_string(),
            ));
        }
        if size_bytes > self.capacity {
            // Check against total pool capacity
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

    pub fn get_fd(&self) -> i32 {
        self.file.as_raw_fd()
    }

    pub fn allocate(&mut self, size_bytes: usize) -> Result<usize, BuddyDiskPoolError> {
        if size_bytes == 0 {
            // Or return a special offset like 0 with size 0, if that's meaningful.
            // For now, error on zero size.
            return Err(BuddyDiskPoolError::InvalidSize(
                "Cannot allocate zero bytes.".to_string(),
            ));
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
                return Err(BuddyDiskPoolError::InternalError(format!(
                    "Parent slab {} is missing child information during merge attempt for slab {}.",
                    parent_idx, slab_idx
                )));
            }
        }
        Ok(())
    }

    pub fn deallocate(&mut self, offset: usize) -> Result<(), BuddyDiskPoolError> {
        let slab_idx = self.active_allocations.remove(&offset).ok_or_else(|| {
            BuddyDiskPoolError::BlockNotFound(format!(
                "Offset {} not found in active allocations or already deallocated.",
                offset
            ))
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
            return Err(BuddyDiskPoolError::AlignmentError(format!(
                "Read offset {} is not aligned to system alignment {}",
                offset, self.system_alignment
            )));
        }
        if buffer.len() == 0 {
            return Ok(()); // Reading zero bytes is a no-op
        }
        if buffer.len() % self.system_alignment != 0 {
            return Err(BuddyDiskPoolError::AlignmentError(format!(
                "Read buffer length {} is not a multiple of system alignment {}",
                buffer.len(),
                self.system_alignment
            )));
        }
        if buffer.as_ptr() as usize % self.system_alignment != 0 {
            return Err(BuddyDiskPoolError::AlignmentError(format!(
                "Read buffer address {:p} is not aligned to system alignment {}",
                buffer.as_ptr(),
                self.system_alignment
            )));
        }
        // Check if offset + buffer.len() is within file capacity
        if offset + buffer.len() > self.capacity {
            return Err(BuddyDiskPoolError::InvalidOffset(format!(
                "Read request (offset {} + len {}) exceeds pool capacity {}",
                offset,
                buffer.len(),
                self.capacity
            )));
        }

        match uio::pread(&self.file, buffer, offset as libc::off_t) {
            Ok(bytes_read) if bytes_read == buffer.len() => Ok(()),
            Ok(bytes_read) => Err(BuddyDiskPoolError::IoError(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Partial read: expected {}, got {}",
                    buffer.len(),
                    bytes_read
                ),
            ))),
            Err(e) => Err(BuddyDiskPoolError::NixError(e)),
        }
    }

    /// Writes data from the provided buffer to the disk pool.
    /// Offset must be from a previous `allocate` call.
    /// Buffer length must be <= allocated size for that offset, and meet O_DIRECT alignment.
    /// Buffer address must also meet O_DIRECT alignment.
    pub fn write(&self, offset: usize, buffer: &[u8]) -> Result<(), BuddyDiskPoolError> {
        if offset % self.system_alignment != 0 {
            return Err(BuddyDiskPoolError::AlignmentError(format!(
                "Write offset {} is not aligned to system alignment {}",
                offset, self.system_alignment
            )));
        }
        if buffer.len() == 0 {
            return Ok(()); // Writing zero bytes is a no-op
        }
        if buffer.len() % self.system_alignment != 0 {
            return Err(BuddyDiskPoolError::AlignmentError(format!(
                "Write buffer length {} is not a multiple of system alignment {}",
                buffer.len(),
                self.system_alignment
            )));
        }
        if buffer.as_ptr() as usize % self.system_alignment != 0 {
            return Err(BuddyDiskPoolError::AlignmentError(format!(
                "Write buffer address {:p} is not aligned to system alignment {}",
                buffer.as_ptr(),
                self.system_alignment
            )));
        }
        if offset + buffer.len() > self.capacity {
            return Err(BuddyDiskPoolError::InvalidOffset(format!(
                "Write request (offset {} + len {}) exceeds pool capacity {}",
                offset,
                buffer.len(),
                self.capacity
            )));
        }

        let mutable_buffer =
            unsafe { std::slice::from_raw_parts_mut(buffer.as_ptr() as *mut u8, buffer.len()) };
        match uio::pwrite(&self.file, mutable_buffer, offset as libc::off_t) {
            Ok(bytes_written) if bytes_written == buffer.len() => Ok(()),
            Ok(bytes_written) => Err(BuddyDiskPoolError::IoError(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Partial write: expected {}, got {}",
                    buffer.len(),
                    bytes_written
                ),
            ))),
            Err(e) => Err(BuddyDiskPoolError::NixError(e)),
        }
    }

    // --- Getter methods for testing or info ---
    pub fn system_alignment(&self) -> usize {
        self.system_alignment
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn min_block_size(&self) -> usize {
        self.min_block_size
    }
    pub fn max_log_factor(&self) -> u32 {
        self.max_log_factor
    }
}

impl Drop for BuddyDiskPool {
    fn drop(&mut self) {
        unsafe { cuFileHandleDeregister(self.cu_file_handle) };
        // The file is closed automatically when `self.file` (File object) is dropped.
        // The temporary directory `self._temp_dir_handle` (TempDir object)
        // and its contents (including our pool file) are removed when it's dropped.
    }
}

#[derive(Clone, Debug)]
struct SafePtr<T> {
    ptr: *const T,
}

unsafe impl<T> Send for SafePtr<T> {}
unsafe impl<T> Sync for SafePtr<T> {}

#[derive(Clone, Debug)]
struct SafePtrMut<T> {
    ptr: *mut T,
}

unsafe impl<T> Send for SafePtrMut<T> {}
unsafe impl<T> Sync for SafePtrMut<T> {}

// Helper functions for reading/writing to disk using O_DIRECT
pub fn cpu_write_to_disk(ptr: *const u8, disk_pos: &Vec<DiskAllocInfo>, size: usize) {
    let safe_ptr = SafePtr { ptr };
    let part_size = size / disk_pos.len();
    disk_pos
        .par_iter()
        .enumerate()
        .for_each(|(disk_id, alloc_info)| {
            let safe_ptr_clone = safe_ptr.clone();
            let ptr = safe_ptr_clone.ptr;
            let offset = alloc_info.offset.clone();
            let fd = alloc_info.fd.clone();
            let cpu_offset = part_size * disk_id;

            unsafe {
                let res = libc::pwrite(fd, ptr.add(cpu_offset).cast(), part_size, offset as i64);
                if res < part_size as isize {
                    panic!(
                        "Failed to write {} bytes to disk {} at offset {}: {}",
                        part_size, disk_id, offset, res
                    );
                }
            }
        });
}

pub fn cpu_read_from_disk(ptr: *mut u8, disk_pos: &Vec<DiskAllocInfo>, size: usize) {
    let safe_ptr = SafePtrMut { ptr };
    let part_size = size / disk_pos.len();

    disk_pos
        .par_iter()
        .enumerate()
        .for_each(|(disk_id, alloc_info)| {
            let safe_ptr_clone = safe_ptr.clone();
            let ptr = safe_ptr_clone.ptr;
            let (fd, offset) = (alloc_info.fd.clone(), alloc_info.offset.clone());
            let cpu_offset = part_size * disk_id;

            unsafe {
                let res = libc::pread(fd, ptr.add(cpu_offset).cast(), part_size, offset as i64);
                if res < part_size as isize {
                    panic!(
                        "Failed to read {} bytes from disk {} at offset {}: {}",
                        part_size, disk_id, offset, res
                    );
                }
            }
        });
}

pub fn gpu_write_to_disk(
    ptr: *const u8,
    disk_pos: &Vec<DiskAllocInfo>,
    size: usize,
    device_id: i32,
) {
    let safe_ptr = SafePtr { ptr };
    let part_size = size / disk_pos.len();
    disk_pos
        .par_iter()
        .enumerate()
        .for_each(|(disk_id, alloc_info)| {
            let safe_ptr_clone = safe_ptr.clone();
            let ptr = safe_ptr_clone.ptr;
            let offset = alloc_info.offset.clone();
            let file_handle = alloc_info.cu_file_handle.clone();
            let cpu_offset = part_size * disk_id;

            unsafe {
                cuda_check!(cudaSetDevice(device_id));
                let res = cuFileWrite(
                    file_handle,
                    ptr as *const c_void,
                    part_size,
                    offset as i64,
                    cpu_offset as i64,
                );
                if res < part_size as isize {
                    panic!(
                        "Failed to write {} bytes to disk {} at offset {}: {}",
                        part_size, disk_id, offset, res
                    );
                }
            }
        });
}

pub fn gpu_read_from_disk(
    ptr: *mut u8,
    disk_pos: &Vec<DiskAllocInfo>,
    size: usize,
    device_id: i32,
) {
    let safe_ptr = SafePtrMut { ptr };
    let part_size = size / disk_pos.len();

    disk_pos
        .par_iter()
        .enumerate()
        .for_each(|(disk_id, alloc_info)| {
            let safe_ptr_clone = safe_ptr.clone();
            let ptr = safe_ptr_clone.ptr;
            let file_handle = alloc_info.cu_file_handle.clone();
            let offset = alloc_info.offset.clone();
            let cpu_offset = part_size * disk_id;

            unsafe {
                cuda_check!(cudaSetDevice(device_id));
                let res = cuFileRead(
                    file_handle,
                    ptr as *mut c_void,
                    part_size,
                    offset as i64,
                    cpu_offset as i64,
                );
                if res < part_size as isize {
                    panic!(
                        "Failed to read {} bytes from disk {} at offset {}: {}",
                        part_size, disk_id, offset, res
                    );
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    fn get_system_alignment_for_test() -> usize {
        let temp_dir = tempfile::Builder::new()
            .prefix("align_test_")
            .tempdir()
            .unwrap();
        let file_path = temp_dir.path().join("align_test.dat");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(&file_path)
            .unwrap();
        let fs_stat = statfs::fstatfs(&file).unwrap();
        fs_stat.block_size() as usize
    }

    fn aligned_vec(size: usize, alignment: usize, fill_val: u8) -> Vec<u8> {
        let layout = Layout::from_size_align(size, alignment)
            .expect("Failed to create layout for aligned_vec");
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
        let max_block_size = sys_align * 8;

        let pool = BuddyDiskPool::new(max_block_size, None).unwrap();

        assert_eq!(pool.get_alignment(), sys_align);
        assert_eq!(pool.min_block_size, sys_align);
        assert_eq!(pool.max_block_size, max_block_size);
        assert_eq!(pool.capacity, max_block_size); // Initially just one max block

        let expected_max_log_factor = (max_block_size / sys_align).ilog2();
        assert_eq!(pool.max_log_factor, expected_max_log_factor);

        // Initial state should be just one max-sized free block
        let mut free_blocks = 0;
        let mut total_free_size = 0;
        for layer_info in &pool.slab_layers {
            if let Some(mut head_idx) = layer_info.head_free_idx {
                let start_idx = head_idx;
                loop {
                    let slab = &pool.slabs[head_idx];
                    assert!(slab.free);
                    free_blocks += 1;
                    total_free_size += pool.get_block_size(slab.log_factor);
                    head_idx = slab.next_free_in_layer.unwrap();
                    if head_idx == start_idx {
                        break;
                    }
                }
            }
        }
        assert_eq!(free_blocks, 1); // Should only have one free block
        assert_eq!(total_free_size, max_block_size); // All space should be free
    }

    #[test]
    fn test_buddy_alloc_dealloc_simple() {
        let sys_align = get_system_alignment_for_test();
        let max_block_size = sys_align * 16; // max_log_factor = 4
        let mut pool = BuddyDiskPool::new(max_block_size, None).unwrap();

        // Allocate a block of min_block_size
        let offset1 = pool.allocate(sys_align).unwrap();
        assert_eq!(offset1 % sys_align, 0);

        // Allocate another
        let offset2 = pool.allocate(sys_align * 2).unwrap(); // Needs log_factor 1
        assert_eq!(offset2 % sys_align, 0);
        assert_ne!(offset1, offset2);

        pool.deallocate(offset1).unwrap();
        pool.deallocate(offset2).unwrap();

        // Try to allocate the whole thing again
        let offset3 = pool.allocate(max_block_size).unwrap();
        assert_eq!(offset3, 0);
        pool.deallocate(offset3).unwrap();
    }

    #[test]
    fn test_buddy_alloc_split_and_merge() {
        let sys_align = get_system_alignment_for_test();
        let min_b = sys_align;
        let max_block_size = min_b * 4; // max_log_factor = 2 (blocks of size min_b, 2*min_b, 4*min_b)
        let mut pool = BuddyDiskPool::new(max_block_size, None).unwrap();

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

        pool.deallocate(off1).unwrap(); // Free D (lf 0, off=0)
                                               // D becomes free. Try merge D and E. Parent is B.
                                               // E is not free (it's off2). So no merge of D+E.
                                               // State:
                                               // lf 0: [idx_D(off=0)] (D is now free)
                                               // lf 1: [idx_C(off=2*min_b)]
                                               // Active: idx_E(off=min_b)

        pool.deallocate(off2).unwrap(); // Free E (lf 0, off=min_b)
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
        let off_final = pool.allocate(max_block_size).unwrap();
        assert_eq!(off_final, 0);
        pool.deallocate(off_final).unwrap();
    }

    #[test]
    fn test_buddy_read_write() {
        let sys_align = get_system_alignment_for_test();
        let max_block_size = sys_align * 8;
        let mut pool = BuddyDiskPool::new(max_block_size, None).unwrap();

        let alloc_size = sys_align * 2; // Request 2 minimum blocks
        let offset = pool.allocate(alloc_size).unwrap();

        // Test full block write/read
        let write_buf = aligned_vec(max_block_size, sys_align, 0xAB);
        let mut read_buf = aligned_vec(max_block_size, sys_align, 0);

        pool.write(offset, &write_buf).unwrap();
        pool.read(offset, &mut read_buf).unwrap();
        assert_eq!(write_buf, read_buf, "Full block read/write mismatch");

        // Test partial block write/read
        let partial_size = sys_align;
        let write_buf_part = aligned_vec(partial_size, sys_align, 0xCD);
        let mut read_buf_part = aligned_vec(partial_size, sys_align, 0);

        // Write to second alignment unit
        pool.write(offset + sys_align, &write_buf_part).unwrap();
        pool.read(offset + sys_align, &mut read_buf_part).unwrap();
        assert_eq!(write_buf_part, read_buf_part, "Partial read/write mismatch");

        // Verify first part is unchanged
        let mut first_part_buf = aligned_vec(sys_align, sys_align, 0);
        pool.read(offset, &mut first_part_buf).unwrap();
        assert_eq!(first_part_buf[0], 0xAB, "First part modified unexpectedly");

        pool.deallocate(offset).unwrap();
    }

    #[test]
    fn test_out_of_memory_buddy() {
        let sys_align = get_system_alignment_for_test();
        let max_block_size = sys_align * 2; // Small pool
        let mut pool = BuddyDiskPool::new(max_block_size, None).unwrap();

        // First allocation should work
        let off1 = pool.allocate(max_block_size).unwrap();

        // Second allocation should fail
        match pool.allocate(sys_align) {
            Err(BuddyDiskPoolError::OutOfMemory) => panic!("should auto expand"),
            Ok(_) => {},
            Err(e) => panic!("Unexpected error type for OOM: {:?}", e),
        }

        // After freeing, we should be able to allocate again
        pool.deallocate(off1).unwrap();
        let off2 = pool.allocate(max_block_size).unwrap();
        pool.deallocate(off2).unwrap();
    }
}
