use std::collections::HashMap;
use std::ptr::NonNull;
use std::ffi::c_void;
use std::path::PathBuf; // For swap_out
use zkpoly_cuda_api::bindings::{cudaError_cudaSuccess, cudaFreeHost, cudaMallocHost};

/// Enum to represent slab location, considering swap-out
#[derive(Debug)]
enum SlabLocation {
    InMemory(NonNull<c_void>),
    SwappedOut { path: PathBuf, offset: u64, size: usize }, // Placeholder for swap details
}

/// Metadata for a memory slab.
#[derive(Debug)]
struct SlabInfo {
    /// Actual memory pointer or swap location.
    /// `None` if this slab is a non-leaf node in the buddy tree and its memory is fully covered by its children.
    location: Option<SlabLocation>,
    
    /// Size of the slab = base_size * (2^log_factor).
    log_factor: u32,
    
    /// Whether the slab is free to be allocated.
    free: bool,
    
    /// Index of the parent slab in the `MemoryPool::slabs` vector.
    parent: Option<usize>,
    /// Index of the left child slab.
    lchild: Option<usize>,
    /// Index of the right child slab.
    rchild: Option<usize>,
    
    /// Index of the next slab in the linked list of all slabs in the same layer.
    next_in_layer: Option<usize>,
    /// Index of the previous slab in the linked list of all slabs in the same layer.
    prev_in_layer: Option<usize>,
    
    /// Index of the next slab in the linked list of free slabs in the same layer.
    next_free_in_layer: Option<usize>,
    /// Index of the previous slab in the linked list of free slabs in the same layer.
    prev_free_in_layer: Option<usize>,
}

impl SlabInfo {
    fn new(log_factor: u32, parent: Option<usize>) -> Self {
        SlabInfo {
            location: None,
            log_factor,
            free: true, // New slabs are initially free
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
    /// Head of the "all slabs" list for this layer. Points to an index in `MemoryPool::slabs`.
    head_all_idx: Option<usize>, 
    /// Head of the "free slabs" list for this layer. Points to an index in `MemoryPool::slabs`.
    head_free_idx: Option<usize>,
}

/// The main memory pool manager.
#[derive(Debug)]
pub struct MemoryPool {
    /// Storage for all slab metadata. Slabs are never removed from this vector, only marked as unused.
    slabs: Vec<SlabInfo>,
    
    /// Indices of `SlabInfo` structs in the `slabs` vector that are currently unused
    /// and can be reclaimed for new slab metadata.
    free_slab_info_indices: Vec<usize>,
    
    /// Information for each slab layer, indexed by `log_factor`.
    /// `slab_layers[i]` corresponds to slabs of size `base_size * 2^i`.
    slab_layers: Vec<SlabLayerInfo>,
    
    /// Maps an active memory pointer (if `InMemory`) to its `SlabInfo` index in `slabs`.
    active_allocations: HashMap<NonNull<c_void>, usize>,
    
    max_log_factor: u32,
    base_size: usize,      // Smallest allocatable unit size.

    // Placeholder for future swap management functionality
    // swap_manager: Option<SwapManager>,
}

impl MemoryPool {
    pub fn new(max_log_factor: u32, base_size: usize) -> Self {
        let num_layers = (max_log_factor + 1) as usize;
        MemoryPool {
            slabs: Vec::new(),
            free_slab_info_indices: Vec::new(),
            slab_layers: vec![SlabLayerInfo::default(); num_layers],
            active_allocations: HashMap::new(),
            max_log_factor,
            base_size,
        }
    }

    /// Gets a new or recycled `SlabInfo` index and initializes the `SlabInfo` struct.
    fn obtain_slab_info_index(&mut self, log_factor: u32, parent_idx: Option<usize>) -> usize {
        if let Some(idx) = self.free_slab_info_indices.pop() {
            self.slabs[idx] = SlabInfo::new(log_factor, parent_idx);
            idx
        } else {
            self.slabs.push(SlabInfo::new(log_factor, parent_idx));
            self.slabs.len() - 1
        }
    }

    /// Marks a `SlabInfo` index as available for reuse.
    fn release_slab_info_index(&mut self, slab_idx: usize) {
        // Optionally, clear/reset the SlabInfo at slab_idx before adding to free list.
        // self.slabs[slab_idx] = SlabInfo::new(0, None); // Example reset
        self.free_slab_info_indices.push(slab_idx);
    }

    // --- Linked list helpers for "all slabs in layer" ---

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
            None => { // First slab in this layer
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

        if slab_next_opt == Some(slab_idx) { // Only element in list
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

    // --- Linked list helpers for "free slabs in layer" ---

    fn layer_insert_free(&mut self, log_factor: u32, slab_idx: usize) {
        let layer_info = &mut self.slab_layers[log_factor as usize];
        // assert!(self.slabs[slab_idx].free, "Only free slabs can be added to free list");

        match layer_info.head_free_idx {
            Some(head_idx) => {
                let head_prev_idx = self.slabs[head_idx].prev_free_in_layer.expect("Head of 'free' list must have a prev pointer");
                
                self.slabs[slab_idx].next_free_in_layer = Some(head_idx);
                self.slabs[slab_idx].prev_free_in_layer = Some(head_prev_idx);

                self.slabs[head_idx].prev_free_in_layer = Some(slab_idx);
                self.slabs[head_prev_idx].next_free_in_layer = Some(slab_idx);
            }
            None => { // First free slab in this layer
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

        if slab_next_opt == Some(slab_idx) { // Only element
            self.slab_layers[log_factor as usize].head_free_idx = None;
        } else {
            let next_s_idx = slab_next_opt.expect("Slab in free list must have next pointer");
            let prev_s_idx = slab_prev_opt.expect("Slab in free list must have prev pointer");
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
        self.slab_layers[log_factor as usize].head_free_idx
    }

    /// Allocates physical memory using `cudaMallocHost`.
    fn allocate_physical_memory(&mut self, size: usize) -> Result<NonNull<c_void>, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: Calling C FFI function. Assume `cudaMallocHost` is safe if inputs are valid.
        let err = unsafe { cudaMallocHost(&mut ptr, size) };
        if err == cudaError_cudaSuccess { // CUDA_SUCCESS is typically 0
            NonNull::new(ptr).ok_or_else(|| "cudaMallocHost reported success but returned null pointer".to_string())
        } else {
            // In a real app, use cudaGetErrorString to get a descriptive error.
            Err(format!("cudaMallocHost failed with error code {}", err))
        }
    }

    /// Frees physical memory using `cudaFreeHost`.
    fn free_physical_memory(&mut self, ptr: NonNull<c_void>) -> Result<(), String> {
        // SAFETY: Calling C FFI function. Assumes `ptr` is valid and was allocated by `cudaMallocHost`.
        let err = unsafe { cudaFreeHost(ptr.as_ptr()) };
        if err == 0 { // CUDA_SUCCESS is typically 0
            Ok(())
        } else {
            Err(format!("cudaFreeHost failed with error code {}", err))
        }
    }

    /// Splits a free slab at `parent_log_factor` into two children at `parent_log_factor - 1`.
    /// Returns indices of the two new child slabs.
    fn split_slab(&mut self, parent_log_factor: u32) -> Result<(usize, usize), String> {
        if parent_log_factor == 0 {
            return Err("Cannot split slab of log_factor 0".to_string());
        }
        if parent_log_factor > self.max_log_factor {
             return Err(format!("Cannot split from log_factor {} which is > max_log_factor {}", parent_log_factor, self.max_log_factor));
        }

        // `ensure_free_slab_exists` should have been called before to guarantee a splittable slab.
        let parent_slab_idx = self.get_first_free_slab_in_layer(parent_log_factor)
            .ok_or_else(|| format!("No free slab to split at log_factor {}", parent_log_factor))?;
        
        self.slabs[parent_slab_idx].free = false;
        self.layer_remove_free(parent_log_factor, parent_slab_idx);

        let child_log_factor = parent_log_factor - 1;
        let lchild_idx = self.obtain_slab_info_index(child_log_factor, Some(parent_slab_idx));
        let rchild_idx = self.obtain_slab_info_index(child_log_factor, Some(parent_slab_idx));

        // Parent slab's memory is now conceptually divided for children.
        // The parent `SlabInfo` retains its original `location` if it had one.
        // Children point to sub-regions if the parent was `InMemory`.
        let parent_location_opt = self.slabs[parent_slab_idx].location.as_ref().map(|loc| match loc {
            SlabLocation::InMemory(ptr) => Some(*ptr),
            SlabLocation::SwappedOut { .. } => None, // Or handle appropriately
        });

        if let Some(Some(parent_ptr)) = parent_location_opt {
            self.slabs[lchild_idx].location = Some(SlabLocation::InMemory(parent_ptr));
            
            let child_actual_size = self.base_size * (1 << child_log_factor);
            // SAFETY: Pointer arithmetic, assumes parent_ptr is valid and large enough.
            let rchild_ptr_val = unsafe { parent_ptr.as_ptr().add(child_actual_size) };
            let rchild_ptr = NonNull::new(rchild_ptr_val)
                .ok_or_else(|| "Failed to calculate rchild pointer due to null".to_string())?;
            self.slabs[rchild_idx].location = Some(SlabLocation::InMemory(rchild_ptr));
        } else if self.slabs[parent_slab_idx].location.is_some() && parent_location_opt.is_none() {
             // This means it was SwappedOut and we decided to treat it as an error for now
            return Err("Splitting swapped-out slabs not yet implemented".to_string());
        }
        else {
            // Parent slab has no memory location.
            return Err(format!("Parent slab {} at log_factor {} has no memory location to split", parent_slab_idx, parent_log_factor));
        }

        self.slabs[parent_slab_idx].lchild = Some(lchild_idx);
        self.slabs[parent_slab_idx].rchild = Some(rchild_idx);

        self.slabs[lchild_idx].free = true;
        self.slabs[rchild_idx].free = true;

        self.layer_insert_all(child_log_factor, lchild_idx);
        self.layer_insert_all(child_log_factor, rchild_idx);
        self.layer_insert_free(child_log_factor, lchild_idx);
        self.layer_insert_free(child_log_factor, rchild_idx);
        
        Ok((lchild_idx, rchild_idx))
    }

    /// Ensures a free slab exists at the given `log_factor`.
    /// If not, it tries to create one by splitting from a higher layer or allocating new physical memory.
    /// Returns the index of a free slab at this `log_factor`.
    fn ensure_free_slab_exists(&mut self, log_factor: u32) -> Result<usize, String> {
        if let Some(free_slab_idx) = self.get_first_free_slab_in_layer(log_factor) {
            return Ok(free_slab_idx);
        }

        if log_factor == self.max_log_factor {
            let slab_actual_size = self.base_size * (1 << log_factor);
            let ptr = self.allocate_physical_memory(slab_actual_size)?; // Use actual size for this specific slab
            let new_slab_idx = self.obtain_slab_info_index(log_factor, None);
            
            self.slabs[new_slab_idx].location = Some(SlabLocation::InMemory(ptr));
            self.slabs[new_slab_idx].free = true;
            
            self.layer_insert_all(log_factor, new_slab_idx);
            self.layer_insert_free(log_factor, new_slab_idx);
            
            Ok(new_slab_idx)
        } else {
            self.ensure_free_slab_exists(log_factor + 1)?; 
            let (lchild_idx, _rchild_idx) = self.split_slab(log_factor + 1)?;
            Ok(lchild_idx) 
        }
    }

    /// Allocates a slab of memory corresponding to `log_factor`.
    pub fn allocate(&mut self, log_factor: u32) -> Result<NonNull<c_void>, String> {
        if log_factor > self.max_log_factor {
            return Err(format!("Requested log_factor {} exceeds max_log_factor {}", log_factor, self.max_log_factor));
        }

        let slab_idx = self.ensure_free_slab_exists(log_factor)?;
        
        self.slabs[slab_idx].free = false;
        self.layer_remove_free(log_factor, slab_idx);

        let ptr = match &self.slabs[slab_idx].location {
            Some(SlabLocation::InMemory(p)) => *p,
            Some(SlabLocation::SwappedOut { .. }) => {
                // TODO: Implement swap-in logic.
                // self.swap_in(slab_idx)?;
                // return self.slabs[slab_idx].location.as_ref().unwrap().as_in_memory_ptr().unwrap();
                return Err("Allocation of swapped-out slab requires swap-in, not yet implemented".to_string());
            }
            None => {
                 return Err(format!("Slab {} (log_factor {}) selected for allocation has no memory location", slab_idx, log_factor));
            }
        };
        
        self.active_allocations.insert(ptr, slab_idx);
        Ok(ptr)
    }

    /// Tries to merge a `parent_slab` with its children if both children are free.
    fn try_merge(&mut self, parent_slab_idx: usize) {
        let (lchild_idx_opt, rchild_idx_opt, parent_log_factor, grandparent_idx_opt) = {
            let parent_slab = &self.slabs[parent_slab_idx];
            if parent_slab.lchild.is_none() || parent_slab.rchild.is_none() { // Already merged or not split
                return;
            }
            (parent_slab.lchild, parent_slab.rchild, parent_slab.log_factor, parent_slab.parent)
        };

        if let (Some(lchild_idx), Some(rchild_idx)) = (lchild_idx_opt, rchild_idx_opt) {
            if self.slabs[lchild_idx].free && self.slabs[rchild_idx].free {
                let child_log_factor = self.slabs[lchild_idx].log_factor;

                self.layer_remove_free(child_log_factor, lchild_idx);
                self.layer_remove_all(child_log_factor, lchild_idx);
                
                self.layer_remove_free(child_log_factor, rchild_idx);
                self.layer_remove_all(child_log_factor, rchild_idx);

                self.release_slab_info_index(lchild_idx);
                self.release_slab_info_index(rchild_idx);

                let parent = &mut self.slabs[parent_slab_idx];
                parent.lchild = None;
                parent.rchild = None;
                parent.free = true;
                self.layer_insert_free(parent_log_factor, parent_slab_idx);

                if let Some(grandparent_idx) = grandparent_idx_opt {
                     // Condition from C++: if (slab->log_factor < max_log_factor)
                     // This is implicitly handled if grandparent_idx exists, as max_log_factor slabs have no parent.
                    self.try_merge(grandparent_idx);
                }
            }
        }
    }

    /// Deallocates a previously allocated slab of memory.
    pub fn deallocate(&mut self, ptr: NonNull<c_void>) -> Result<(), String> {
        let slab_idx = self.active_allocations.remove(&ptr)
            .ok_or_else(|| "Pointer not found in active allocations or already deallocated".to_string())?;

        let (log_factor, parent_idx_opt) = {
            let slab = &mut self.slabs[slab_idx];
            slab.free = true;
            (slab.log_factor, slab.parent)
        };
        
        self.layer_insert_free(log_factor, slab_idx);

        if let Some(parent_idx) = parent_idx_opt {
            // Only try to merge if it's not a top-level slab from initial allocation (max_log_factor with no parent)
            // and it actually has a parent (was created by a split).
             if log_factor < self.max_log_factor { // Redundant if parent_idx implies it's not a root max_log_factor slab
                self.try_merge(parent_idx);
            }
        }
        Ok(())
    }
    
    /// Clears the memory pool, freeing all allocated physical memory.
    pub fn clear(&mut self) -> Result<(), String> {
        let mut errors = Vec::new();
        
        // Iterate over all slabs. If a slab is a max_log_factor slab and has memory, free it.
        // This is simpler than C++'s list traversal for clear, as we only care about top-level physical allocations.
        for i in 0..self.slabs.len() {
            if self.slabs[i].log_factor == self.max_log_factor {
                 if let Some(SlabLocation::InMemory(ptr)) = self.slabs[i].location.take() { // take to avoid borrow issues
                    if let Err(e) = self.free_physical_memory(ptr) {
                        errors.push(format!("Clear: Failed to free memory for slab {}: {}", i, e));
                        // Potentially put it back if needed: self.slabs[i].location = Some(SlabLocation::InMemory(ptr));
                    }
                }
            }
        }

        self.slabs.clear();
        self.free_slab_info_indices.clear();
        self.slab_layers.iter_mut().for_each(|layer| *layer = SlabLayerInfo::default());
        self.active_allocations.clear();

        if errors.is_empty() { Ok(()) } else { Err(errors.join("; ")) }
    }

    /// Tries to shrink the memory pool by freeing unused top-level slabs.
    pub fn shrink(&mut self) -> Result<(), String> {
        let mut errors = Vec::new();
        let mut candidate_slabs_for_freeing: Vec<(usize, NonNull<c_void>)> = Vec::new();

        // Phase 1: Identify candidate slabs for freeing. (Read-only for self.slabs contents)
        if let Some(layer_info) = self.slab_layers.get(self.max_log_factor as usize) {
            let mut current_idx_opt = layer_info.head_all_idx;
            let mut visited_indices = std::collections::HashSet::new();

            while let Some(current_idx) = current_idx_opt {
                if !visited_indices.insert(current_idx) {
                    errors.push(format!("Shrink (Phase 1): Loop detected in slab list at index {}.", current_idx));
                    break;
                }
                // Ensure current_idx is valid before accessing self.slabs
                if current_idx >= self.slabs.len() {
                    errors.push(format!("Shrink (Phase 1): Invalid slab index {} encountered.", current_idx));
                    break;
                }

                let slab = &self.slabs[current_idx]; // Immutable borrow for reading
                let next_idx_opt = slab.next_in_layer;

                if slab.free && slab.log_factor == self.max_log_factor {
                    if let Some(SlabLocation::InMemory(ptr)) = slab.location {
                        candidate_slabs_for_freeing.push((current_idx, ptr));
                    }
                }
                
                if next_idx_opt == layer_info.head_all_idx { break; }
                current_idx_opt = next_idx_opt;
            }
        }

        let mut successfully_freed_indices: Vec<usize> = Vec::new();
        // Phase 2: Attempt to free memory. (self.free_physical_memory takes &mut self)
        for (slab_idx, ptr_to_free) in candidate_slabs_for_freeing {
            // Check if the slab's current location still matches ptr_to_free before attempting to free.
            // This is a safeguard, though in a single-threaded context it should match if Phase 1 was correct.
            // However, direct access to self.slabs[slab_idx].location here would re-introduce borrow issues
            // if not careful. The key is that free_physical_memory only needs `ptr_to_free`.
            if let Err(e) = self.free_physical_memory(ptr_to_free) {
                errors.push(format!("Shrink (Phase 2): Failed to free memory for slab {}: {}", slab_idx, e));
                // If freeing failed, the original location in self.slabs[slab_idx] remains untouched for now.
                // No need to "put it back" as we haven't `take()`-n it yet.
            } else {
                successfully_freed_indices.push(slab_idx);
            }
        }

        let mut to_remove_from_lists_final = Vec::new();
        let mut to_release_info_final = Vec::new();

        // Phase 3: Update metadata for successfully freed slabs.
        for slab_idx in successfully_freed_indices {
             // Ensure slab_idx is valid before mutable access
            if slab_idx < self.slabs.len() {
                self.slabs[slab_idx].location = None; // Mark as freed by removing its location
                // Add to lists for final cleanup only if it was successfully processed
                to_remove_from_lists_final.push(slab_idx);
                to_release_info_final.push(slab_idx);
            } else {
                 errors.push(format!("Shrink (Phase 3): Invalid slab index {} during metadata update.", slab_idx));
            }
        }
        
        // Phase 4: Update linked lists and release SlabInfo indices.
        for slab_idx in &to_remove_from_lists_final { // Iterate by reference
            if *slab_idx < self.slabs.len() && self.slabs[*slab_idx].free {
                 self.layer_remove_free(self.max_log_factor, *slab_idx);
            } else if *slab_idx >= self.slabs.len() {
                 errors.push(format!("Shrink (Phase 4): Invalid slab index {} during free list removal.", slab_idx));
                 continue;
            }
            self.layer_remove_all(self.max_log_factor, *slab_idx);
        }
        for slab_idx in to_release_info_final { // Consumes the vec
            if slab_idx < self.slabs.len() { // Check before releasing, though it should be fine if added correctly
                self.release_slab_info_index(slab_idx);
            } else {
                 errors.push(format!("Shrink (Phase 4): Invalid slab index {} during info release.", slab_idx));
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors.join("; ")) }
    }
}