// This file implements a page allocator using CUDA virtual memory APIs.

use zkpoly_common::devices::DeviceType;

use crate::{
    bindings::{
        cuInit, cuMemAddressFree, cuMemAddressReserve, cuMemCreate, cuMemGetAllocationGranularity, cuMemMap, cuMemRelease, cuMemSetAccess, cuMemUnmap, CUdeviceptr, CUmemAccessDesc, CUmemAccess_flags_enum_CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_MINIMUM, CUmemAllocationProp, CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED, CUmemGenericAllocationHandle, CUmemLocation, CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE, CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_HOST_NUMA
    },
    cuda_driver_check,
};

pub struct PageAllocator {
    device: DeviceType,
    page_size: usize,
    page_table: Vec<CUmemGenericAllocationHandle>, // handle for physical memory
}

fn get_location(device: &DeviceType) -> CUmemLocation {
    let mut location: CUmemLocation = unsafe { std::mem::zeroed() };
    match device {
        DeviceType::CPU => {
            location.type_ = CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_HOST_NUMA;
            location.id = 0; // TODO: this should be the NUMA node id
        }
        DeviceType::GPU { device_id } => {
            location.type_ = CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE;
            location.id = *device_id;
        }
        DeviceType::Disk => {
            panic!("Disk allocation is not supported");
        }
    }
    location
}

fn get_access_desc(device: &DeviceType) -> CUmemAccessDesc {
    let mut access_desc: CUmemAccessDesc = unsafe { std::mem::zeroed() };
    access_desc.location = get_location(device);
    access_desc.flags = CUmemAccess_flags_enum_CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access_desc
}

// Helper function to get allocation properties
fn get_allocation_prop(device: &DeviceType) -> CUmemAllocationProp {
    let mut prop: CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location = get_location(device);
    prop
}

impl PageAllocator {
    pub fn new(device: DeviceType, page_size: usize, page_num: usize) -> Self {
        unsafe {
            cuda_driver_check!(cuInit(0)); // Initialize the CUDA driver API
            let prop = get_allocation_prop(&device);
            let mut min_page_size: usize = 0;
            cuda_driver_check!(cuMemGetAllocationGranularity(
                &mut min_page_size,
                &prop,
                CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ));
            assert!(
                page_size % min_page_size == 0,
                "page size must be aligned to minimum page size"
            );

            // now we start to allocate the page table
            let page_table = (0..page_num)
                .into_iter()
                .map(|_| {
                    let mut handle: CUmemGenericAllocationHandle = std::mem::zeroed();
                    cuda_driver_check!(cuMemCreate(&mut handle, page_size, &prop, 0));
                    handle
                })
                .collect();

            Self {
                device,
                page_size,
                page_table,
            }
        }
    }

    pub fn allocate<T: Sized>(&self, va_size: usize, page_ids: Vec<usize>) -> *mut T {
        // allocate a virtual address space, and map the pages to it
        // check the virtual address space size is larger than the page size
        assert!(va_size >= self.page_size * page_ids.len());

        unsafe {
            let mut va: CUdeviceptr = std::mem::zeroed();

            // allocate the virtual address space
            cuda_driver_check!(cuMemAddressReserve(&mut va, va_size, self.page_size, 0, 0));
            let mut offset = 0;
            let access_desc = get_access_desc(&self.device);

            for page_id in page_ids.iter() {
                // map the pages to the virtual address space
                cuda_driver_check!(cuMemMap(
                    va + offset,
                    self.page_size,
                    0, // must be 0
                    self.page_table[*page_id],
                    0 // must be 0
                ));

                // set the access permission
                cuda_driver_check!(cuMemSetAccess(va + offset, self.page_size, &access_desc, 1));

                offset += self.page_size as u64;
            }

            va as *mut T
        }
    }

    pub fn extend<T: Sized>(
        &self,
        va: *mut T,
        va_size: usize,
        allocated_pages: usize,
        page_ids: Vec<usize>,
    ) {
        // extend the virtual address space, and map the pages to it
        // check the virtual address space size is larger than the page size
        assert!(va_size >= self.page_size * (allocated_pages + page_ids.len()));
        unsafe {
            let mut offset = (self.page_size * allocated_pages) as u64;
            let access_desc = get_access_desc(&self.device);

            for page_id in page_ids.iter() {
                // map the pages to the virtual address space
                cuda_driver_check!(cuMemMap(
                    va as u64 + offset,
                    self.page_size,
                    0, // must be 0
                    self.page_table[*page_id],
                    0 // must be 0
                ));

                // set the access permission
                cuda_driver_check!(cuMemSetAccess(
                    va as u64 + offset,
                    self.page_size,
                    &access_desc,
                    1
                ));

                offset += self.page_size as u64;
            }
        }
    }

    pub fn deallocate<T: Sized>(&self, va: *mut T, va_size: usize, allocated_pages: usize) {
        unsafe {
            let mut offset = 0;
            let va = va as u64;
            for _ in 0..allocated_pages {
                // unmap the pages from the virtual address space
                cuda_driver_check!(cuMemUnmap(va + offset, self.page_size));
                offset += self.page_size as u64;
            }
            // release the virtual address space
            cuda_driver_check!(cuMemAddressFree(va, va_size));
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }
}

impl Drop for PageAllocator {
    fn drop(&mut self) {
        unsafe {
            for handle in &self.page_table {
                cuda_driver_check!(cuMemRelease(*handle));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use super::*;

    #[test]
    fn test_allocate_write_read_deallocate() {
        let page_size = 1024 * 1024 * 2; // 2MB
        let allocator = PageAllocator::new(DeviceType::GPU { device_id: 0 }, page_size, 10);

        let num_pages_to_alloc = 2;
        let va_size = page_size * num_pages_to_alloc * 2; // Reserve more VA than needed for the mapping
        let page_ids_to_alloc = vec![0, 1];

        let ptr = allocator.allocate::<u8>(va_size, page_ids_to_alloc.clone());
        assert!(!ptr.is_null());

        println!("Preparing data to write...");
        let data_to_write_size = page_size * num_pages_to_alloc;
        let mut host_data_write = vec![0u8; data_to_write_size];
        for i in 0..data_to_write_size {
            host_data_write[i] = (i % 256) as u8;
        }

        unsafe {
            println!("Writing data to device...");
            crate::cuda_check!(crate::bindings::cudaMemcpy(
                ptr as *mut c_void,
                host_data_write.as_ptr() as *const c_void,
                data_to_write_size,
                crate::bindings::cudaMemcpyKind_cudaMemcpyHostToDevice
            ));

            let mut host_data_read = vec![0u8; data_to_write_size];
            println!("Reading data from device...");
            crate::cuda_check!(crate::bindings::cudaMemcpy(
                host_data_read.as_mut_ptr() as *mut c_void,
                ptr as *const c_void,
                data_to_write_size,
                crate::bindings::cudaMemcpyKind_cudaMemcpyDeviceToHost
            ));
            assert_eq!(host_data_write, host_data_read);
        }
        println!("Deallocating memory...");
        allocator.deallocate(ptr, va_size,  num_pages_to_alloc);
    }

    #[test]
    fn test_extend_memory() {
        let page_size = 1024 * 1024 * 2; // 2MB
        let total_physical_pages = 10;
        let allocator = PageAllocator::new(
            DeviceType::GPU { device_id: 0 },
            page_size,
            total_physical_pages,
        );

        let initial_pages_count = 2;
        let initial_page_ids = vec![0, 1];
        // VA size must be large enough for the sum of initial and extended pages.
        let va_size = page_size * (initial_pages_count + 3 + 2); // e.g., for 7 pages total if extending by 3

        let ptr = allocator.allocate::<u8>(va_size, initial_page_ids.clone());
        assert!(!ptr.is_null());

        let initial_data_size = page_size * initial_pages_count;
        let mut host_data_initial_write = vec![0u8; initial_data_size];
        for i in 0..initial_data_size {
            host_data_initial_write[i] = (i % 128) as u8;
        }

        unsafe {
            crate::cuda_driver_check!(crate::bindings::cudaMemcpy(
                ptr as *mut c_void,
                host_data_initial_write.as_ptr() as *const c_void,
                initial_data_size,
                crate::bindings::cudaMemcpyKind_cudaMemcpyHostToDevice
            ));

            let mut host_data_initial_read = vec![0u8; initial_data_size];
            crate::cuda_driver_check!(crate::bindings::cudaMemcpy(
                host_data_initial_read.as_mut_ptr() as *mut c_void,
                ptr as *const c_void,
                initial_data_size,
                crate::bindings::cudaMemcpyKind_cudaMemcpyDeviceToHost
            ));
            assert_eq!(host_data_initial_write, host_data_initial_read);
        }

        let extend_pages_count = 3;
        let extend_page_ids = vec![2, 3, 4]; // Use different physical pages
        allocator.extend(ptr, va_size, initial_pages_count, extend_page_ids.clone());

        let extended_data_size = page_size * extend_pages_count;
        let mut host_data_extended_write = vec![0u8; extended_data_size];
        for i in 0..extended_data_size {
            host_data_extended_write[i] = ((i + 128) % 256) as u8; // Different data pattern
        }

        unsafe {
            let extended_ptr_offset = ptr.add(initial_data_size);
            crate::cuda_driver_check!(crate::bindings::cudaMemcpy(
                extended_ptr_offset as *mut c_void,
                host_data_extended_write.as_ptr() as *const c_void,
                extended_data_size,
                crate::bindings::cudaMemcpyKind_cudaMemcpyHostToDevice
            ));

            let mut host_data_extended_read = vec![0u8; extended_data_size];
            crate::cuda_driver_check!(crate::bindings::cudaMemcpy(
                host_data_extended_read.as_mut_ptr() as *mut c_void,
                extended_ptr_offset as *const c_void,
                extended_data_size,
                crate::bindings::cudaMemcpyKind_cudaMemcpyDeviceToHost
            ));
            assert_eq!(host_data_extended_write, host_data_extended_read);

            // Verify the entire mapped region
            let total_data_size = initial_data_size + extended_data_size;
            let mut host_data_total_read = vec![0u8; total_data_size];
            crate::cuda_driver_check!(crate::bindings::cudaMemcpy(
                host_data_total_read.as_mut_ptr() as *mut c_void,
                ptr as *const c_void,
                total_data_size,
                crate::bindings::cudaMemcpyKind_cudaMemcpyDeviceToHost
            ));

            let mut expected_total_data = host_data_initial_write.clone();
            expected_total_data.extend_from_slice(&host_data_extended_write);
            assert_eq!(expected_total_data, host_data_total_read);
        }

        allocator.deallocate(ptr, va_size, initial_pages_count + extend_pages_count);
    }

    #[test]
    #[should_panic(expected = "page size must be aligned to minimum page size")]
    fn test_new_with_unaligned_page_size() {
        let mut min_page_size: usize = 0;
        unsafe {
            crate::cuda_driver_check!(crate::bindings::cuInit(0));
            let prop = super::get_allocation_prop(&DeviceType::GPU { device_id: 0 });
            crate::cuda_driver_check!(crate::bindings::cuMemGetAllocationGranularity(
                &mut min_page_size,
                &prop,
                crate::bindings::CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ));
        }
        assert!(
            min_page_size > 0,
            "Minimum page size must be greater than 0 for GPU."
        );

        // Create an unaligned page size, assuming min_page_size > 1 (typical for CUDA)
        let unaligned_page_size = min_page_size + 1;
        if unaligned_page_size % min_page_size == 0 {
            // This would only happen if min_page_size is 1.
            // CUDA virtual memory granularity is expected to be much larger.
            // If this branch is hit, the assumption about CUDA's min_page_size is wrong for this environment.
            panic!("min_page_size is 1 or unaligned logic failed; test needs review for this CUDA environment.");
        }
        // This call should panic due to the alignment assertion in CudaPageAllocator::new
        let _allocator =
            PageAllocator::new(DeviceType::GPU { device_id: 0 }, unaligned_page_size, 10);
    }

    #[test]
    #[should_panic(expected = "assertion failed: va_size >= self.page_size * page_ids.len()")]
    fn test_allocate_insufficient_va_size() {
        let page_size = 1024 * 1024 * 2;
        let allocator = PageAllocator::new(DeviceType::GPU { device_id: 0 }, page_size, 10);
        let page_ids_to_alloc = vec![0, 1]; // Requesting 2 pages (total size: page_size * 2)
        let insufficient_va_size = page_size * 1; // VA reserved for only 1 page
                                                  // This should panic because va_size is not enough for page_ids_to_alloc.len() pages.
        allocator.allocate::<u8>(insufficient_va_size, page_ids_to_alloc);
    }

    #[test]
    #[should_panic(
        expected = "assertion failed: va_size >= self.page_size * (allocated_pages + page_ids.len())"
    )]
    fn test_extend_insufficient_va_size() {
        let page_size = 1024 * 1024 * 2;
        let allocator = PageAllocator::new(DeviceType::GPU { device_id: 0 }, page_size, 10);

        let initial_pages_count = 1;
        let initial_page_ids = vec![0];
        // Reserve VA for only 2 pages initially.
        let va_size_allocated_initially = page_size * 2;

        let ptr = allocator.allocate::<u8>(va_size_allocated_initially, initial_page_ids);
        assert!(!ptr.is_null());

        // Try to extend by 2 more pages. Total required: 1 (initial) + 2 (extend) = 3 pages.
        // The initially allocated VA (for 2 pages) is insufficient for 3 pages.
        let extend_page_ids = vec![1, 2];
        allocator.extend(
            ptr,
            va_size_allocated_initially,
            initial_pages_count,
            extend_page_ids,
        );
    }

    #[test]
    fn test_cpu_page_allocator_allocate_deallocate() {
        let mut min_granularity: usize = 0;
        let device = DeviceType::CPU;
        let prop = super::get_allocation_prop(&device);
        unsafe {
            crate::cuda_driver_check!(crate::bindings::cuInit(0));
            crate::cuda_driver_check!(crate::bindings::cuMemGetAllocationGranularity(
                &mut min_granularity,
                &prop,
                crate::bindings::CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ));
        }
        assert!(min_granularity > 0, "CPU min granularity must be > 0");

        let desired_page_size = 1024 * 64; // 64KB target
                                           // Align desired_page_size to min_granularity, ensuring it's at least min_granularity
        let page_size = if desired_page_size < min_granularity {
            min_granularity
        } else {
            (desired_page_size / min_granularity) * min_granularity
        };
        if page_size == 0 {
            // Should not happen if min_granularity > 0
            panic!(
                "Calculated page_size for CPU is 0. min_granularity: {}",
                min_granularity
            );
        }

        let allocator = PageAllocator::new(device, page_size, 5);

        let num_pages_to_alloc = 2;
        let va_size = page_size * num_pages_to_alloc * 2; // Reserve more VA
        let page_ids_to_alloc = vec![0, 1];

        let ptr = allocator.allocate::<u8>(va_size, page_ids_to_alloc.clone());
        assert!(!ptr.is_null());

        let data_size = page_size * num_pages_to_alloc;
        let mut host_data_write = vec![0u8; data_size];
        for i in 0..data_size {
            host_data_write[i] = (i % 250) as u8;
        }

        unsafe {
            // For CPU pinned memory mapped to VA, we can write/read directly via the pointer
            let slice_mut = std::slice::from_raw_parts_mut(ptr, data_size);
            slice_mut.copy_from_slice(&host_data_write);

            let slice_read = std::slice::from_raw_parts(ptr as *const u8, data_size);
            assert_eq!(host_data_write, slice_read);
        }

        allocator.deallocate(ptr, va_size, num_pages_to_alloc);
    }
}
