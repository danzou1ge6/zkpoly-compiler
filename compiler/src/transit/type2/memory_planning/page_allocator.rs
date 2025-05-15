// this file implements the Belady Optimal page replacement algorithm

use std::collections::{BTreeMap, BTreeSet};

use zkpoly_common::{define_usize_id, heap::IdAllocator, segment_tree::SegmentTree};

use crate::utils::div_ceil;

define_usize_id!(PageId);

static PAGE_ALLIGN: usize = 2 * 1024 * 1024; // 2MB, as requried by cuda

#[derive(Clone, Copy, Debug)]
pub enum PageLocation {
    InMemory(usize), // the index of the page in the page table
    Out,
}


pub struct BeladyAllocator {
    pub page_size: usize,
    pub page_table: Vec<Option<PageId>>,
    pub page_id_allocator: IdAllocator<PageId>,
    pub empty_pages: BTreeSet<usize>,
    pub total_page_num: usize,
    pub page_location: BTreeMap<PageId, PageLocation>,
    pub next_use: SegmentTree<u64>,
}

impl BeladyAllocator {
    pub fn new(page_size: usize, total_page_num: usize) -> Self {
        assert!(page_size % PAGE_ALLIGN == 0);
        Self {
            page_size,
            page_table: vec![None; total_page_num],
            page_id_allocator: IdAllocator::new(),
            empty_pages: (0..total_page_num).collect(),
            total_page_num,
            page_location: BTreeMap::new(),
            next_use: SegmentTree::new(&vec![0; total_page_num]), // all pages are not used, so the next use time is 0
        }
    }

    fn evict(&mut self) -> PageId {
        // find the page that will be used the farthest in the future
        let max_next_use = self.next_use.query_max_info(0, self.total_page_num - 1).unwrap(); // [l, r]
        let index = max_next_use.index;
        
        // check the index is valid
        assert!(self.page_table[index].is_some(), "the found page is already empty");

        // update the segment tree
        self.next_use.modify_set(index, index, 0); // set the next use time to 0

        // evict the page
        let page_id = self.page_table[index].unwrap();
        self.page_table[index] = None;
        self.empty_pages.insert(index);
        *self.page_location.get_mut(&page_id).unwrap() = PageLocation::Out;

        page_id
    }

    pub fn allocate(&mut self, size: usize, next_use: u64) -> (Vec<PageId>, Vec<PageId>) {
        // allocate pages for the given size
        // return: (allocated pages, evicted pages)
        let page_num = div_ceil(size, self.page_size);
        assert!(page_num <= self.total_page_num, "not enough pages");
        let mut allocated_pages = Vec::new();
        let mut evicted_pages = Vec::new();
        
        // check if there are not enough empty pages
        if self.empty_pages.len() < page_num {
            // find the pages to evict
            let evict_num = page_num - self.empty_pages.len();
            for _ in 0..evict_num {
                evicted_pages.push(self.evict());
            }
        }

        assert!(self.empty_pages.len() >= page_num, "not enough pages after evicting");
        // allocate pages
        for _ in 0..page_num {
            let page_id = self.page_id_allocator.alloc();
            let page_index = self.empty_pages.iter().next().unwrap().clone();
            self.empty_pages.remove(&page_index);
            self.page_table[page_index] = Some(page_id);
            allocated_pages.push(page_id);
            self.page_location.insert(page_id, PageLocation::InMemory(page_index));
            // update the segment tree
            self.next_use.modify_set(page_index, page_index, next_use);
        }
        
        (allocated_pages, evicted_pages)
    }

    pub fn deallocate(&mut self, page_id: PageId) {
        // deallocate the page
        let page_location = self.page_location.remove(&page_id).unwrap();
        match page_location {
            PageLocation::InMemory(page_index) => {
                self.page_table[page_index] = None;
                self.empty_pages.insert(page_index);
                self.next_use.modify_set(page_index, page_index, 0); // set the next use time to 0
            }
            PageLocation::Out => {}
        }
    }

    pub fn get_page_location(&self, page_id: PageId) -> Option<PageLocation> {
        self.page_location.get(&page_id).cloned()
    }
}