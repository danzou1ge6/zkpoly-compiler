// this file implements the Belady Optimal page replacement algorithm

use std::collections::{BTreeMap, BTreeSet};

use super::{define_usize_id, heap::IdAllocator, segment_tree::SegmentTree};

define_usize_id!(PageId);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageLocation {
    InMemory(usize), // the index of the page in the page table
    Out,
}

impl PageLocation {
    pub fn unwrap_in_memory(&self) -> usize {
        match self {
            PageLocation::InMemory(idx) => *idx,
            PageLocation::Out => panic!("page is not in memory"),
        }
    }
}

pub struct BeladyAllocator {
    pub page_size: usize,
    pub page_table: Vec<Option<PageId>>,
    pub page_id_allocator: IdAllocator<PageId>,
    pub empty_pages: BTreeSet<usize>,
    pub total_page_num: usize,
    pub page_location: BTreeMap<PageId, PageLocation>,
    pub next_use: SegmentTree<u64>,
    pub max_used_pages: usize,
}

pub struct SpillPageManger {
    pub empty_spill_pos: BTreeSet<usize>,
    pub pageid2spill_pos: BTreeMap<PageId, usize>,
    pub max_spill_pos: usize,
}

impl SpillPageManger {
    pub fn new() -> Self {
        Self {
            empty_spill_pos: BTreeSet::new(),
            pageid2spill_pos: BTreeMap::new(),
            max_spill_pos: 0,
        }
    }

    pub fn allocate(&mut self, page_id: PageId) -> usize {
        let pos = if let Some(pos) = self.empty_spill_pos.iter().next().cloned() {
            self.empty_spill_pos.remove(&pos);
            pos
        } else {
            self.max_spill_pos += 1;
            self.max_spill_pos - 1
        };
        self.pageid2spill_pos.insert(page_id, pos);
        pos
    }

    pub fn deallocate(&mut self, page_id: PageId) {
        if let Some(pos) = self.pageid2spill_pos.remove(&page_id) {
            self.empty_spill_pos.insert(pos);
        }
    }

    pub fn get_spill_pos(&self, page_id: PageId) -> Option<usize> {
        self.pageid2spill_pos.get(&page_id).cloned()
    }
}

impl BeladyAllocator {
    pub fn new(page_size: usize, total_page_num: usize) -> Self {
        Self {
            page_size,
            page_table: vec![None; total_page_num],
            page_id_allocator: IdAllocator::new(),
            empty_pages: (0..total_page_num).collect(),
            total_page_num,
            page_location: BTreeMap::new(),
            next_use: SegmentTree::new(&vec![0; total_page_num]), // all pages are not used, so the next use time is 0
            max_used_pages: 0,
        }
    }

    fn evict(&mut self) -> (PageId, usize) {
        // find the page that will be used the farthest in the future
        let max_next_use = self
            .next_use
            .query_max_info(0, self.total_page_num - 1)
            .unwrap(); // [l, r]
        let index = max_next_use.index;
        assert!(max_next_use.value > 0, "evicting a page with next use 0 (likely to be a page just restored), probably because the needed pages are more total pages");

        // check the index is valid
        assert!(
            self.page_table[index].is_some(),
            "the found page is already empty"
        );

        // update the segment tree
        self.next_use.modify_set(index, index, 0); // set the next use time to 0

        // evict the page
        let page_id = self.page_table[index].unwrap();
        self.page_table[index] = None;
        self.empty_pages.insert(index);
        *self.page_location.get_mut(&page_id).unwrap() = PageLocation::Out;

        (page_id, index)
    }

    pub fn allocate(&mut self, size: usize, next_use: u64) -> (Vec<PageId>, Vec<(PageId, usize)>) {
        // allocate pages for the given size
        // return: (allocated pages, evicted pages)
        let page_num = size.div_ceil(self.page_size);
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

        assert!(
            self.empty_pages.len() >= page_num,
            "not enough pages after evicting"
        );
        // allocate pages
        for _ in 0..page_num {
            let page_id = self.page_id_allocator.alloc();
            let page_index = self.empty_pages.iter().next().unwrap().clone();
            self.empty_pages.remove(&page_index);
            self.page_table[page_index] = Some(page_id);
            allocated_pages.push(page_id);
            self.page_location
                .insert(page_id, PageLocation::InMemory(page_index));
            // update the segment tree
            self.next_use.modify_set(page_index, page_index, next_use);
        }

        self.max_used_pages = self
            .max_used_pages
            .max(self.total_page_num - self.empty_pages.len());
        (allocated_pages, evicted_pages)
    }

    pub fn restore(&mut self, page_id: PageId) -> (usize, Option<(PageId, usize)>) {
        let next_use = 0; // we don't want to evict this page untill its next use is updated
        
        // restore the page to the page table if moved out
        if let PageLocation::Out = self.page_location[&page_id] {
            let mut evict_page = None;
            // check if there are not enough empty pages
            if self.empty_pages.len() == 0 {
                // find the pages to evict
                evict_page = Some(self.evict());
            }

            assert!(
                self.empty_pages.len() > 0,
                "not enough pages after evicting"
            );
            // allocate pages
            let page_index = self.empty_pages.iter().next().unwrap().clone();
            self.empty_pages.remove(&page_index);
            self.page_table[page_index] = Some(page_id);
            self.page_location
                .insert(page_id, PageLocation::InMemory(page_index));
            // update the segment tree
            self.next_use.modify_set(page_index, page_index, next_use);

            self.max_used_pages = self
                .max_used_pages
                .max(self.total_page_num - self.empty_pages.len());
            (page_index, evict_page)
        } else {
            unreachable!("page is already in memory")
        }
    }

    pub fn get_max_used_pages(&self) -> usize {
        self.max_used_pages
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

    pub fn update_next_use(&mut self, page_id: PageId, next_use: u64) {
        // update the next use time of the page
        let page_location = self.page_location.get_mut(&page_id).unwrap();
        match page_location {
            PageLocation::InMemory(page_index) => {
                self.next_use.modify_set(*page_index, *page_index, next_use);
            }
            PageLocation::Out => {}
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_SIZE: usize = 2 * 1024 * 1024;

    #[test]
    fn test_new_allocator() {
        let total_page_num = 10;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);
        assert_eq!(allocator.page_size, PAGE_SIZE);
        assert_eq!(allocator.total_page_num, total_page_num);
        assert_eq!(allocator.page_table.len(), total_page_num);
        assert_eq!(
            allocator.page_table.iter().filter(|p| p.is_some()).count(),
            0
        );
        assert_eq!(allocator.empty_pages.len(), total_page_num);
        assert_eq!(allocator.page_location.len(), 0);
        // Check segment tree initialization, assuming 0 for unused pages
        for i in 0..total_page_num {
            assert_eq!(allocator.next_use.query_max_info(i, i).unwrap().index, i);
            assert_eq!(allocator.next_use.query_max_info(i, i).unwrap().value, 0);
        }
        assert_eq!(
            allocator
                .next_use
                .query_max_info(0, total_page_num - 1)
                .unwrap()
                .value,
            0
        );
    }

    #[test]
    fn test_allocate_simple() {
        let total_page_num = 10;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);

        let (allocated, evicted) = allocator.allocate(PAGE_SIZE * 2, 100);
        assert_eq!(allocated.len(), 2);
        assert_eq!(evicted.len(), 0);
        assert_eq!(allocator.empty_pages.len(), total_page_num - 2);
        assert_eq!(allocator.page_location.len(), 2);

        for page_id in allocated.iter() {
            match allocator.get_page_location(*page_id) {
                Some(PageLocation::InMemory(idx)) => {
                    assert_eq!(allocator.page_table[idx], Some(*page_id));
                    assert_eq!(
                        allocator.next_use.query_max_info(idx, idx).unwrap().value,
                        100
                    );
                }
                _ => panic!("Page should be in memory"),
            }
        }
    }

    #[test]
    fn test_allocate_with_eviction() {
        let total_page_num = 2;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);

        // Allocate all pages
        let (allocated1, evicted1) = allocator.allocate(PAGE_SIZE, 10);
        assert_eq!(allocated1.len(), 1);
        assert_eq!(evicted1.len(), 0);
        let page_id1 = allocated1[0];

        let (allocated2, evicted2) = allocator.allocate(PAGE_SIZE, 20);
        assert_eq!(allocated2.len(), 1);
        assert_eq!(evicted2.len(), 0);
        let page_id2 = allocated2[0];

        assert_eq!(allocator.empty_pages.len(), 0);

        // Allocate one more page, should trigger eviction
        // page_id1 should be evicted as its next_use (10) is smaller than page_id2's (20)
        // However, Belady evicts the one with largest next_use.
        // Let's re-evaluate. The one with the *largest* next_use time is evicted.
        // So page_id2 (next_use 20) should be kept, page_id1 (next_use 10) should be kept.
        // The segment tree stores next_use, evict picks max next_use.
        // If we allocate with next_use 5, it should evict the one with next_use 20.

        // Let's set next_use for page_id1 to 100 and page_id2 to 200
        // First, find their in-memory indices
        let idx1 = match allocator.get_page_location(page_id1).unwrap() {
            PageLocation::InMemory(i) => i,
            _ => panic!(),
        };
        let idx2 = match allocator.get_page_location(page_id2).unwrap() {
            PageLocation::InMemory(i) => i,
            _ => panic!(),
        };
        allocator.next_use.modify_set(idx1, idx1, 100);
        allocator.next_use.modify_set(idx2, idx2, 200);

        let (allocated3, evicted3) = allocator.allocate(PAGE_SIZE, 50); // New page has next_use 50
        assert_eq!(allocated3.len(), 1);
        assert_eq!(evicted3.len(), 1);
        let page_id3 = allocated3[0];

        // page_id2 (next_use 200) should be evicted
        assert_eq!(evicted3[0].0, page_id2);
        assert_eq!(
            allocator.get_page_location(page_id2),
            Some(PageLocation::Out)
        );
        assert!(allocator.get_page_location(page_id1).is_some());
        assert!(matches!(
            allocator.get_page_location(page_id1).unwrap(),
            PageLocation::InMemory(_)
        ));
        assert!(allocator.get_page_location(page_id3).is_some());
        assert!(matches!(
            allocator.get_page_location(page_id3).unwrap(),
            PageLocation::InMemory(_)
        ));

        let new_page_idx = match allocator.get_page_location(page_id3).unwrap() {
            PageLocation::InMemory(i) => i,
            _ => panic!(),
        };
        assert_eq!(
            allocator
                .next_use
                .query_max_info(new_page_idx, new_page_idx)
                .unwrap()
                .value,
            50
        );

        // The evicted page's old slot should now have next_use 50 (from page_id3)
        // and the other slot (page_id1) should have next_use 100.
        // The slot that held page_id2 (idx2) should now hold page_id3.
        assert_eq!(allocator.page_table[idx2], Some(page_id3));
        assert_eq!(
            allocator.next_use.query_max_info(idx2, idx2).unwrap().value,
            50
        );
        assert_eq!(allocator.page_table[idx1], Some(page_id1));
        assert_eq!(
            allocator.next_use.query_max_info(idx1, idx1).unwrap().value,
            100
        );
    }

    #[test]
    fn test_deallocate_in_memory() {
        let total_page_num = 5;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);
        let (allocated, _) = allocator.allocate(PAGE_SIZE, 100);
        let page_id = allocated[0];

        let page_idx = match allocator.get_page_location(page_id).unwrap() {
            PageLocation::InMemory(idx) => idx,
            _ => panic!("Page should be in memory"),
        };

        allocator.deallocate(page_id);

        assert_eq!(allocator.get_page_location(page_id), None);
        assert!(allocator.page_table[page_idx].is_none());
        assert!(allocator.empty_pages.contains(&page_idx));
        assert_eq!(
            allocator
                .next_use
                .query_max_info(page_idx, page_idx)
                .unwrap()
                .value,
            0
        );
        assert_eq!(allocator.empty_pages.len(), total_page_num);
    }

    #[test]
    fn test_deallocate_out_of_memory() {
        let total_page_num = 1;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);
        let (allocated1, _) = allocator.allocate(PAGE_SIZE, 100);
        let page_id1 = allocated1[0];

        // Evict page_id1
        let (_, evicted) = allocator.allocate(PAGE_SIZE, 200); // This will allocate a new page and evict page_id1
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, page_id1);
        assert_eq!(
            allocator.get_page_location(page_id1),
            Some(PageLocation::Out)
        );

        allocator.deallocate(page_id1);
        assert_eq!(allocator.get_page_location(page_id1), None);
        // empty_pages should still reflect the one page that is in memory, not the one just deallocated from "Out"
        assert_eq!(allocator.empty_pages.len(), 0);
        assert_eq!(allocator.page_location.len(), 1); // The second page is still there
    }

    #[test]
    fn test_evict_logic() {
        let total_page_num = 3;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);

        let (p1_vec, _) = allocator.allocate(PAGE_SIZE, 10);
        let p1 = p1_vec[0];
        let _ = match allocator.get_page_location(p1).unwrap() {
            PageLocation::InMemory(i) => i,
            _ => panic!(),
        };

        let (p2_vec, _) = allocator.allocate(PAGE_SIZE, 30);
        let p2 = p2_vec[0];
        let idx2 = match allocator.get_page_location(p2).unwrap() {
            PageLocation::InMemory(i) => i,
            _ => panic!(),
        };

        let (p3_vec, _) = allocator.allocate(PAGE_SIZE, 20);
        let p3 = p3_vec[0];
        let idx3 = match allocator.get_page_location(p3).unwrap() {
            PageLocation::InMemory(i) => i,
            _ => panic!(),
        };

        // p2 has the largest next_use (30), so it should be evicted.
        let evicted_page_id = allocator.evict();
        assert_eq!(evicted_page_id.0, p2);
        assert_eq!(allocator.get_page_location(p2), Some(PageLocation::Out));
        assert!(allocator.page_table[idx2].is_none());
        assert!(allocator.empty_pages.contains(&idx2));
        assert_eq!(
            allocator.next_use.query_max_info(idx2, idx2).unwrap().value,
            0
        );

        assert_eq!(allocator.page_location.len(), 3); // p1, p3 in memory, p2 out
        assert_eq!(allocator.empty_pages.len(), 1);

        // Next, p3 (next_use 20) should be evicted.
        let evicted_page_id_2 = allocator.evict();
        assert_eq!(evicted_page_id_2.0, p3);
        assert_eq!(allocator.get_page_location(p3), Some(PageLocation::Out));
        assert!(allocator.page_table[idx3].is_none());
        assert!(allocator.empty_pages.contains(&idx3));
        assert_eq!(
            allocator.next_use.query_max_info(idx3, idx3).unwrap().value,
            0
        );

        assert_eq!(allocator.empty_pages.len(), 2);
    }

    #[test]
    fn test_get_page_location() {
        let total_page_num = 2;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);

        let (allocated1, _) = allocator.allocate(PAGE_SIZE, 10);
        let page_id1 = allocated1[0];
        let idx1 = match allocator.get_page_location(page_id1).unwrap() {
            PageLocation::InMemory(i) => i,
            _ => panic!(),
        };

        assert!(
            matches!(allocator.get_page_location(page_id1), Some(PageLocation::InMemory(i)) if i == idx1)
        );

        let (allocated2, evicted) = allocator.allocate(PAGE_SIZE * 2, 20); // Needs 2 pages, has 1 empty. Evicts page_id1
        let page_id2 = allocated2[0]; // First of the two new pages
        let page_id3 = allocated2[1]; // Second of the two new pages

        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, page_id1);

        assert_eq!(
            allocator.get_page_location(page_id1),
            Some(PageLocation::Out)
        );
        assert!(matches!(
            allocator.get_page_location(page_id2),
            Some(PageLocation::InMemory(_))
        ));
        assert!(matches!(
            allocator.get_page_location(page_id3),
            Some(PageLocation::InMemory(_))
        ));

        let non_existent_page_id = PageId::from(999);
        assert_eq!(allocator.get_page_location(non_existent_page_id), None);
    }

    #[test]
    #[should_panic(expected = "not enough pages")]
    fn test_allocate_too_large() {
        let total_page_num = 1;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);
        allocator.allocate(PAGE_SIZE * 2, 100); // Request 2 pages when only 1 is available
    }

    #[test]
    fn test_allocate_exact_fit_after_eviction() {
        let total_page_num = 2;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);

        // Fill memory
        let (p1_alloc, _) = allocator.allocate(PAGE_SIZE, 100);
        let p1 = p1_alloc[0];
        let (p2_alloc, _) = allocator.allocate(PAGE_SIZE, 200);
        let p2 = p2_alloc[0];

        // p2 will be evicted (next_use 200 is max)
        let (allocated, evicted) = allocator.allocate(PAGE_SIZE, 50);
        assert_eq!(allocated.len(), 1);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, p2);
        assert_eq!(allocator.get_page_location(p2), Some(PageLocation::Out));
        assert!(matches!(
            allocator.get_page_location(p1),
            Some(PageLocation::InMemory(_))
        ));
        assert!(matches!(
            allocator.get_page_location(allocated[0]),
            Some(PageLocation::InMemory(_))
        ));
        assert_eq!(allocator.empty_pages.len(), 0);
    }

    #[test]
    fn test_evict_empty_does_not_panic_if_no_pages_to_evict() {
        // This test is tricky because evict() is private and called by allocate().
        // allocate() has a check: `if self.empty_pages.len() < page_num`
        // If page_num is 0, or if empty_pages >= page_num, evict won't be called.
        // If we want to test evict() directly under a scenario where page_table might be empty
        // (which evict() asserts against: `assert!(self.page_table[index].is_some())`),
        // we'd need to manipulate state in a way not normally possible or make evict public.
        // Given current structure, a direct test for "evicting from fully empty" is hard.
        // The existing assert `assert!(self.page_table[index].is_some(), "the found page is already empty");`
        // should catch incorrect states if evict is called when it shouldn't be.
        //
        // Let's test a scenario where allocate might try to evict more than available.
        // The `assert!(page_num <= self.total_page_num, "not enough pages");` in allocate
        // should prevent requesting more pages than physically exist.
        //
        // Consider a case where all pages are full, and we request to allocate 1 page.
        // This will call evict once.
        let total_page_num = 1;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);
        allocator.allocate(PAGE_SIZE, 10); // Fill the page

        // At this point, page_table[0] is Some, empty_pages is 0.
        // If we allocate again, it will call evict.
        let (allocated, evicted) = allocator.allocate(PAGE_SIZE, 20);
        assert_eq!(allocated.len(), 1);
        assert_eq!(evicted.len(), 1);
        // The assert in evict() should not have triggered.
    }

    #[test]
    fn test_page_id_reuse_after_deallocate_and_reallocate() {
        let total_page_num = 1;
        let mut allocator = BeladyAllocator::new(PAGE_SIZE, total_page_num);

        let (allocated1, _) = allocator.allocate(PAGE_SIZE, 10);
        let page_id1 = allocated1[0];
        assert_eq!(page_id1, PageId::from(0)); // First allocated ID

        allocator.deallocate(page_id1);
        // page_id_allocator does not currently reuse IDs by default design of IdAllocator.
        // This test confirms current behavior. If IdAllocator changes, this might need update.

        let (allocated2, _) = allocator.allocate(PAGE_SIZE, 20);
        let page_id2 = allocated2[0];
        assert_eq!(page_id2, PageId::from(1)); // Next ID
    }
}
