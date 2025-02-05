use super::{Addr, AddrId, AddrMappingHandler, Instant, IntegralSize, Size};
use std::collections::BTreeMap;
use zkpoly_common::mm_heap::MmHeap;

pub struct Transfer {
    pub from: AddrId,
    pub to: AddrId,
}

#[derive(Clone, Debug)]
enum BlockStatus {
    /// The block is free. `.0` holds the last die_at of the block.
    /// A block becomes free after the die_at instant.
    Free(usize),
    /// The block is splitted. `.0` holds the live_at of the block, `.1` holds the next use of children.
    /// This next_use information is for deciding which block to eject.
    /// A block becomes nonfree before the live_at of the block.
    Splitted(AddrId, usize, MmHeap<u64, usize>),
    /// The block is occupied. `.0` holds the live_at of the block, `.1` holds the next_use of the block.
    Occupied(AddrId, usize, usize),
}

impl BlockStatus {
    pub fn unwrap_splitted(&self) -> &MmHeap<u64, usize> {
        match self {
            BlockStatus::Free(..) => panic!("free block does not have splitted"),
            BlockStatus::Splitted(_, _, heap) => heap,
            BlockStatus::Occupied(..) => panic!("occupied block does not have splitted"),
        }
    }

    pub fn unwrap_splitted_mut(&mut self) -> &mut MmHeap<u64, usize> {
        match self {
            BlockStatus::Free(..) => panic!("free block does not have splitted"),
            BlockStatus::Splitted(_, _, heap) => heap,
            BlockStatus::Occupied(..) => panic!("occupied block does not have splitted"),
        }
    }

    pub fn nonfree_child_addrs<'a>(&'a self) -> impl Iterator<Item = u64> + 'a {
        match self {
            BlockStatus::Splitted(_, _, heap) => heap.keys().copied(),
            _ => panic!("only splitted block has children"),
        }
    }

    pub fn free(&self) -> bool {
        match self {
            BlockStatus::Free(..) => true,
            _ => false,
        }
    }

    pub fn live_at(&self) -> Option<usize> {
        match self {
            BlockStatus::Free(..) => None,
            BlockStatus::Splitted(live_at, ..) | BlockStatus::Occupied(live_at, ..) => {
                Some(*live_at)
            }
        }
    }

    pub fn last_die_at(&self) -> Option<usize> {
        match self {
            BlockStatus::Free(x) => Some(*x),
            _ => None,
        }
    }

    pub fn next_use(&self) -> usize {
        match self {
            BlockStatus::Free(..) => panic!("free block does not have next_use"),
            BlockStatus::Splitted(_, _, children_next_use) => children_next_use.max().unwrap().0,
            BlockStatus::Occupied(_, _, next_use) => *next_use,
        }
    }

    pub fn update_next_use(&mut self, next_use: usize) {
        match self {
            BlockStatus::Free(..) => panic!("free block does not have next_use"),
            BlockStatus::Splitted(..) => panic!("splitted block does not have next_use"),
            BlockStatus::Occupied(_, _, nu) => *nu = next_use,
        }
    }

    pub fn addr_id(&self) -> AddrId {
        match self {
            BlockStatus::Free(..) => panic!("free block does not have addr_id"),
            BlockStatus::Splitted(addr_id, ..) | BlockStatus::Occupied(addr_id, ..) => *addr_id,
        }
    }
}

#[derive(Debug, Clone)]
struct Block {
    /// [`AddrId`] Identifies the underlying data of the block, regardless of the block's status.
    /// This is useful when the allocator wants to regret a previous allocation, where it can alter the actual address
    /// associated with the ID without notifying the upper software layer.
    status: BlockStatus,
}

#[derive(Debug, Clone)]
pub struct Allocator {
    capacity: u64,
    /// Log2 block sizes, from small to big
    lbss: Vec<u32>,
    blocks: BTreeMap<u32, BTreeMap<u64, Block>>,
    next_new_biggest_block_addr: u64,
    now: usize,
}

impl Allocator {
    pub fn new(capacity: u64, lbss: Vec<IntegralSize>) -> Self {
        let lbss = lbss.into_iter().map(|x| x.0).collect::<Vec<_>>();
        Self {
            capacity,
            lbss,
            blocks: BTreeMap::new(),
            next_new_biggest_block_addr: 0,
            now: 0,
        }
    }

    pub fn tick(&mut self, t: Instant) {
        assert!(t.0 > self.now);
        self.now = t.0;
    }

    fn max_lbs(&self) -> u32 {
        self.lbss.last().cloned().unwrap()
    }

    fn parent_lbs(&self, lbs: u32) -> Option<u32> {
        let idx = self
            .lbss
            .iter()
            .position(|&x| x == lbs)
            .expect("invalid block size");
        self.lbss.get(idx + 1).cloned()
    }

    fn child_lbs(&self, lbs: u32) -> Option<u32> {
        let idx = self
            .lbss
            .iter()
            .position(|&x| x == lbs)
            .expect("invalid block size");
        if idx == 0 {
            return None;
        }
        Some(self.lbss[idx - 1])
    }

    fn parent_addr(&self, addr: u64, lbs: u32) -> Option<u64> {
        let parent_lbs = self.parent_lbs(lbs)?;
        Some(addr & (!0u64 << parent_lbs))
    }

    fn child_addrs(&self, addr: u64, lbs: u32) -> Option<impl Iterator<Item = u64>> {
        let child_lbs = self.child_lbs(lbs)?;
        Some((addr..(addr + (1 << lbs))).step_by(1 << child_lbs))
    }

    pub fn reuse_addr(&mut self, addr_id: AddrId, next_use: Instant, mapping: &impl AddrMappingHandler) {
        self.update_next_use(addr_id, next_use, mapping);
    }

    fn update_next_use_in_parent(&mut self, addr: u64, lbs: u32, next_use: usize) {
        if let Some(parent_addr) = self.parent_addr(addr, lbs) {
            let parent_lbs = self.parent_lbs(lbs).unwrap();
            let parent_block = self.block_mut(parent_addr, parent_lbs);

            if let BlockStatus::Splitted(_, _, ref mut children_next_use) = parent_block.status {
                children_next_use.insert(addr, next_use);
                self.update_next_use_in_parent(
                    parent_addr,
                    parent_lbs,
                    self.blocks[&parent_lbs][&parent_addr].status.next_use(),
                );
            } else {
                panic!("any block's parent should be splitted")
            }
        }
    }

    pub fn update_next_use(
        &mut self,
        addr_id: AddrId,
        next_use: Instant,
        mapping: &impl AddrMappingHandler,
    ) {
        let (Addr(addr), bs) = mapping.get(addr_id);
        let lbs: IntegralSize = bs.try_into().expect("block size should be power of 2");
        let lbs = lbs.0;

        self.block_mut(addr, lbs).status.update_next_use(next_use.0);
        self.update_next_use_in_parent(addr, lbs, next_use.0);
    }

    fn remove_next_use_in_parent(&mut self, addr: u64, lbs: u32) {
        if let Some(parent_addr) = self.parent_addr(addr, lbs) {
            let parent_lbs = self.parent_lbs(lbs).unwrap();
            let parent_block = self.block_mut(parent_addr, parent_lbs);

            if let BlockStatus::Splitted(_, _, ref mut children_next_use) = parent_block.status {
                children_next_use.remove(&addr);
                self.update_next_use_in_parent(
                    parent_addr,
                    parent_lbs,
                    self.blocks[&parent_lbs][&parent_addr].status.next_use(),
                );
            } else {
                panic!("any block's parent should be splitted")
            }
        }
    }

    fn new_addr_id(addr: u64, lbs: u32, mapping: &mut impl AddrMappingHandler) -> AddrId {
        mapping.add(Addr(addr), Size::Integral(IntegralSize(lbs)))
    }

    fn addr_id_of(&self, addr: u64, lbs: u32) -> AddrId {
        self.blocks[&lbs][&addr].addr
    }

    fn block_mut(&mut self, addr: u64, lbs: u32) -> &mut Block {
        self.blocks.get_mut(&lbs).unwrap().get_mut(&addr).unwrap()
    }

    fn plan_relocation_of_children<const ALLOW_TRANSFER: bool>(
        &mut self,
        addr: u64,
        lbs: u32,
        child_lbs: u32,
        mapping: &mut impl AddrMappingHandler,
    ) -> Option<Vec<Transfer>> {
        // TODO use argumenting path algorithm instead of greedy
        let mut reassign_dest2src = BTreeMap::<u64, u64>::new();
        let mut transfer_dest2src = BTreeMap::<u64, u64>::new();
        for child_addr in self.blocks[&lbs][&addr].status.nonfree_child_addrs() {
            if let Some(child_live_at) = self.blocks[&child_lbs][&child_addr].status.live_at() {
                // For each nonfree child, first try find a free block to reassign the location of the child
                let mut candidates_addrs = self.blocks[&child_lbs]
                    .iter()
                    .filter(|(_, can_block)| {
                        can_block
                            .status
                            .last_die_at()
                            .is_some_and(|x| x < child_live_at)
                    })
                    .filter(|(can_addr, _)| {
                        !reassign_dest2src.contains_key(&can_addr)
                            && !transfer_dest2src.contains_key(&can_addr)
                    });
                if let Some((dest, _)) = candidates_addrs.next() {
                    reassign_dest2src.insert(*dest, child_addr);
                } else if ALLOW_TRANSFER {
                    // If no free block that accomodate the child for its whole lifetime, move the child to a free block
                    let mut candidate_addrs = self.blocks[&child_lbs]
                        .iter()
                        .filter(|(_, can_block)| can_block.status.free())
                        .filter(|(can_addr, _)| {
                            !reassign_dest2src.contains_key(&can_addr)
                                && !transfer_dest2src.contains_key(&can_addr)
                        });
                    if let Some((dest, _)) = candidate_addrs.next() {
                        transfer_dest2src.insert(*dest, child_addr);
                    } else {
                        // Otherwise, relocation fails
                        return None;
                    }
                } else {
                    return None;
                }
            }
        }
        // Reassign addresses of blocks
        reassign_dest2src.iter().for_each(|(dest, src)| {
            let src_block = self
                .blocks
                .get_mut(&child_lbs)
                .unwrap()
                .remove(src)
                .unwrap();
            let dest_block = self
                .blocks
                .get_mut(&child_lbs)
                .unwrap()
                .remove(dest)
                .unwrap();
            mapping.update(
                src_block.addr,
                Addr(*dest),
                Size::Integral(IntegralSize(child_lbs)),
            );
            mapping.update(
                dest_block.addr,
                Addr(*src),
                Size::Integral(IntegralSize(child_lbs)),
            );
            self.blocks
                .get_mut(&child_lbs)
                .unwrap()
                .insert(*dest, src_block);
            self.blocks
                .get_mut(&child_lbs)
                .unwrap()
                .insert(*src, dest_block);
        });

        // Transfer blocks
        transfer_dest2src.iter().for_each(|(dest, src)| {
            let mut src_block = self
                .blocks
                .get_mut(&child_lbs)
                .unwrap()
                .remove(src)
                .unwrap();
            let mut dest_block = self
                .blocks
                .get_mut(&child_lbs)
                .unwrap()
                .remove(dest)
                .unwrap();
            std::mem::swap(&mut src_block.addr, &mut dest_block.addr);
            // The block "swapped" here died after last instant, as the transfer should be executed before `now`
            dest_block.status = BlockStatus::Free(self.now - 1);
            self.blocks
                .get_mut(&child_lbs)
                .unwrap()
                .insert(*dest, src_block);
            self.blocks
                .get_mut(&child_lbs)
                .unwrap()
                .insert(*src, dest_block);
        });

        let transfers = transfer_dest2src
            .into_iter()
            .map(|(dest, src)| Transfer {
                from: self.addr_id_of(src, child_lbs),
                to: self.addr_id_of(dest, child_lbs),
            })
            .collect();

        *self.block_mut(addr, lbs).status.unwrap_splitted_mut() = MmHeap::new();

        Some(transfers)
    }

    fn try_condense_smaller_blocks<const ALLOW_TRANSFER: bool>(
        &mut self,
        lbs: u32,
        mapping: &mut impl AddrMappingHandler,
    ) -> Option<(Vec<Transfer>, AddrId)> {
        if let Some(child_lbs) = self.child_lbs(lbs) {
            let n_nonfree_blocks: Vec<(u64, usize)> = self.blocks[&lbs]
                .iter()
                .filter_map(|(addr, block)| {
                    if let BlockStatus::Splitted(_, _, ref children_next_use) = &block.status {
                        Some((*addr, children_next_use.len()))
                    } else {
                        None
                    }
                })
                .collect();

            // Take the block with the least non-free children to try emptying its children
            if let Some((addr, _)) = n_nonfree_blocks.iter().min_by_key(|(_, n)| *n).cloned() {
                if let Some(transfers) = self
                    .plan_relocation_of_children::<ALLOW_TRANSFER>(addr, lbs, child_lbs, mapping)
                {
                    return Some((transfers, self.addr_id_of(addr, lbs)));
                }
            }
        }
        None
    }

    fn _allocate<const TRY_CONDENSING: bool>(
        &mut self,
        size: IntegralSize,
        next_use: Instant,
        mapping: &mut impl AddrMappingHandler,
    ) -> Option<(Vec<Transfer>, AddrId)> {
        let lbs = size.0;
        let next_use = next_use.0;

        if !self.lbss.contains(&lbs) {
            panic!("invalid block size")
        }

        // First try find a free block of exact size
        if let Some((addr, block)) = self
            .blocks
            .get_mut(&lbs)
            .map(|blocks| {
                blocks
                    .iter_mut()
                    .filter(|(_, block)| matches!(block.status, BlockStatus::Free(..)))
                    .next()
            })
            .flatten()
        {
            let addr = *addr;
            let addr_id = Self::new_addr_id(addr, lbs, mapping);

            block.status = BlockStatus::Occupied(addr_id, self.now, next_use);
            self.update_next_use_in_parent(addr, lbs, next_use);
            return Some((vec![], addr_id));
        }

        // Otherwise, try condensing smaller blocks together by regretting previous allocations, where transfers are not allowed
        if TRY_CONDENSING {
            if let Some((transfers, r)) = self.try_condense_smaller_blocks::<false>(lbs, mapping) {
                return Some((transfers, r));
            }
        }

        // Otherwise, if size is exactly maximum, allocate a new block at tail of currently occupied memory, if there is space left
        if lbs == self.max_lbs() {
            if self.next_new_biggest_block_addr + 2u64.pow(self.max_lbs()) > self.capacity {
                return None;
            }

            let addr_id = Self::new_addr_id(self.next_new_biggest_block_addr, lbs, mapping);

            let block = Block {
                status: BlockStatus::Occupied(addr_id, self.now, next_use),
            };

            self.blocks
                .entry(self.max_lbs())
                .or_default()
                .insert(self.next_new_biggest_block_addr, block);

            self.next_new_biggest_block_addr += 2u64.pow(self.max_lbs());

            return Some((vec![], addr_id));
        }

        // Otherwise, try allocate a block with bigger size and split it, returning the first child block
        let parent_lbs = self.parent_lbs(lbs).unwrap();
        // For blocks of `parent_lbs` to condense successfully, there must be a free block of `lbs`, so there is no need trying condensing
        // when allocating a block of `parent_lbs` since we already know here that there is no free block of `lbs`.
        if let Some((transfers, parent_addr)) =
            self._allocate::<false>(IntegralSize(parent_lbs), Instant(next_use), mapping)
        {
            assert!(transfers.len() == 0);

            let (Addr(parent_addr), _) = mapping.get(parent_addr);

            let now = self.now;
            let parent_addr_id = self.blocks[&parent_lbs][&parent_addr].status.addr_id();
            self.block_mut(parent_addr, parent_lbs).status =
                BlockStatus::Splitted(parent_addr_id, now, MmHeap::new());

            for addr in self.child_addrs(parent_addr, parent_lbs).unwrap() {
                let block = Block {
                    status: BlockStatus::Free(self.now - 1),
                };
                self.blocks.get_mut(&lbs).unwrap().insert(addr, block);
            }

            let addr_id = Self::new_addr_id(parent_addr, lbs, mapping);
            self.block_mut(parent_addr, lbs).status =
                BlockStatus::Occupied(addr_id, self.now, next_use);
            self.update_next_use_in_parent(parent_addr, lbs, next_use);
            let addr_id = self.addr_id_of(parent_addr, lbs);

            return Some((vec![], addr_id));
        }

        // Otherwise, try condensing smaller blocks together by regretting previous allocations, while allowing transfers
        if TRY_CONDENSING {
            if let Some((transfers, r)) = self.try_condense_smaller_blocks::<true>(lbs, mapping) {
                return Some((transfers, r));
            }
        }

        // Otherwise, allocation fails
        None
    }

    pub fn allocate(
        &mut self,
        size: IntegralSize,
        next_use: Instant,
        mapping: &mut impl AddrMappingHandler,
    ) -> Option<(Vec<Transfer>, AddrId)> {
        self._allocate::<true>(size, next_use, mapping)
    }

    fn gather_occupied_child_blocks(&self, addr: u64, lbs: u32, append: &mut impl FnMut(AddrId)) {
        match &self.blocks[&lbs][&addr].status {
            BlockStatus::Occupied(..) => {
                append(self.blocks[&lbs][&addr].addr);
            }
            BlockStatus::Splitted(_, _, children_next_use) => {
                for addr in children_next_use.keys().copied() {
                    self.gather_occupied_child_blocks(addr, lbs, append);
                }
            }
            BlockStatus::Free(..) => {}
        }
    }

    pub fn decide_and_realloc_victim(
        &mut self,
        size: IntegralSize,
        next_use: Instant,
        mapping: &mut impl AddrMappingHandler,
    ) -> (AddrId, Vec<AddrId>) {
        let lbs = size.0;

        // First try find a occupied block of exact size, choosing the one that is used latest
        if let Some((addr, _)) = self
            .blocks
            .get(&lbs)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|(addr, block)| {
                        if let BlockStatus::Occupied(_, _, next_use) = block.status {
                            Some((addr, next_use))
                        } else {
                            None
                        }
                    })
                    .max_by_key(|(_, next_use)| *next_use)
            })
            .flatten()
        {
            let addr = *addr;
            let now = self.now;
            let block = self.block_mut(addr, lbs);
            let old_addr_id = block.addr;
            let addr_id = Self::new_addr_id(addr, lbs, mapping);
            block.status = BlockStatus::Occupied(addr_id, now, next_use.0);

            return (block.addr, vec![old_addr_id]);
        }

        // Otherwise, try find a splitted block of exact size, choosing the one that is used latest
        if let Some((addr, _)) = self
            .blocks
            .get(&lbs)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|(addr, block)| {
                        if let BlockStatus::Splitted(_, _, children_next_use) = &block.status {
                            Some((addr, children_next_use.max().unwrap()))
                        } else {
                            None
                        }
                    })
                    .max_by_key(|(_, next_use)| *next_use)
            })
            .flatten()
        {
            let addr = *addr;

            let mut occupied_blocks = Vec::new();
            self.gather_occupied_child_blocks(addr, lbs, &mut |id| occupied_blocks.push(id));

            let now = self.now;
            let block = self.block_mut(addr, lbs);
            let addr_id = Self::new_addr_id(addr, lbs, mapping);
            block.status = BlockStatus::Occupied(addr_id, now, next_use.0);

            return (block.addr, occupied_blocks);
        }

        // Otherwise, try find a occupied or splitted block of bigger size
        self.decide_and_realloc_victim(size.double(), next_use, mapping)
    }

    fn _deallocate<const CHECK_OCCUPIED: bool>(&mut self, addr: u64, lbs: u32) {
        if let Some(block) = self
            .blocks
            .get_mut(&lbs)
            .map(|blocks| blocks.get_mut(&addr))
            .flatten()
        {
            if CHECK_OCCUPIED && !matches!(block.status, BlockStatus::Occupied(..)) {
                panic!("deallocated block is not occupied")
            }
            if matches!(block.status, BlockStatus::Free(..)) {
                panic!("deallocated block is already free")
            }

            block.status = BlockStatus::Free(self.now);

            if lbs == self.max_lbs() {
                return;
            }

            self.remove_next_use_in_parent(addr, lbs);

            // If parent block has no nonfree child block, free it.
            if let Some(parent_lbs) = self.parent_lbs(lbs) {
                let parent_addr = self.parent_addr(addr, lbs).unwrap();
                if self.blocks[&parent_lbs][&parent_addr]
                    .status
                    .unwrap_splitted()
                    .is_empty()
                {
                    for child_addr in self.child_addrs(parent_addr, parent_lbs).unwrap() {
                        self.blocks
                            .get_mut(&lbs)
                            .unwrap()
                            .remove(&child_addr)
                            .unwrap();
                    }
                    self._deallocate::<false>(parent_addr, parent_lbs);
                }
            }
        }
    }

    pub fn deallocate(&mut self, addr_id: AddrId, mapping: &mut impl AddrMappingHandler) {
        let (Addr(addr), bs) = mapping.get(addr_id);
        let lbs: IntegralSize = bs.try_into().expect("should be power of 2");
        self._deallocate::<true>(addr, lbs.0);
    }
}
