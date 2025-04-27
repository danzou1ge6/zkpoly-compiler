use super::{Addr, AddrId, AddrMappingHandler, Instant, IntegralSize, Size};
use std::collections::{BTreeMap, BTreeSet};
use zkpoly_common::mm_heap::MmHeap;

static DEBUG: bool = true;

#[derive(Clone, Debug)]
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
    Splitted(usize, MmHeap<u64, usize>),
    /// The block is occupied. `.0` holds the live_at of the block, `.1` holds the next_use of the block.
    Occupied(usize, usize),
}

impl BlockStatus {
    pub fn unwrap_splitted(&self) -> &MmHeap<u64, usize> {
        match self {
            BlockStatus::Free(..) => panic!("free block does not have splitted"),
            BlockStatus::Splitted(_, heap) => heap,
            BlockStatus::Occupied(..) => panic!("occupied block does not have splitted"),
        }
    }

    pub fn unwrap_splitted_mut(&mut self) -> &mut MmHeap<u64, usize> {
        match self {
            BlockStatus::Free(..) => panic!("free block does not have splitted"),
            BlockStatus::Splitted(_, heap) => heap,
            BlockStatus::Occupied(..) => panic!("occupied block does not have splitted"),
        }
    }

    pub fn nonfree_child_addrs<'a>(&'a self) -> impl Iterator<Item = u64> + 'a {
        match self {
            BlockStatus::Splitted(_, heap) => heap.keys().copied(),
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
        self.try_next_use()
            .expect("free block does not have next_use")
    }

    pub fn try_next_use(&self) -> Option<usize> {
        match self {
            BlockStatus::Free(..) => None,
            BlockStatus::Splitted(_, children_next_use) => Some(children_next_use.min().unwrap().0),
            BlockStatus::Occupied(_, next_use) => Some(*next_use),
        }
    }

    pub fn update_next_use(&mut self, next_use: usize) {
        match self {
            BlockStatus::Free(..) => panic!("free block does not have next_use"),
            BlockStatus::Splitted(..) => panic!("splitted block does not have next_use"),
            BlockStatus::Occupied(_, nu) => *nu = next_use,
        }
    }
}

#[derive(Debug, Clone)]
struct Block {
    status: BlockStatus,
    /// Identifies the underlying data of the block, regardless of the block's status.
    /// This is useful when the allocator wants to regret a previous allocation, where it can alter the actual address
    /// associated with the ID without notifying the upper software layer.
    addr: AddrId,
}

#[derive(Clone)]
pub struct Allocator {
    capacity: u64,
    /// Log2 block sizes, from small to big
    lbss: Vec<u32>,
    blocks: BTreeMap<u32, BTreeMap<u64, Block>>,
    next_new_biggest_block_addr: u64,
    now: usize,
}

struct BlocksDebugger<'a>(&'a Allocator);

impl<'a> std::fmt::Debug for BlocksDebugger<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn fmt_block(
            f: &mut std::fmt::Formatter<'_>,
            lbs: u32,
            addr: u64,
            a: &Allocator,
            depth: usize,
            marker: &mut BTreeSet<(u32, u64)>,
        ) -> std::fmt::Result {
            let block = &a.blocks[&lbs][&addr];
            match &block.status {
                BlockStatus::Free(..) => {}
                BlockStatus::Occupied(lve_at, next_use) => write!(
                    f,
                    "{}({}, {}, {}): Occupied({},{})\n",
                    " ".repeat(depth * 2),
                    lbs,
                    addr,
                    usize::from(block.addr),
                    lve_at,
                    if *next_use == usize::MAX {
                        "INF".to_string()
                    } else {
                        next_use.to_string()
                    }
                )?,
                BlockStatus::Splitted(live_at, children_next_use) => {
                    write!(
                        f,
                        "{}({}, {}, {}): Splitted({}, {})\n",
                        " ".repeat(depth * 2),
                        lbs,
                        addr,
                        usize::from(block.addr),
                        live_at,
                        {
                            let next_use = children_next_use.min().unwrap();
                            if next_use.0 == usize::MAX {
                                "INF".to_string()
                            } else {
                                next_use.0.to_string()
                            }
                        }
                    )?;
                    for child_addr in a.child_addrs(addr, lbs).into_iter().flatten() {
                        let child_lbs = a.child_lbs(lbs).unwrap();
                        if let Some(next_use) =
                            a.blocks[&child_lbs][&child_addr].status.try_next_use()
                        {
                            if next_use != children_next_use[&child_addr] {
                                write!(f, "{} ERROR: child at {} next_use {} does not match that recorded in parent {}\n", " ".repeat(depth * 2), child_addr, next_use, children_next_use[&child_addr])?;
                            }
                        }
                    }
                    fmt_children(f, lbs, addr, a, depth + 1, marker)?;
                }
            };
            Ok(())
        }
        fn fmt_children(
            f: &mut std::fmt::Formatter<'_>,
            lbs: u32,
            addr: u64,
            a: &Allocator,
            depth: usize,
            marker: &mut BTreeSet<(u32, u64)>,
        ) -> std::fmt::Result {
            for child_addr in a.child_addrs(addr, lbs).into_iter().flatten() {
                let child_lbs = a.child_lbs(lbs).unwrap();
                marker.insert((child_lbs, child_addr));
                fmt_block(f, child_lbs, child_addr, a, depth, marker)?;
            }
            Ok(())
        }

        let max_lbs = self.0.max_lbs();
        let mut marker = BTreeSet::new();

        write!(f, "\n")?;

        if self.0.blocks.is_empty() {
            return Ok(());
        }

        for (&addr, _) in self.0.blocks[&max_lbs].iter() {
            fmt_block(f, max_lbs, addr, &self.0, 1, &mut marker)?;
        }

        for (&lbs, blocks) in self.0.blocks.iter().filter(|(lbs, _)| **lbs != max_lbs) {
            for (&addr, _) in blocks.iter() {
                if !marker.contains(&(lbs, addr)) {
                    write!(
                        f,
                        "WARNING: block ({}, {}), parent ({}, {}) not included in tree",
                        lbs,
                        addr,
                        self.0.parent_lbs(lbs).unwrap(),
                        self.0.parent_addr(addr, lbs).unwrap()
                    )?
                }
            }
        }

        Ok(())
    }
}

impl std::fmt::Debug for Allocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Allocator")
            .field("capacity", &self.capacity)
            .field("lbss", &self.lbss)
            .field("blocks", &BlocksDebugger(&self))
            .field(
                "next_new_biggest_block_addr",
                &self.next_new_biggest_block_addr,
            )
            .field("now", &self.now)
            .finish()
    }
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
        assert!(self.now == 0 || t.0 > self.now);
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

    fn update_next_use_in_parent(&mut self, addr: u64, lbs: u32, next_use: usize) {
        if let Some(parent_addr) = self.parent_addr(addr, lbs) {
            let parent_lbs = self.parent_lbs(lbs).unwrap();
            let parent_block = self.block_mut(parent_addr, parent_lbs);

            if let BlockStatus::Splitted(_, ref mut children_next_use) = parent_block.status {
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

        if DEBUG {
            println!("Update next use of ({}, {}) to {}", lbs, addr, next_use.0);
        }

        self.block_mut(addr, lbs).status.update_next_use(next_use.0);
        self.update_next_use_in_parent(addr, lbs, next_use.0);

        if DEBUG {
            println!("{:#?}", self);
        }
    }

    fn remove_next_use_in_parent(&mut self, addr: u64, lbs: u32) {
        if let Some(parent_addr) = self.parent_addr(addr, lbs) {
            let parent_lbs = self.parent_lbs(lbs).unwrap();
            let parent_block = self.block_mut(parent_addr, parent_lbs);

            if let BlockStatus::Splitted(_, ref mut children_next_use) = parent_block.status {
                children_next_use.remove(&addr);
                let updated_parent_next_use = children_next_use.min().map_or(usize::MAX, |x| x.0);
                self.update_next_use_in_parent(parent_addr, parent_lbs, updated_parent_next_use);
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

    fn insert_block(&mut self, addr: u64, lbs: u32, block: Block) {
        if let Some(next_use) = block.status.try_next_use() {
            self.update_next_use_in_parent(addr, lbs, next_use);
        }
        self.blocks.get_mut(&lbs).unwrap().insert(addr, block);
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
                    .filter(|(child_addr, _)| {
                        self.parent_addr(**child_addr, child_lbs).unwrap() != addr
                    })
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
                        .filter(|(child_addr, _)| {
                            self.parent_addr(**child_addr, child_lbs).unwrap() != addr
                        })
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
            self.insert_block(*dest, child_lbs, src_block);
            self.insert_block(*src, child_lbs, dest_block);
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
            self.insert_block(*dest, child_lbs, src_block);
            self.insert_block(*src, child_lbs, dest_block);
        });

        let transfers = transfer_dest2src
            .into_iter()
            .map(|(dest, src)| Transfer {
                from: self.addr_id_of(src, child_lbs),
                to: self.addr_id_of(dest, child_lbs),
            })
            .collect();

        for child_addr in self.child_addrs(addr, lbs).unwrap() {
            self.blocks.get_mut(&child_lbs).unwrap().remove(&child_addr);
        }
        self.block_mut(addr, lbs).status = BlockStatus::Free(self.now - 1);
        self.remove_next_use_in_parent(addr, lbs);

        Some(transfers)
    }

    fn try_condense_smaller_blocks<const ALLOW_TRANSFER: bool>(
        &mut self,
        lbs: u32,
        next_use: usize,
        mapping: &mut impl AddrMappingHandler,
    ) -> Option<(Vec<Transfer>, AddrId)> {
        if let Some(child_lbs) = self.child_lbs(lbs) {
            let n_nonfree_blocks: Vec<(u64, usize)> = self.blocks.get(&lbs).map_or_else(
                || Vec::new(),
                |lbs_blocks| {
                    lbs_blocks
                        .iter()
                        .filter_map(|(addr, block)| {
                            if let BlockStatus::Splitted(_, ref children_next_use) = &block.status {
                                Some((*addr, children_next_use.len()))
                            } else {
                                None
                            }
                        })
                        .collect()
                },
            );
            if n_nonfree_blocks.len() <= 1 {
                return None;
            }
            // Take the block with the least non-free children to try emptying its children
            if let Some((addr, _)) = n_nonfree_blocks.iter().min_by_key(|(_, n)| *n).cloned() {
                if let Some(transfers) = self
                    .plan_relocation_of_children::<ALLOW_TRANSFER>(addr, lbs, child_lbs, mapping)
                {
                    self.block_mut(addr, lbs).status = BlockStatus::Occupied(self.now, next_use);
                    self.update_next_use_in_parent(addr, lbs, next_use);
                    return Some((transfers, self.addr_id_of(addr, lbs)));
                }
            }
        }
        None
    }

    fn subdivide_parent_and_alloc_first(
        &mut self,
        parent_addr: u64,
        parent_lbs: u32,
        next_use: usize,
        mapping: &mut impl AddrMappingHandler,
    ) -> u64 {
        let lbs = self.child_lbs(parent_lbs).unwrap();

        let now = self.now;
        self.block_mut(parent_addr, parent_lbs).status = BlockStatus::Splitted(now, MmHeap::new());

        for addr in self.child_addrs(parent_addr, parent_lbs).unwrap() {
            let block = Block {
                addr: Self::new_addr_id(addr, lbs, mapping),
                status: BlockStatus::Free(self.now.saturating_sub(1)),
            };
            self.blocks.entry(lbs).or_default().insert(addr, block);
        }

        self.block_mut(parent_addr, lbs).status = BlockStatus::Occupied(self.now, next_use);
        self.update_next_use_in_parent(parent_addr, lbs, next_use);

        parent_addr
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
            block.status = BlockStatus::Occupied(self.now, next_use);
            block.addr = Self::new_addr_id(*addr, lbs, mapping);
            let addr_id = block.addr;
            let addr = *addr;
            self.update_next_use_in_parent(addr, lbs, next_use);
            return Some((vec![], addr_id));
        }

        // Otherwise, try condensing smaller blocks together by regretting previous allocations, where transfers are not allowed
        if TRY_CONDENSING {
            if let Some((transfers, r)) =
                self.try_condense_smaller_blocks::<false>(lbs, next_use, mapping)
            {
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
                addr: addr_id,
                status: BlockStatus::Occupied(self.now, next_use),
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
            let addr = self.subdivide_parent_and_alloc_first(parent_addr, parent_lbs, next_use, mapping);
            let addr_id = self.addr_id_of(addr, lbs);

            return Some((vec![], addr_id));
        }

        // Otherwise, try condensing smaller blocks together by regretting previous allocations, while allowing transfers
        if TRY_CONDENSING {
            if let Some((transfers, r)) =
                self.try_condense_smaller_blocks::<true>(lbs, next_use, mapping)
            {
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
        if DEBUG {
            println!("Allocate {:?}, next_use = {:?}", size, next_use);
        }

        let r = self._allocate::<true>(size, next_use, mapping);

        if DEBUG {
            if let Some((_, addr)) = &r {
                println!("Allocated at {:?}", mapping.get(*addr))
            } else {
                println!("Allocation failed");
            }
            println!("{:#?}", self);
        }
        r
    }

    fn gather_and_remove_occupied_child_blocks(
        &mut self,
        addr: u64,
        lbs: u32,
        append: &mut impl FnMut(AddrId),
        remove_self: bool,
    ) {
        match &self.blocks[&lbs][&addr].status {
            BlockStatus::Occupied(..) => {
                append(self.blocks[&lbs][&addr].addr);
            }
            BlockStatus::Splitted(_, _) => {
                let child_lbs = self.child_lbs(lbs).unwrap();
                for addr in self.child_addrs(addr, lbs).unwrap() {
                    self.gather_and_remove_occupied_child_blocks(addr, child_lbs, append, true);
                }
            }
            BlockStatus::Free(..) => {}
        }
        if remove_self {
            self.blocks.get_mut(&lbs).unwrap().remove(&addr);
        }
    }

    pub fn decide_and_realloc_victim(
        &mut self,
        size: IntegralSize,
        next_use: Instant,
        mapping: &mut impl AddrMappingHandler,
    ) -> Option<(AddrId, Vec<AddrId>)> {
        let lbs = size.0;

        if DEBUG {
            println!("Decide victim {:?}, next_use = {:?}", size, next_use);
        }

        // Try choosing the block of exact size that is used latest
        if let Some((addr, _)) = self
            .blocks
            .get(&lbs)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|(addr, block)| {
                        block.status.try_next_use().map(|nu| (addr, nu))
                    })
                    .filter(|(_, next_use) |*next_use != self.now)
                    .max_by_key(|(_, next_use)| *next_use)
            })
            .flatten()
        {
            let addr = *addr;
            let now = self.now;

            let mut occupied_blocks = Vec::new();
            self.gather_and_remove_occupied_child_blocks(
                addr,
                lbs,
                &mut |id| occupied_blocks.push(id),
                false,
            );

            let block = self.block_mut(addr, lbs);
            block.addr = Self::new_addr_id(addr, lbs, mapping);
            block.status = BlockStatus::Occupied(now, next_use.0);

            let addr_id = block.addr;
            self.update_next_use_in_parent(addr, lbs, next_use.0);

            if DEBUG {
                println!("Decided victim at {:?}", mapping.get(addr_id));
                println!("{:#?}", self)
            }

            return Some((addr_id, occupied_blocks));
        }

        // Otherwise, try find a occupied or splitted block of bigger size
        if let Some(parent_lbs) = self.parent_lbs(lbs) {
            let (addr_id, victims) = self.decide_and_realloc_victim(IntegralSize(parent_lbs), next_use, mapping)?;
            let (Addr(parent_addr), _) = mapping.get(addr_id);

            let addr = self.subdivide_parent_and_alloc_first(parent_addr, parent_lbs, next_use.0, mapping);
            let addr_id = self.addr_id_of(addr, lbs);

            Some((addr_id, victims))
        } else {
            None
        }
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

        if DEBUG {
            println!("Deallocate ({}, {})", lbs.0, addr);
        }

        self._deallocate::<true>(addr, lbs.0);

        if DEBUG {
            println!("{:#?}", self);
        }
    }
}

#[cfg(test)]
mod test {
    use std::collections::{BTreeMap, BTreeSet};

    use super::super::AddrMappingHandler;
    use crate::transit::{
        type2::memory_planning::{GpuAddrMappingHandler, Instant},
        type3::{AddrMapping, IntegralSize, Size},
    };

    struct A {
        log_size_idx: usize,
        next_use_after_alloc: usize,
        deallocate_after_next_use: usize,
    }

    impl A {
        fn new(
            log_size_idx: usize,
            next_use_after_alloc: usize,
            deallocate_after_next_use: usize,
        ) -> Self {
            Self {
                log_size_idx,
                next_use_after_alloc,
                deallocate_after_next_use,
            }
        }
    }

    fn run_list(list: Vec<A>, lbss: Vec<u32>, cap: usize) {
        let mut allocator =
            super::Allocator::new(cap as u64, lbss.iter().copied().map(IntegralSize).collect());
        let mut mapping = AddrMapping::new();
        let mut handler = GpuAddrMappingHandler(&mut mapping, 0);

        let mut deallocation_queue = BTreeSet::new();
        let mut ejected = BTreeSet::new();

        for (
            t,
            A {
                log_size_idx,
                next_use_after_alloc,
                deallocate_after_next_use,
            },
        ) in list.into_iter().enumerate()
        {
            allocator.tick(Instant(t));
            let log_size = lbss[log_size_idx];

            let addr_id = if let Some((_, addr_id)) = allocator.allocate(
                IntegralSize(log_size),
                Instant(t + next_use_after_alloc),
                &mut handler,
            ) {
                addr_id
            } else {
                let (addr_id, victims) = allocator.decide_and_realloc_victim(
                    IntegralSize(log_size),
                    Instant(t + next_use_after_alloc),
                    &mut handler,
                ).unwrap_or_else(|| panic!("no valid victim"));
                let _ = victims.into_iter().map(|v| ejected.insert(v));
                addr_id
            };

            deallocation_queue.insert((
                t + next_use_after_alloc + deallocate_after_next_use,
                addr_id,
            ));

            loop {
                if let Some((t1, _)) = deallocation_queue.first() {
                    if *t1 > t {
                        break;
                    }
                    let (_, addr_id) = deallocation_queue.pop_first().unwrap();
                    if !ejected.contains(&addr_id) {
                        allocator.deallocate(addr_id, &mut handler);
                    }
                }
            }
        }
    }

    #[test]
    fn test1() {
        let lbss = vec![2, 4, 8];
        let list = vec![
            // Some random entries
            A::new(0, 1, 2),
            A::new(0, 0, 3),
            A::new(0, 3, 4),
            A::new(1, 3, 3),
            A::new(1, 0, 4),
            A::new(0, 4, 6),
            A::new(0, 5, 2),
            A::new(2, 0, 5),
            A::new(2, 6, 6),
            A::new(2, 0, 1),
            A::new(0, 0, 2),
            A::new(1, 3, 4),
            A::new(0, 0, 1),
            A::new(2, 3, 1),
            A::new(0, 0, 2),
            A::new(0, 0, 1),
            A::new(1, 3, 4),
            A::new(0, 0, 2),
        ];

        run_list(list, lbss, 1024);
    }

    #[test]
    fn test_eject() {
        let lbss = vec![1, 2, 3];
        let list = vec![
            A::new(0, 4, 6),
            A::new(1, 2, 5),
            A::new(2, 3, 6),
            A::new(2, 4, 7), // some is ejected here
            A::new(0, 3, 1),
            A::new(1, 2, 3),
            A::new(2, 3, 4),
            A::new(0, 3, 1),
        ];

        run_list(list, lbss, 16);
    }
}
