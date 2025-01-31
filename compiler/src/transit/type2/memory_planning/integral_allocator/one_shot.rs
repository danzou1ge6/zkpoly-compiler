use super::{Addr, IntegralSize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockStatus {
    Occupied(usize),
    Splitted(usize),
    Free,
}

#[derive(Debug, Clone)]
struct Block {
    log2_size: u32,
    status: BlockStatus,
}

impl Block {
    fn die_at(&self) -> Option<usize> {
        match self.status {
            BlockStatus::Occupied(die_at) => Some(die_at),
            BlockStatus::Splitted(die_at) => Some(die_at),
            _ => None,
        }
    }
    fn set_die_at(&mut self, die_at: usize) {
        match self.status {
            BlockStatus::Occupied(_) => {
                self.status = BlockStatus::Occupied(die_at);
            }
            BlockStatus::Splitted(_) => {
                self.status = BlockStatus::Splitted(die_at);
            }
            BlockStatus::Free => panic!("called set_die_at on a free block"),
        }
    }
    fn occupied_to_splitted(&mut self) {
        match self.status {
            BlockStatus::Occupied(die_at) => self.status = BlockStatus::Splitted(die_at),
            BlockStatus::Splitted(_) => panic!("called occupied_to_splitted on a splitted block"),
            BlockStatus::Free => panic!("called occupied_to_splitted on a free block"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Allocator {
    capacity: u64,
    log2_max_block_size: u32,
    /// log2_block_size |-> block_addr |-> block
    blocks: BTreeMap<u32, BTreeMap<u64, Block>>,
    last_biggest_block_addr: u64,
}

fn parent_addr(addr: u64, log2_size: u32) -> u64 {
    addr & (!0u64 << (log2_size + 1))
}

fn splitted_addrs(addr: u64, log2_size: u32) -> (u64, u64) {
    (addr, addr + 2u64.pow(log2_size - 1))
}

impl Allocator {
    /// Calculate new die_at of a block and propagate the die_at to the parent block
    fn update_die_at(&mut self, addr: u64, log2_size: u32) {
        if self.blocks[&log2_size][&addr].status == BlockStatus::Free {
            return;
        }

        let (splitted1_addr, splitted2_addr) = splitted_addrs(addr, log2_size);
        let log2_child_bs = log2_size - 1;
        let splitted1_die_at = self.blocks[&log2_child_bs][&splitted1_addr].die_at();
        let splitted2_die_at = self.blocks[&log2_child_bs][&splitted2_addr].die_at();

        let updated_die_at = match (splitted1_die_at, splitted2_die_at) {
            (Some(splitted1_die_at), Some(splitted2_die_at)) => {
                splitted1_die_at.max(splitted2_die_at)
            }
            (None, Some(splitted2_die_at)) => splitted2_die_at,
            (Some(splitted1_die_at), None) => splitted1_die_at,
            _ => unreachable!(),
        };

        self.blocks
            .get_mut(&log2_size)
            .unwrap()
            .get_mut(&addr)
            .unwrap()
            .set_die_at(updated_die_at);

        if log2_size == self.log2_max_block_size {
            return;
        }

        let parent_addr = parent_addr(addr, log2_size);
        self.update_die_at(parent_addr, log2_size + 1);
    }

    fn update_parent_die_at(&mut self, addr: u64, log2_size: u32) {
        self.update_die_at(parent_addr(addr, log2_size), log2_size + 1);
    }

    /// Allocate a block of size `size` and return its address, if there is space left, otherwise return [`None`].
    pub fn allocate(&mut self, size: IntegralSize, die_at: usize) -> Option<Addr> {
        let log2_size = size.0;

        // block size is limited
        if log2_size > self.log2_max_block_size {
            return None;
        }

        // First try find a free block of exact size
        if let Some((addr, block)) = self
            .blocks
            .get_mut(&log2_size)
            .map(|blocks| {
                blocks
                    .iter_mut()
                    .filter(|(_, block)| block.status == BlockStatus::Free)
                    .next()
            })
            .flatten()
        {
            block.status = BlockStatus::Occupied(die_at);
            let addr = *addr;
            self.update_parent_die_at(addr, log2_size);
            return Some(Addr(addr));
        }

        // If we can't find a free block of exact size

        // If size is exactly maximum, allocate a new block at tail of currently occupied memory, if there is space left
        if log2_size == self.log2_max_block_size {
            self.last_biggest_block_addr += 2u64.pow(self.log2_max_block_size);

            if self.last_biggest_block_addr + 2u64.pow(self.log2_max_block_size) > self.capacity {
                return None;
            }

            let block = Block {
                log2_size,
                status: BlockStatus::Occupied(die_at),
            };

            self.blocks
                .entry(log2_size)
                .or_default()
                .insert(self.last_biggest_block_addr, block);

            return Some(Addr(self.last_biggest_block_addr));
        }

        // If size is smaller than maximum, allocate a block twice the size and split it, returning the first half
        // Here the die_at of the bigger block is exactly that of the allocating block, and it will propagate up the tree
        let log2_parent_bs = log2_size + 1;
        let parent_block_addr = self.allocate(IntegralSize(log2_parent_bs), die_at)?.0;

        self.blocks
            .get_mut(&log2_parent_bs)
            .unwrap()
            .get_mut(&parent_block_addr)
            .unwrap()
            .occupied_to_splitted();
        let (splitted1_addr, splitted2_addr) = splitted_addrs(parent_block_addr, log2_parent_bs);
        let splitted1 = Block {
            log2_size,
            status: BlockStatus::Occupied(die_at),
        };
        let splitted2 = Block {
            log2_size,
            status: BlockStatus::Free,
        };

        self.blocks
            .entry(log2_size)
            .or_default()
            .insert(splitted1_addr, splitted1);
        self.blocks
            .get_mut(&log2_size)
            .unwrap()
            .insert(splitted2_addr, splitted2);

        Some(Addr(splitted1_addr))
    }

    fn gather_occupied_child_blocks(
        &self,
        addr: u64,
        log2_size: u32,
        append: &mut impl FnMut(IntegralSize, Addr),
    ) {
        match self.blocks[&log2_size][&addr].status {
            BlockStatus::Occupied(_) => {
                append(IntegralSize(log2_size), Addr(addr));
            }
            BlockStatus::Splitted(_) => {
                let (splitted1_addr, splitted2_addr) = splitted_addrs(addr, log2_size);

                self.gather_occupied_child_blocks(splitted1_addr, log2_size - 1, append);
                self.gather_occupied_child_blocks(splitted2_addr, log2_size - 1, append);
            }
            BlockStatus::Free => {}
        }
    }

    /// Decide some victim blocks and deallocate them so that a space at least of `size` can be allocated.
    /// Returns the list of victim blocks.
    pub fn decide_and_dealloc_victim(&mut self, size: IntegralSize) -> Vec<(IntegralSize, Addr)> {
        let log2_size = size.0;

        // First try find a occupied block of exact size, choosing the one that die latest
        if let Some((addr, _)) = self
            .blocks
            .get(&log2_size)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|(addr, block)| {
                        if let BlockStatus::Occupied(die_at) = block.status {
                            Some((addr, die_at))
                        } else {
                            None
                        }
                    })
                    .max_by_key(|(_, die_at)| *die_at)
            })
            .flatten()
        {
            let addr = *addr;
            self.deallocate(size, Addr(addr));
            return vec![(size, Addr(addr))];
        }

        // Otherwise, try find a splitted block of exact size, choosing the one that die latest
        if let Some((addr, _)) = self
            .blocks
            .get(&log2_size)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|(addr, block)| {
                        if let BlockStatus::Splitted(die_at) = block.status {
                            Some((addr, die_at))
                        } else {
                            None
                        }
                    })
                    .max_by_key(|(_, die_at)| *die_at)
            })
            .flatten()
        {
            let addr = *addr;

            let mut occupied_blocks = Vec::new();
            self.gather_occupied_child_blocks(addr, log2_size, &mut |s, a| {
                occupied_blocks.push((s, a))
            });

            self._deallocate::<false>(addr, log2_size);

            return occupied_blocks;
        }

        // Otherwise, try find a occupied or splitted block of bigger size
        self.decide_and_dealloc_victim(size.double())
    }

    fn _deallocate<const CHECK_OCCUPIED: bool>(&mut self, addr: u64, log2_size: u32) {
        if let Some(block) = self
            .blocks
            .get_mut(&log2_size)
            .map(|blocks| blocks.get_mut(&addr))
            .flatten()
        {
            if CHECK_OCCUPIED && !matches!(block.status, BlockStatus::Occupied(..)) {
                panic!("deallocated block is not occupied")
            }
            if matches!(block.status, BlockStatus::Free) {
                panic!("deallocated block is already free")
            }
            // Mark block as free
            block.status = BlockStatus::Free;

            if log2_size == self.log2_max_block_size {
                return;
            }

            // If the sibling block is also free, free the parent block.
            // Otherwise, die_at of the parent block is updated.
            let log2_parent_bs = log2_size + 1;
            let parent_addr = parent_addr(addr, log2_size);

            let splitted1_addr = addr;
            let splitted2_addr = addr + 2u64.pow(log2_size);
            if self.blocks[&log2_size][&splitted1_addr].status == BlockStatus::Free
                && self.blocks[&log2_size][&splitted2_addr].status == BlockStatus::Free
            {
                self.blocks
                    .get_mut(&log2_size)
                    .unwrap()
                    .remove(&splitted1_addr);
                self.blocks
                    .get_mut(&log2_size)
                    .unwrap()
                    .remove(&splitted2_addr);

                self._deallocate::<false>(parent_addr, log2_parent_bs);
            } else {
                self.update_parent_die_at(addr, log2_size);
            }
        } else {
            panic!("deallocated block does not exist")
        }
    }

    /// Deallocate an occupied block
    pub fn deallocate(&mut self, size: IntegralSize, addr: Addr) {
        let log2_size = size.0;
        let addr = addr.0;

        self._deallocate::<true>(addr, log2_size);
    }
}
