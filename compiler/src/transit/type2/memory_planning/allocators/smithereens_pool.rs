//! A module for planning memory for very small fragments using best-fit algorithm.

use std::collections::BTreeMap;
use super::super::{SmithereenSize, Addr};

static DEBUG: bool = false;

#[derive(Debug, Clone)]
struct Chunk {
    size: u64,
    occupied: bool,
    // If Cursor API of BTreeMap is stable, we no longer need keep track of this
    pred_addr: Option<u64>,
    succ_addr: Option<u64>,
}

#[derive(Clone)]
pub struct Pool {
    chunks: BTreeMap<u64, Chunk>,
    last_chunk_addr: u64,
    aligned_addr2chunk_addr: BTreeMap<u64, u64>,
    capacity: u64,
}

struct ChunksDebugger<'a>(&'a Pool);

impl<'a> std::fmt::Debug for ChunksDebugger<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();

        for (((addr, chunk), pred_addr), succ_addr) in self
            .0
            .chunks
            .iter()
            .zip(std::iter::once(None).chain(self.0.chunks.iter().map(|x| Some(*x.0))))
            .zip(
                self.0
                    .chunks
                    .iter()
                    .skip(1)
                    .map(|x| Some(*x.0))
                    .chain(std::iter::once(None)),
            )
        {
            if chunk.occupied {
                list.entry(&format!("{}: Occupied({})\n", addr, chunk.size));
            } else {
                list.entry(&format!("{}: Free({})\n", addr, chunk.size));
            }
            if pred_addr != chunk.pred_addr {
                list.entry(&format!(
                    " ERROR: chunk recorded pred_addr {:?} wrong, expected {:?}\n",
                    chunk.pred_addr, pred_addr
                ));
            }
            if succ_addr != chunk.succ_addr {
                list.entry(&format!(
                    " ERROR: chunk recorded succ_addr {:?} wrong, expected {:?}\n",
                    chunk.succ_addr, succ_addr
                ));
            }
        }

        list.finish()
    }
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Allocator")
            .field("last_chunk_addr", &self.last_chunk_addr)
            .field("aligned_addr2chunk_addr", &self.aligned_addr2chunk_addr)
            .field("capacity", &self.capacity)
            .field("chunks", &ChunksDebugger(self))
            .finish()
    }
}

fn aligned_addr(addr: u64) -> u64 {
    if addr == 0 {
        0
    } else {
        (addr + 255) & !255
    }
}

impl Pool {
    pub fn new(capacity: u64) -> Self {
        Self {
            chunks: [(
                0,
                Chunk {
                    size: capacity,
                    occupied: false,
                    pred_addr: None,
                    succ_addr: None,
                },
            )]
            .into_iter()
            .collect(),
            last_chunk_addr: 0,
            aligned_addr2chunk_addr: BTreeMap::new(),
            capacity,
        }
    }

    fn find_best_fit(&self, size: u64) -> Option<u64> {
        let (addr, _) = self
            .chunks
            .iter()
            .filter(|(&addr, chunk)| {
                !chunk.occupied && addr + chunk.size >= aligned_addr(addr) + size
            })
            .min_by_key(|(_, chunk)| chunk.size - size)?;

        Some(*addr)
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub(super) fn allocate(&mut self, payload_size: SmithereenSize) -> Option<Addr> {
        if DEBUG {
            println!("Allocate Smithereen {}", payload_size.0);
        }

        let payload_size = payload_size.0;
        let addr = self.find_best_fit(payload_size)?;
        let aligned_addr = aligned_addr(addr);
        let size = aligned_addr + payload_size - addr;

        self.aligned_addr2chunk_addr.insert(aligned_addr, addr);

        let mut chunk = self.chunks.remove(&addr).unwrap();

        if chunk.size == size {
            chunk.occupied = true;
            self.chunks.insert(addr, chunk);
            return Some(Addr(aligned_addr));
        }

        let addr_splitted1 = addr;
        let addr_splitted2 = addr + size;
        let chunk_splitted1 = Chunk {
            size,
            occupied: true,
            pred_addr: chunk.pred_addr,
            succ_addr: Some(addr_splitted2),
        };
        let chunk_splitted2 = Chunk {
            size: chunk.size - size,
            occupied: false,
            pred_addr: Some(addr_splitted1),
            succ_addr: chunk.succ_addr,
        };

        if let Some(pred_addr) = chunk.pred_addr {
            self.chunks.get_mut(&pred_addr).unwrap().succ_addr = Some(addr_splitted1);
        }

        if let Some(succ_addr) = chunk.succ_addr {
            self.chunks.get_mut(&succ_addr).unwrap().pred_addr = Some(addr_splitted2);
        }

        self.chunks.insert(addr_splitted1, chunk_splitted1);
        self.chunks.insert(addr_splitted2, chunk_splitted2);

        if self.last_chunk_addr == addr {
            self.last_chunk_addr = addr_splitted2;
        }

        if DEBUG {
            println!("{:#?}", self);
        }

        return Some(Addr(aligned_addr));
    }

    pub(super) fn deallocate(&mut self, aligned_addr: Addr) {
        let Addr(aligned_addr) = aligned_addr;
        let addr = self.aligned_addr2chunk_addr.remove(&aligned_addr).unwrap();

        if DEBUG {
            println!(
                "Deallocate Smithereen at {}, aligned from {}",
                aligned_addr, addr
            );
        }

        let mut chunk = self.chunks.remove(&addr).expect("chunk does not exist");
        if !chunk.occupied {
            panic!("chunk is not occupied");
        }

        chunk.occupied = false;

        // Merge free chunks
        let pred_chunk = chunk
            .pred_addr
            .map(|addr| self.chunks.remove(&addr).unwrap());
        let succ_chunk = chunk
            .succ_addr
            .map(|addr| self.chunks.remove(&addr).unwrap());

        match (pred_chunk, succ_chunk) {
            (Some(pred_chunk), Some(succ_chunk))
                if !pred_chunk.occupied && !succ_chunk.occupied =>
            {
                // Merge pred_chunk, chunk, succ_chunk
                let merged_chunk = Chunk {
                    size: pred_chunk.size + chunk.size + succ_chunk.size,
                    occupied: false,
                    pred_addr: pred_chunk.pred_addr,
                    succ_addr: succ_chunk.succ_addr,
                };

                let merged_addr = chunk.pred_addr.unwrap();

                if chunk.succ_addr.unwrap() == self.last_chunk_addr {
                    self.last_chunk_addr = merged_addr;
                }

                if let Some(succ_succ_chunk_addr) = succ_chunk.succ_addr {
                    self.chunks
                        .get_mut(&succ_succ_chunk_addr)
                        .unwrap()
                        .pred_addr = Some(merged_addr);
                }

                self.chunks.insert(merged_addr, merged_chunk);
            }
            (Some(pred_chunk), option_succ_chunk) if !pred_chunk.occupied => {
                // Merge pred_chunk, chunk
                let merged_chunk = Chunk {
                    size: pred_chunk.size + chunk.size,
                    occupied: false,
                    pred_addr: pred_chunk.pred_addr,
                    succ_addr: chunk.succ_addr,
                };
                let merged_addr = chunk.pred_addr.unwrap();

                if addr == self.last_chunk_addr {
                    self.last_chunk_addr = chunk.pred_addr.unwrap();
                }

                self.chunks.insert(merged_addr, merged_chunk);

                if let Some(mut succ_chunk) = option_succ_chunk {
                    succ_chunk.pred_addr = Some(merged_addr);
                    self.chunks.insert(chunk.succ_addr.unwrap(), succ_chunk);
                }
            }
            (option_pred_chunk, Some(succ_chunk)) if !succ_chunk.occupied => {
                // Merge chunk, succ_chunk
                let merged_chunk = Chunk {
                    size: chunk.size + succ_chunk.size,
                    occupied: false,
                    pred_addr: chunk.pred_addr,
                    succ_addr: succ_chunk.succ_addr,
                };

                if chunk.succ_addr.unwrap() == self.last_chunk_addr {
                    self.last_chunk_addr = addr;
                }

                if let Some(succ_succ_chunk_addr) = succ_chunk.succ_addr {
                    self.chunks
                        .get_mut(&succ_succ_chunk_addr)
                        .unwrap()
                        .pred_addr = Some(addr);
                }

                self.chunks.insert(addr, merged_chunk);

                if let Some(mut pred_chunk) = option_pred_chunk {
                    pred_chunk.succ_addr = Some(addr);
                    self.chunks.insert(chunk.pred_addr.unwrap(), pred_chunk);
                }
            }
            (option_pred_chunk, option_succ_chunk) => {
                if let Some(pred_chunk) = option_pred_chunk {
                    self.chunks.insert(chunk.pred_addr.unwrap(), pred_chunk);
                }

                if let Some(succ_chunk) = option_succ_chunk {
                    self.chunks.insert(chunk.succ_addr.unwrap(), succ_chunk);
                }

                chunk.occupied = false;
                self.chunks.insert(addr, chunk);
            }
        }

        if DEBUG {
            println!("{:#?}", self);
        }
    }
}
