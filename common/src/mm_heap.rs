//! A module for min/max-heap

use std::collections::{BTreeMap, BTreeSet};

/// A random-access, random-remove capable min-heap or max-heap.
/// [`K`] is the key type and [`S`] is the sequence number type.
#[derive(Debug, Clone)]
pub struct MmHeap<K, S> {
    heap: BTreeSet<(S, K)>,
    k2s: BTreeMap<K, S>,
}

impl<K, S> MmHeap<K, S> {
    pub fn new() -> Self {
        Self {
            heap: BTreeSet::new(),
            k2s: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.k2s.len()
    }

    pub fn is_empty(&self) -> bool {
        self.k2s.is_empty()
    }
}

impl<K, S> MmHeap<K, S>
where
    K: Ord + Clone,
    S: Ord + Clone,
{
    /// Inserts a new element with key `k` and sequence number `s`.
    /// If the element already exists, it will be removed first.
    pub fn insert(&mut self, k: K, s: S) {
        if let Some(old_s) = self.k2s.remove(&k) {
            self.heap.remove(&(old_s, k.clone()));
        }
        self.heap.insert((s.clone(), k.clone()));
        self.k2s.insert(k.clone(), s);
    }

    /// Removes the element with key `k`.
    pub fn remove(&mut self, k: &K) {
        let s = self.k2s.remove(k).unwrap();
        self.heap.remove(&(s, k.clone()));
    }

    /// Returns the key of the element with maximum sequence number.
    pub fn max(&self) -> Option<&(S, K)> {
        self.heap.last()
    }

    /// Returns the key of the element with minimum sequence number.
    pub fn min(&self) -> Option<&(S, K)> {
        self.heap.first()
    }

    pub fn pop_max(&mut self) -> Option<K> {
        self.heap.pop_last().map(|(_, k)| k)
    }

    pub fn pop_min(&mut self) -> Option<K> {
        self.heap.pop_first().map(|(_, k)| k)
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.k2s.keys()
    }

}

impl<K, S> std::ops::Index<&K> for MmHeap<K, S>
where
    K: Ord,
{
    type Output = S;

    fn index(&self, index: &K) -> &Self::Output {
        &self.k2s[index]
    }
}
