/// UnionFind data structure.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    count: usize,      // Number of disjoint sets
}

impl UnionFind {
    /// Creates a new UnionFind structure with `n` elements.
    /// Initially, each element is in its own set.
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            count: n,
        }
    }

    /// Finds the representative (root) of the set containing `i`.
    /// Implements path compression.
    pub fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            i
        } else {
            self.parent[i] = self.find(self.parent[i]);
            self.parent[i]
        }
    }

    /// Unites the sets containing `i` and `j`.
    /// The root of `i`'s set will point to the root of `j`'s set.
    /// Returns `true` if `i` and `j` were in different sets, `false` otherwise.
    pub fn union(&mut self, i: usize, j: usize) -> bool {
        let root_i = self.find(i);
        let root_j = self.find(j);

        if root_i != root_j {
            self.parent[root_i] = root_j;
            self.count -= 1;
            true
        } else {
            false
        }
    }

    /// Checks if elements `i` and `j` are in the same set.
    pub fn connected(&mut self, i: usize, j: usize) -> bool {
        self.find(i) == self.find(j)
    }

    /// Returns the number of disjoint sets.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Returns true if the structure is empty.
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let uf = UnionFind::new(5);
        assert_eq!(uf.len(), 5);
        assert_eq!(uf.count(), 5);
        for i in 0..5 {
            assert_eq!(uf.parent[i], i);
        }
    }

    #[test]
    fn test_find() {
        let mut uf = UnionFind::new(5);
        // 0->1, 1->1, 2->3, 3->3, 4->4
        uf.parent = vec![1, 1, 3, 3, 4];
        // Path compression check
        assert_eq!(uf.find(0), 1);
        assert_eq!(uf.parent[0], 1); // Path compression applied
        assert_eq!(uf.find(2), 3);
        assert_eq!(uf.parent[2], 3); // Path compression applied
        assert_eq!(uf.find(4), 4);

        // Deeper path
        // 0->1, 1->2, 2->2
        let mut uf2 = UnionFind::new(3);
        uf2.parent = vec![1, 2, 2];
        assert_eq!(uf2.find(0), 2);
        assert_eq!(uf2.parent[0], 2);
        assert_eq!(uf2.parent[1], 2);
    }

    #[test]
    fn test_union_and_connected() {
        let mut uf = UnionFind::new(10);

        // union(1, 2) -> root of 1 (which is 1) points to root of 2 (which is 2)
        // parent[1] = 2
        assert!(uf.union(1, 2));
        assert!(uf.connected(1, 2));
        assert_eq!(uf.find(1), 2); // Root of 1 is 2
        assert_eq!(uf.parent[1], 2);
        assert_eq!(uf.count(), 9);

        // union(2, 3) -> root of 2 (which is 2) points to root of 3 (which is 3)
        // parent[2] = 3. After find(1), parent[1] becomes 3.
        assert!(uf.union(2, 3));
        assert!(uf.connected(1, 3));
        assert_eq!(uf.find(1), 3); // Root of 1 is 3
        assert_eq!(uf.find(2), 3); // Root of 2 is 3
        assert_eq!(uf.parent[2], 3);
        assert_eq!(uf.parent[1], 3); // Path compression from connected or find
        assert_eq!(uf.count(), 8);

        // union(4, 5) -> parent[4] = 5
        assert!(uf.union(4, 5));
        assert!(uf.connected(4, 5));
        assert_eq!(uf.find(4), 5);
        assert_eq!(uf.count(), 7);

        assert!(!uf.connected(1, 4));

        // union(1, 4) -> root of 1 (which is 3) points to root of 4 (which is 5)
        // parent[3] = 5
        assert!(uf.union(1, 4));
        assert!(uf.connected(1, 5));
        assert!(uf.connected(3, 4));
        assert_eq!(uf.find(1), 5);
        assert_eq!(uf.find(3), 5);
        assert_eq!(uf.parent[3], 5);
        assert_eq!(uf.count(), 6);

        // Test union on already connected elements
        // find(1) is 5, find(5) is 5. They are connected.
        assert!(!uf.union(1, 5));
        assert_eq!(uf.count(), 6); // Count should not change
    }

    #[test]
    fn test_union_specific_root_direction() {
        let mut uf = UnionFind::new(5);
        // union(0,1) => parent[find(0)] = find(1) => parent[0] = 1
        uf.union(0, 1);
        assert_eq!(uf.find(0), 1);
        assert_eq!(uf.parent[0], 1);

        // union(2,3) => parent[find(2)] = find(3) => parent[2] = 3
        uf.union(2, 3);
        assert_eq!(uf.find(2), 3);
        assert_eq!(uf.parent[2], 3);

        // union(0,2) => parent[find(0)] = find(2) => parent[1] = 3
        // find(0) is 1, find(2) is 3
        uf.union(0, 2);
        assert_eq!(uf.find(0), 3);
        assert_eq!(uf.find(1), 3);
        assert_eq!(uf.find(2), 3);
        assert_eq!(uf.parent[1], 3); // parent of old root of 0 (which was 1) is now root of 2 (which is 3)
    }


    #[test]
    fn test_complex_scenario() {
        let mut uf = UnionFind::new(7);
        // 0-1 (0->1)
        uf.union(0, 1);
        // 2-3 (2->3)
        uf.union(2, 3);
        // 4-5 (4->5)
        uf.union(4, 5);
        // 5-6 (find(5)->find(6) => 5->6)
        uf.union(5, 6); // Now 4->5, 5->6. find(4) is 6, find(5) is 6.

        assert!(uf.connected(0, 1));
        assert_eq!(uf.find(0), 1);
        assert!(uf.connected(2, 3));
        assert_eq!(uf.find(2), 3);
        assert!(uf.connected(4, 6));
        assert_eq!(uf.find(4), 6);
        assert!(uf.connected(5, 4)); // same as connected(4,5)
        assert_eq!(uf.find(5), 6);


        assert!(!uf.connected(0, 2));
        assert!(!uf.connected(1, 5));
        assert_eq!(uf.count(), 3); // {0,1 (root 1)}, {2,3 (root 3)}, {4,5,6 (root 6)}

        // Connect {0,1} and {2,3}
        // union(0,3) => find(0) is 1, find(3) is 3. So parent[1] = 3.
        uf.union(0, 3);
        assert!(uf.connected(1, 2));
        assert_eq!(uf.find(0), 3);
        assert_eq!(uf.find(1), 3);
        assert_eq!(uf.count(), 2); // {0,1,2,3 (root 3)}, {4,5,6 (root 6)}

        // Connect all
        // union(1,5) => find(1) is 3, find(5) is 6. So parent[3] = 6.
        uf.union(1, 5);
        assert!(uf.connected(0,6));
        assert_eq!(uf.find(0), 6);
        assert_eq!(uf.find(1), 6);
        assert_eq!(uf.find(2), 6);
        assert_eq!(uf.find(3), 6);
        assert_eq!(uf.find(4), 6);
        assert_eq!(uf.find(5), 6);
        assert_eq!(uf.count(), 1); // {0,1,2,3,4,5,6 (root 6)}
    }

    #[test]
    fn test_is_empty() {
        let uf_empty = UnionFind::new(0);
        assert!(uf_empty.is_empty());
        assert_eq!(uf_empty.len(), 0);
        assert_eq!(uf_empty.count(), 0);

        let uf_not_empty = UnionFind::new(1);
        assert!(!uf_not_empty.is_empty());
    }
}