//! A datastructure for testing whether an interval intersects with a growing set of intervals.

use std::cmp;

/// An half-open interval.
#[derive(Debug, Clone, Copy)]
pub struct Interval<I> {
    pub begin: I,
    pub end: I,
}

impl<I: Ord> Interval<I> {
    fn new(begin: I, end: I) -> Self {
        Interval { begin, end }
    }

    fn contains(&self, x: I) -> bool {
        self.begin <= x && x <= self.end
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OverlappingError;

#[derive(Clone, Debug)]
struct Node<I> {
    interval: Interval<I>,
    children: Option<(Box<Node<I>>, Box<Node<I>>)>,
}

impl<I: Ord + Copy> Node<I> {
    pub fn leaf(interval: Interval<I>) -> Self {
        Self {
            interval,
            children: None,
        }
    }

    pub fn insert(&mut self, interval: Interval<I>) -> Result<(), OverlappingError> {
        if let Some((left, right)) = &mut self.children {
            if interval.end <= right.interval.begin {
                left.insert(interval)?;
                self.interval.begin = cmp::min(self.interval.begin, interval.begin);
                Ok(())
            } else if interval.begin >= left.interval.end {
                right.insert(interval)?;
                self.interval.end = cmp::max(self.interval.end, interval.end);
                Ok(())
            } else {
                Err(OverlappingError)
            }
        } else {
            if interval.begin >= self.interval.end {
                self.children = Some((
                    Box::new(Node::leaf(self.interval)),
                    Box::new(Node::leaf(interval)),
                ));
                self.interval = Interval::new(self.interval.begin, interval.end);
                Ok(())
            } else if interval.end <= self.interval.begin {
                self.children = Some((
                    Box::new(Node::leaf(interval)),
                    Box::new(Node::leaf(self.interval)),
                ));
                self.interval = Interval::new(interval.begin, self.interval.end);
                Ok(())
            } else {
                Err(OverlappingError)
            }
        }
    }

    pub fn query_scalar(&self, x: I) -> bool {
        match &self.children {
            None => self.interval.contains(x),
            Some((left, right)) => {
                if x < left.interval.end {
                    left.query_scalar(x)
                } else if x >= right.interval.begin {
                    right.query_scalar(x)
                } else {
                    false
                }
            }
        }
    }
    
    pub fn query_interval(&self, interval: Interval<I>) -> bool {
        match &self.children {
            None => !(interval.begin > self.interval.end || interval.end < self.interval.begin),
            Some((left, right)) => {
                if interval.end <= right.interval.begin {
                    left.query_interval(interval)
                } else if interval.begin >= left.interval.end {
                    right.query_interval(interval)
                } else {
                    true
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tree<I>(Option<Node<I>>);

impl<I: Ord + Copy> Tree<I> {
    pub fn new() -> Self {
        Self(None)
    }

    pub fn insert(&mut self, interval: Interval<I>) -> Result<(), OverlappingError> {
        match &mut self.0 {
            None => {
                self.0 = Some(Node::leaf(interval));
                Ok(())
            }
            Some(node) => node.insert(interval),
        }
    }

    pub fn query_scalar(&self, x: I) -> bool {
        match &self.0 {
            None => false,
            Some(node) => node.query_scalar(x),
        }
    }

    pub fn query_interval(&self, interval: Interval<I>) -> bool {
        match &self.0 {
            None => false,
            Some(node) => node.query_interval(interval),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> Tree<u32> {
        let mut tree = Tree::new();
        tree.insert(Interval::new(1, 10)).unwrap();
        tree.insert(Interval::new(15, 20)).unwrap();
        tree.insert(Interval::new(21, 30)).unwrap();
        tree
    }

    #[test]
    fn test_interval_tree() {
        let tree = tree();

        assert!(tree.query_scalar(5));
        assert!(tree.query_scalar(1));
        assert!(!tree.query_scalar(20));
        assert!(!tree.query_scalar(31));

        assert!(tree.query_interval(Interval::new(5, 6)));
        assert!(tree.query_interval(Interval::new(11, 16)));
        assert!(tree.query_interval(Interval::new(10, 16)));
        assert!(!tree.query_interval(Interval::new(11, 12)));
        assert!(!tree.query_interval(Interval::new(31, 35)));
        assert!(!tree.query_interval(Interval::new(10, 14)));
    }
}

