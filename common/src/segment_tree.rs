use std::cmp::max;

/// 定义一个 trait 来约束线段树中存储的值类型
pub trait SegmentValue: Default + Copy + Ord {
    fn neg_infinity() -> Self; // 代表负无穷大
}

/// 为 i64 实现 SegmentValue
impl SegmentValue for i64 {
    fn neg_infinity() -> Self {
        i64::MIN
    }
}

/// 为 u64 实现 SegmentValue
impl SegmentValue for u64 {
    fn neg_infinity() -> Self {
        u64::MIN
    }
}


#[derive(Debug, Clone)]
struct Node<T: SegmentValue> {
    left: usize,        // 区间左端点 (0-indexed)
    right: usize,       // 区间右端点 (0-indexed)
    max_val: T,         // 区间最大值
    set_tag: Option<T>, // 区间赋值的懒标记: node.val = tag_val
}

impl<T: SegmentValue> Default for Node<T> {
    fn default() -> Self {
        Node {
            left: 0,
            right: 0,
            max_val: T::neg_infinity(),
            set_tag: None,
        }
    }
}

#[derive(Debug)]
pub struct SegmentTree<T: SegmentValue> {
    tree: Vec<Node<T>>,
    data_len: usize,
}

impl<T: SegmentValue> SegmentTree<T> {
    pub fn new(data: &[T]) -> Self {
        let n = data.len();
        if n == 0 {
            return SegmentTree {
                tree: Vec::new(),
                data_len: 0,
            };
        }
        let tree = vec![Node::default(); 4 * n];
        let mut st = SegmentTree {
            tree,
            data_len: n,
        };
        st.build(0, 0, n - 1, data);
        st
    }

    fn build(&mut self, u: usize, l: usize, r: usize, data: &[T]) {
        self.tree[u].left = l;
        self.tree[u].right = r;
        self.tree[u].set_tag = None;

        if l == r {
            self.tree[u].max_val = data[l];
            return;
        }

        let mid = l + (r - l) / 2;
        let ls = 2 * u + 1;
        let rs = 2 * u + 2;

        self.build(ls, l, mid, data);
        self.build(rs, mid + 1, r, data);
        self.push_up(u);
    }

    fn push_up(&mut self, u: usize) {
        let ls = 2 * u + 1;
        let rs = 2 * u + 2;
        self.tree[u].max_val = max(self.tree[ls].max_val, self.tree[rs].max_val);
    }

    // 应用赋值标记到节点 u
    fn apply_set_tag(&mut self, u: usize, tag_val: T) {
        self.tree[u].max_val = tag_val;
        self.tree[u].set_tag = Some(tag_val); // 新的赋值标记覆盖旧的
    }

    // 下传标记
    fn push_down(&mut self, u: usize) {
        if self.tree[u].left == self.tree[u].right { // 叶子节点没有子节点
            self.tree[u].set_tag = None; // 清除叶子节点的标记（如果有的话）
            return;
        }

        if let Some(tag_val) = self.tree[u].set_tag {
            let ls = 2 * u + 1;
            let rs = 2 * u + 2;
            self.apply_set_tag(ls, tag_val);
            self.apply_set_tag(rs, tag_val);
            self.tree[u].set_tag = None;
        }
    }

    /// 区间修改：将 [query_l, query_r] 内的所有值修改为 val
    pub fn modify_set(&mut self, query_l: usize, query_r: usize, val: T) {
        if self.tree.is_empty() || query_l > query_r || query_r >= self.data_len {
            return; // 无效范围或空树
        }
        self._modify_set(0, query_l, query_r, val);
    }

    fn _modify_set(&mut self, u: usize, query_l: usize, query_r: usize, val: T) {
        let node_l = self.tree[u].left;
        let node_r = self.tree[u].right;

        // 如果当前区间完全被查询区间覆盖
        if query_l <= node_l && node_r <= query_r {
            self.apply_set_tag(u, val);
            return;
        }

        self.push_down(u); // 在进一步递归或更新叶子节点之前下推现有标记

        let mid = node_l + (node_r - node_l) / 2;
        let ls = 2 * u + 1;
        let rs = 2 * u + 2;

        if query_l <= mid {
            self._modify_set(ls, query_l, query_r, val);
        }
        if query_r > mid {
            self._modify_set(rs, query_l, query_r, val);
        }

        self.push_up(u);
    }

    /// 查询区间 [query_l, query_r] 内的最大值
    pub fn query_max(&mut self, query_l: usize, query_r: usize) -> Option<T> {
        if self.tree.is_empty() || query_l > query_r || query_r >= self.data_len {
            return None;
        }
        Some(self._query_max(0, query_l, query_r))
    }

    fn _query_max(&mut self, u: usize, query_l: usize, query_r: usize) -> T {
        let node_l = self.tree[u].left;
        let node_r = self.tree[u].right;

        // 如果当前区间完全在查询区间内
        if query_l <= node_l && node_r <= query_r {
            return self.tree[u].max_val;
        }

        self.push_down(u); // 在查询子节点之前下推标记

        let mid = node_l + (node_r - node_l) / 2;
        let ls = 2 * u + 1;
        let rs = 2 * u + 2;
        let mut res = T::neg_infinity();

        if query_l <= mid {
            res = max(res, self._query_max(ls, query_l, query_r));
        }
        if query_r > mid {
            res = max(res, self._query_max(rs, query_l, query_r));
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_query_max_i64() {
        let data: [i64; 10] = [1, 5, 2, 8, 3, 9, 4, 6, 7, 0];
        let mut st = SegmentTree::new(&data);

        assert_eq!(st.query_max(0, 9), Some(9));
        assert_eq!(st.query_max(3, 6), Some(9)); // max of [8,3,9,4] is 9
        assert_eq!(st.query_max(0, 0), Some(1));
        assert_eq!(st.query_max(9, 9), Some(0));
        assert_eq!(st.query_max(1, 1), Some(5));
    }

    #[test]
    fn test_modify_set_and_query_max_i64() {
        let data: [i64; 5] = [10, 20, 5, 30, 15];
        let mut st = SegmentTree::new(&data);

        // 初始查询
        assert_eq!(st.query_max(0, 4), Some(30));

        // 修改 [1, 3] 为 7
        // data 变为 [10, 7, 7, 7, 15]
        st.modify_set(1, 3, 7);
        assert_eq!(st.query_max(0, 4), Some(15)); // max of [10,7,7,7,15]
        assert_eq!(st.query_max(1, 1), Some(7));
        assert_eq!(st.query_max(0, 1), Some(10)); // max of [10,7]
        assert_eq!(st.query_max(3, 4), Some(15)); // max of [7,15]

        // 修改整个数组为 1
        // data 变为 [1, 1, 1, 1, 1]
        st.modify_set(0, 4, 1);
        assert_eq!(st.query_max(0, 4), Some(1));
        assert_eq!(st.query_max(2, 3), Some(1));

        // 修改部分回较大的值
        // data 变为 [1, 1, 50, 50, 1]
        st.modify_set(2, 3, 50);
        assert_eq!(st.query_max(0, 4), Some(50));
        assert_eq!(st.query_max(0, 1), Some(1));
        assert_eq!(st.query_max(2, 2), Some(50));
        assert_eq!(st.query_max(4, 4), Some(1));
    }

    #[test]
    fn test_empty_and_single_element_i64() {
        let data_empty: [i64; 0] = [];
        let mut st_empty = SegmentTree::new(&data_empty);
        assert_eq!(st_empty.query_max(0, 0), None);
        st_empty.modify_set(0, 0, 10); // 不应 panic

        let data_single: [i64; 1] = [42];
        let mut st_single = SegmentTree::new(&data_single);
        assert_eq!(st_single.query_max(0, 0), Some(42));
        st_single.modify_set(0, 0, 10);
        assert_eq!(st_single.query_max(0, 0), Some(10));
    }

    #[test]
    fn test_overlapping_modifications_i64() {
        let data: [i64; 5] = [0, 0, 0, 0, 0];
        let mut st = SegmentTree::new(&data);

        st.modify_set(0, 2, 5); // [5,5,5,0,0]
        assert_eq!(st.query_max(0, 4), Some(5));
        assert_eq!(st.query_max(3, 4), Some(0));

        st.modify_set(2, 4, 10); // [5,5,10,10,10]
        assert_eq!(st.query_max(0, 4), Some(10));
        assert_eq!(st.query_max(0, 1), Some(5));
        assert_eq!(st.query_max(2, 2), Some(10));

        st.modify_set(1, 3, 2); // [5,2,2,2,10]
        assert_eq!(st.query_max(0, 4), Some(10));
        assert_eq!(st.query_max(0, 0), Some(5));
        assert_eq!(st.query_max(1, 3), Some(2));
        assert_eq!(st.query_max(4, 4), Some(10));
    }

     #[test]
    fn test_u64_type() {
        let data: [u64; 3] = [100, 20, 3000];
        let mut st = SegmentTree::new(&data);
        assert_eq!(st.query_max(0,2), Some(3000));
        st.modify_set(1,1, 5000); // [100, 5000, 3000]
        assert_eq!(st.query_max(0,2), Some(5000));
        assert_eq!(st.query_max(1,1), Some(5000));
    }
}