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

#[derive(Debug, Clone, Copy)] // 让 MaxInfo 可以被复制
pub struct MaxInfo<T: SegmentValue> {
    pub value: T,
    pub index: usize, // 0-indexed
}

impl<T: SegmentValue> Default for MaxInfo<T> {
    fn default() -> Self {
        MaxInfo {
            value: T::neg_infinity(),
            index: usize::MAX, // 使用一个无效索引作为默认值
        }
    }
}

#[derive(Debug, Clone)]
struct Node<T: SegmentValue> {
    left: usize,        // 区间左端点 (0-indexed)
    right: usize,       // 区间右端点 (0-indexed)
    max_info: MaxInfo<T>, // 区间最大值及其索引
    set_tag: Option<T>, // 区间赋值的懒标记: node.val = tag_val
}

impl<T: SegmentValue> Default for Node<T> {
    fn default() -> Self {
        Node {
            left: 0,
            right: 0,
            max_info: MaxInfo::default(),
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
            self.tree[u].max_info = MaxInfo { value: data[l], index: l };
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
        if self.tree[ls].max_info.value >= self.tree[rs].max_info.value {
            self.tree[u].max_info = self.tree[ls].max_info;
        } else {
            self.tree[u].max_info = self.tree[rs].max_info;
        }
    }

    // 应用赋值标记到节点 u
    fn apply_set_tag(&mut self, u: usize, tag_val: T) {
        self.tree[u].max_info = MaxInfo { value: tag_val, index: self.tree[u].left }; // 区间赋值后，最大值索引默认为区间左端点
        self.tree[u].set_tag = Some(tag_val);
    }

    // 下传标记
    fn push_down(&mut self, u: usize) {
        if self.tree[u].left == self.tree[u].right {
            self.tree[u].set_tag = None;
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
            return;
        }
        self._modify_set(0, query_l, query_r, val);
    }

    fn _modify_set(&mut self, u: usize, query_l: usize, query_r: usize, val: T) {
        let node_l = self.tree[u].left;
        let node_r = self.tree[u].right;

        if query_l <= node_l && node_r <= query_r {
            self.apply_set_tag(u, val);
            return;
        }

        self.push_down(u);

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

    /// 查询区间 [query_l, query_r] 内的最大值及其最左边的索引
    pub fn query_max_info(&mut self, query_l: usize, query_r: usize) -> Option<MaxInfo<T>> {
        if self.tree.is_empty() || query_l > query_r || query_r >= self.data_len {
            return None;
        }
        Some(self._query_max_info(0, query_l, query_r))
    }

    fn _query_max_info(&mut self, u: usize, query_l: usize, query_r: usize) -> MaxInfo<T> {
        let node_l = self.tree[u].left;
        let node_r = self.tree[u].right;

        if query_l <= node_l && node_r <= query_r {
            return self.tree[u].max_info;
        }

        self.push_down(u);

        let mid = node_l + (node_r - node_l) / 2;
        let ls = 2 * u + 1;
        let rs = 2 * u + 2;
        let mut res_info = MaxInfo::default();

        if query_l <= mid {
            let left_info = self._query_max_info(ls, query_l, query_r);
            if left_info.value >= res_info.value { //  >= 保证取最左边的索引
                res_info = left_info;
            }
        }
        if query_r > mid {
            let right_info = self._query_max_info(rs, query_l, query_r);
            if right_info.value > res_info.value {
                 res_info = right_info;
            } else if right_info.value == res_info.value && right_info.index < res_info.index { // 如果值相等，取更小的索引
                 res_info = right_info;
            }
        }
        res_info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_query_max_info_i64() {
        let data: [i64; 10] = [1, 5, 2, 8, 3, 9, 4, 6, 7, 0];
        let mut st = SegmentTree::new(&data);

        let info1 = st.query_max_info(0, 9).unwrap();
        assert_eq!(info1.value, 9);
        assert_eq!(info1.index, 5);

        let info2 = st.query_max_info(3, 6).unwrap(); // data[3..=6] is [8,3,9,4]
        assert_eq!(info2.value, 9);
        assert_eq!(info2.index, 5);

        let info3 = st.query_max_info(0, 0).unwrap();
        assert_eq!(info3.value, 1);
        assert_eq!(info3.index, 0);

        let data2: [i64; 5] = [5, 5, 3, 5, 2];
        let mut st2 = SegmentTree::new(&data2);
        let info4 = st2.query_max_info(0, 4).unwrap();
        assert_eq!(info4.value, 5);
        assert_eq!(info4.index, 0); // 最左边的5

        let info5 = st2.query_max_info(1, 3).unwrap(); // data[1..=3] is [5,3,5]
        assert_eq!(info5.value, 5);
        assert_eq!(info5.index, 1); // data[1] is 5
    }

    #[test]
    fn test_modify_set_and_query_max_info_i64() {
        let data: [i64; 5] = [10, 20, 5, 30, 15];
        let mut st = SegmentTree::new(&data);

        let info1 = st.query_max_info(0, 4).unwrap();
        assert_eq!(info1.value, 30);
        assert_eq!(info1.index, 3);

        st.modify_set(1, 3, 7); // data 变为 [10, 7, 7, 7, 15]
        let info2 = st.query_max_info(0, 4).unwrap();
        assert_eq!(info2.value, 15);
        assert_eq!(info2.index, 4);

        let info3 = st.query_max_info(1, 1).unwrap();
        assert_eq!(info3.value, 7);
        assert_eq!(info3.index, 1); // apply_set_tag 将 index 设置为区间的 left

        st.modify_set(0, 4, 50); // data 变为 [50,50,50,50,50]
        let info4 = st.query_max_info(0,4).unwrap();
        assert_eq!(info4.value, 50);
        assert_eq!(info4.index, 0); // apply_set_tag 将 index 设置为区间的 left

        let info5 = st.query_max_info(2,3).unwrap();
        assert_eq!(info5.value, 50);
        assert_eq!(info5.index, 2); // apply_set_tag 将 index 设置为区间的 left
    }

    #[test]
    fn test_empty_and_single_element_i64() {
        let data_empty: [i64; 0] = [];
        let mut st_empty = SegmentTree::new(&data_empty);
        assert!(st_empty.query_max_info(0, 0).is_none());
        st_empty.modify_set(0, 0, 10); // 不应 panic

        let data_single: [i64; 1] = [42];
        let mut st_single = SegmentTree::new(&data_single);
        let info1 = st_single.query_max_info(0, 0).unwrap();
        assert_eq!(info1.value, 42);
        assert_eq!(info1.index, 0);

        st_single.modify_set(0, 0, 10);
        let info2 = st_single.query_max_info(0, 0).unwrap();
        assert_eq!(info2.value, 10);
        assert_eq!(info2.index, 0);
    }

    #[test]
    fn test_set_tag_index_logic() {
        let data: [i64; 5] = [1,2,3,4,5];
        let mut st = SegmentTree::new(&data);
        st.modify_set(1,3,10); // [1, 10, 10, 10, 5]
                               // Node for [1,3] will have max_info {value: 10, index: 1}
        let info = st.query_max_info(1,3).unwrap();
        assert_eq!(info.value, 10);
        assert_eq!(info.index, 1); // Querying the exact modified range

        let info_overall = st.query_max_info(0,4).unwrap(); // Max of [1,10,10,10,5]
        assert_eq!(info_overall.value, 10);
        assert_eq!(info_overall.index, 1); // Index 1 is the first 10
    }
}