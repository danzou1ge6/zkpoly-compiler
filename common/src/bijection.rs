use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct Bijection<T1, T2>
{
    forward: BTreeMap<T1, T2>,
    backward: BTreeMap<T2, T1>,
}

impl<T1, T2> Bijection<T1, T2>
where
    T1: Ord + Clone,
    T2: Ord + Clone,
{
    pub fn new() -> Self {
        Self {
            forward: BTreeMap::new(),
            backward: BTreeMap::new(),
        }
    }

    pub fn insert_checked(&mut self, a: T1, b: T2)
    where
        T1: std::fmt::Debug,
        T2: std::fmt::Debug,
    {
        if let Some(b_old) = self.forward.get(&a) {
            panic!("insertion failed: {:?} already mapped to {:?}", a, b_old)
        }

        if let Some(a_old) = self.backward.get(&b) {
            panic!("insertion failed: {:?} already mapped to {:?}", b, a_old)
        }

        self.forward.insert(a.clone(), b.clone());
        self.backward.insert(b, a);
    }

    pub fn insert(&mut self, a: T1, b: T2) {
        if let Some(b_old) = self.forward.get(&a) {
            self.backward.remove(b_old);
        }

        if let Some(a_old) = self.backward.get(&b) {
            self.forward.remove(a_old);
        }

        self.forward.insert(a.clone(), b.clone());
        self.backward.insert(b, a);
    }

    pub fn remove_forward(&mut self, a: &T1) -> Option<T2> {
        self.forward.remove(a).map(|b| {
            self.backward.remove(&b);
            b
        })
    }

    pub fn remove_backward(&mut self, b: &T2) -> Option<T1> {
        self.backward.remove(b).map(|a| {
            self.forward.remove(&a);
            a
        })
    }

    pub fn get_forward(&self, a: &T1) -> Option<&T2> {
        self.forward.get(a)
    }

    pub fn get_backward(&self, b: &T2) -> Option<&T1> {
        self.backward.get(b)
    }

    pub fn len(&self) -> usize {
        self.forward.len()
    }

    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&T1, &T2)> {
        self.forward.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_operations() {
        let mut bijection = Bijection::new();

        // 测试插入和获取
        bijection.insert(1, "a");
        assert_eq!(bijection.get_forward(&1), Some(&"a"));
        assert_eq!(bijection.get_backward(&"a"), Some(&1));

        // 测试覆盖插入
        bijection.insert(1, "b");
        assert_eq!(bijection.get_backward(&"a"), None);
        assert_eq!(bijection.get_backward(&"b"), Some(&1));

        // 测试双向删除
        assert_eq!(bijection.remove_forward(&1), Some("b"));
        assert!(bijection.is_empty());
    }

    #[test]
    fn cross_removal() {
        let mut bijection = Bijection::new();
        bijection.insert(1, "a");
        bijection.insert(2, "b");

        // 测试通过反向键删除
        assert_eq!(bijection.remove_backward(&"a"), Some(1));
        assert_eq!(bijection.len(), 1);
        assert_eq!(bijection.get_forward(&2), Some(&"b"));
    }

    #[test]
    fn complex_insert() {
        let mut bijection = Bijection::new();
        bijection.insert(1, "a");
        bijection.insert(2, "b");

        // 插入冲突键值对 (1, "b")
        // 预期行为：
        // 1. 移除旧映射 1 -> "a"
        // 2. 移除旧映射 "b" -> 2
        bijection.insert(1, "b");
        assert_eq!(bijection.get_forward(&2), None);
        assert_eq!(bijection.get_backward(&"a"), None);
        assert_eq!(bijection.get_forward(&1), Some(&"b"));
        assert_eq!(bijection.get_backward(&"b"), Some(&1));
    }
}
