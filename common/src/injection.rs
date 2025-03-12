use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Default, Clone)]
pub struct Injection<T1, T2> {
    forward: BTreeMap<T1, T2>,
    backward: BTreeMap<T2, BTreeSet<T1>>,
}

impl<T1, T2> Injection<T1, T2>
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

    pub fn insert(&mut self, t1: T1, t2: T2) {
        if let Some(t2_old) = self.forward.remove(&t1) {
            self.backward.get_mut(&t2_old).unwrap().remove(&t1);
        }

        self.forward.insert(t1.clone(), t2.clone());
        self.backward.entry(t2).or_default().insert(t1);
    }

    pub fn remove(&mut self, t1: &T1) {
        let t2 = self.forward.remove(t1).unwrap();
        self.backward.get_mut(&t2).unwrap().remove(t1);
    }

    pub fn get_forward(&self, t1: &T1) -> Option<&T2> {
        self.forward.get(t1)
    }

    pub fn get_backward(&self, t2: &T2) -> Option<impl Iterator<Item = &T1> + Clone> {
        self.backward.get(t2).map(|s| s.iter())
    }
}
