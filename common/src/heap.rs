//! Used for expressiong graph structures while avoiding grumblings of the borrow checker

use std::{fmt::Debug, marker::PhantomData};

pub trait UsizeId:
    From<usize> + Into<usize> + Eq + PartialOrd + Ord + std::hash::Hash + Copy + Default + Debug
{
}

#[macro_export]
macro_rules! define_usize_id {
    ($name:ident) => {
        #[derive(
            Debug,
            Clone,
            Copy,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            Default,
            serde::Serialize,
            serde::Deserialize,
        )]
        pub struct $name(usize);
        impl From<usize> for $name {
            fn from(value: usize) -> Self {
                Self(value)
            }
        }
        impl From<$name> for usize {
            fn from(value: $name) -> Self {
                value.0
            }
        }
        impl $crate::heap::UsizeId for $name {}
    };
}

#[derive(Debug, Clone, Default)]
pub struct IdAllocator<I>(usize, PhantomData<I>);

impl<I> IdAllocator<I> {
    pub fn new() -> Self {
        Self(0, PhantomData)
    }
}

impl<I: UsizeId> IdAllocator<I> {
    pub fn alloc(&mut self) -> I {
        let r = I::from(self.0);
        self.0 += 1;
        r
    }

    pub fn decompose<I2: UsizeId>(self) -> (IdAllocator<I2>, impl Fn(I) -> I2) {
        (IdAllocator(self.0, PhantomData), move |i: I| {
            let i: usize = i.into();
            assert!(i < self.0);
            i.into()
        })
    }

    pub fn n_allocated(&self) -> usize {
        self.0
    }

    pub fn allocated_ids(&self) -> impl Iterator<Item = I> {
        (0..self.n_allocated()).map(I::from)
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Heap<I, T>(pub Vec<T>, PhantomData<I>);

impl<I, T> std::fmt::Debug for Heap<I, T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.0.iter().enumerate()).finish()
    }
}

const PANIC_MSG: &'static str =
    "Assigned indices should never exceed vector length. Perhaps more than one heap is used?";

impl<I: UsizeId, T> std::ops::Index<I> for Heap<I, T> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        self.0.get(index.into()).expect(PANIC_MSG)
    }
}

impl<I: UsizeId, T> std::ops::IndexMut<I> for Heap<I, T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.get_mut(index.into()).expect(PANIC_MSG)
    }
}

impl<I: UsizeId, T> Heap<I, T> {
    pub fn push(&mut self, t: T) -> I {
        let id = self.0.len().into();
        self.0.push(t);
        id
    }
    pub fn iter_with_id(&self) -> impl Iterator<Item = (I, &T)> {
        self.0.iter().enumerate().map(|(i, x)| (i.into(), x))
    }
    pub fn into_iter_with_id(self) -> impl Iterator<Item = (I, T)> {
        self.0.into_iter().enumerate().map(|(i, x)| (i.into(), x))
    }
    pub fn map<I1, T1>(self, f: &mut impl FnMut(I, T) -> T1) -> Heap<I1, T1> {
        Heap(
            self.0
                .into_iter()
                .enumerate()
                .map(|(i, x)| f(i.into(), x))
                .collect(),
            PhantomData,
        )
    }
    pub fn map_by_ref<I1, T1>(&self, f: &mut impl FnMut(I, &T) -> T1) -> Heap<I1, T1> {
        Heap(
            self.0
                .iter()
                .enumerate()
                .map(|(i, x)| f(i.into(), x))
                .collect(),
            PhantomData,
        )
    }
    pub fn map_by_ref_result<I1, T1, E>(
        &self,
        f: &mut impl FnMut(I, &T) -> Result<T1, E>,
    ) -> Result<Heap<I1, T1>, E> {
        Ok(Heap(
            self.0
                .iter()
                .enumerate()
                .map(|(i, x)| f(i.into(), x))
                .collect::<Result<Vec<_>, _>>()?,
            PhantomData,
        ))
    }
    pub fn ids(&self) -> impl Iterator<Item = I> {
        (0..self.len()).map(|i| i.into())
    }
}

impl<I, T> Heap<I, T> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn new() -> Self {
        Self(Vec::new(), PhantomData)
    }
    pub fn repeat(t: T, n: usize) -> Self
    where
        T: Clone,
    {
        Self(vec![t; n], PhantomData)
    }

    pub fn repeat_with(f: impl Fn() -> T, n: usize) -> Self {
        Self((0..n).map(|_| f()).collect(), PhantomData)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }
    pub fn freeze(self) -> (RoHeap<I, T>, IdAllocator<I>) {
        let a = IdAllocator(self.0.len(), PhantomData);
        (RoHeap(self.0, PhantomData), a)
    }
}

impl<I, T> Default for Heap<I, T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct RoHeap<I, T>(Vec<T>, PhantomData<I>);

impl<I, T> Default for RoHeap<I, T> {
    fn default() -> Self {
        Self(vec![], PhantomData)
    }
}

impl<I: UsizeId, T> std::ops::Index<I> for RoHeap<I, T> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        self.0.get(index.into()).expect(PANIC_MSG)
    }
}

impl<I, T> RoHeap<I, T> {
    pub fn to_mutable(self, a: IdAllocator<I>) -> Heap<I, T> {
        if self.0.len() != a.0 {
            panic!("invalid id allocator");
        }
        Heap(self.0, PhantomData)
    }
}
