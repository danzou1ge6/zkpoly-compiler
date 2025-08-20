use std::ops::Deref;

use crate::transit::type3::Device;
use zkpoly_common::arith::Mutability;
use zkpoly_common::define_usize_id;
use zkpoly_common::typ::{template::Typ, Slice};
use zkpoly_runtime::args::RuntimeType;

define_usize_id!(ObjectId);

impl From<&ObjectId> for ObjectId {
    fn from(value: &ObjectId) -> Self {
        value.clone()
    }
}

pub type ValueNode = Typ<Slice>;

/// An object is an immutable piece of data on some storage device.
/// Objects that are slices belonging to the same [`ObjectId`] can be concatenated to a complete object,
/// If an object is part of a polynomial, `slice` must be non-None
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Object {
    id: ObjectId,
    node: ValueNode,
}

impl Object {
    pub fn new(object_id: ObjectId, node: ValueNode) -> Self {
        Object {
            id: object_id,
            node,
        }
    }

    pub fn id(&self) -> ObjectId {
        self.id
    }

    pub fn with_object_id(&self, object_id: ObjectId) -> Self {
        Object {
            id: object_id,
            node: self.node.clone(),
        }
    }
}

/// Represents what we know now about what's inside a runtime register
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Atom<O, N> {
    /// The object it points to
    object: O,
    /// On which memory device the object resides.
    /// For [`super::Operation`]'s, this is determined by execution device of the operation.
    device: Device,
    /// Variant of the value
    node: N,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tree<S> {
    Single(S),
    Tuple(Vec<Tree<S>>),
}

impl<S> Default for Tree<S> {
    fn default() -> Self {
        Tree::Tuple(Vec::new())
    }
}

pub mod tree_iter {
    use super::Tree;

    pub struct Iter<'t, S> {
        pub(super) stack: Vec<&'t Tree<S>>,
    }

    impl<'t, S> Iterator for Iter<'t, S> {
        type Item = &'t S;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                let top = self.stack.pop()?;
                match top {
                    Tree::Single(x) => return Some(x),
                    Tree::Tuple(xs) => xs.iter().for_each(|x| self.stack.push(x)),
                }
            }
        }
    }

    pub struct IterMut<'t, S> {
        pub(super) stack: Vec<&'t mut Tree<S>>,
    }

    impl<'t, S> Iterator for IterMut<'t, S> {
        type Item = &'t mut S;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                let top = self.stack.pop()?;
                match top {
                    Tree::Single(x) => return Some(x),
                    Tree::Tuple(xs) => xs.iter_mut().for_each(|x| self.stack.push(x)),
                }
            }
        }
    }
}

impl<S> Tree<S> {
    pub fn map_ref<S1>(&self, mut f: impl FnMut(&S) -> S1) -> Tree<S1> {
        match self {
            Tree::Single(s) => Tree::Single(f(s)),
            Tree::Tuple(vs) => Tree::Tuple(vs.iter().map(|x| x.map_ref(&mut f)).collect()),
        }
    }

    pub fn iter1<'a>(&'a self) -> Box<dyn Iterator<Item = &'a Tree<S>> + 'a> {
        match self {
            Tree::Single(..) => Box::new(std::iter::once(self)),
            Tree::Tuple(xs) => Box::new(xs.iter()),
        }
    }

    pub fn iter1_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Tree<S>> + 'a> {
        match self {
            Tree::Single(..) => Box::new(std::iter::once(self)),
            Tree::Tuple(xs) => Box::new(xs.iter_mut()),
        }
    }

    pub fn unwrap_leaf_mut(&mut self) -> &mut S {
        match self {
            Tree::Single(x) => x,
            _ => panic!("called unwrap_leaf_mut on Tuple node"),
        }
    }

    pub fn unwrap_leaf(&self) -> &S {
        match self {
            Tree::Single(x) => x,
            _ => panic!("called unwrap_leaf on Tuple node"),
        }
    }

    pub fn unwrap_tuple(&self) -> &[Tree<S>] {
        match self {
            Tree::Single(..) => panic!("called unwrap_tuple on Single node"),
            Tree::Tuple(xs) => xs.as_slice(),
        }
    }

    pub fn iter<'a>(&'a self) -> tree_iter::Iter<'a, S> {
        tree_iter::Iter { stack: vec![self] }
    }

    pub fn iter_mut<'a>(&'a mut self) -> tree_iter::IterMut<'a, S> {
        tree_iter::IterMut { stack: vec![self] }
    }
}

impl<O, N> Atom<O, N> {
    pub fn new(object: O, device: Device, node: N) -> Self {
        Atom {
            object,
            device,
            node,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn object(&self) -> &O {
        &self.object
    }
    pub fn object_mut(&mut self) -> &mut O {
        &mut self.object
    }

    pub fn with_device(&self, device: Device) -> Self
    where
        N: Clone,
        O: Clone,
    {
        Atom {
            node: self.node.clone(),
            object: self.object.clone(),
            device,
        }
    }

    pub fn with_object(&self, object: O) -> Self
    where
        N: Clone,
    {
        Atom {
            object,
            device: self.device,
            node: self.node.clone(),
        }
    }

    pub fn with_node(&self, node: N) -> Self
    where
        O: Clone,
    {
        Atom {
            object: self.object.clone(),
            device: self.device,
            node,
        }
    }

    pub fn node(&self) -> &N {
        &self.node
    }

    pub fn node_mut(&mut self) -> &mut N {
        &mut self.node
    }
}

impl<N> Atom<Object, N> {
    pub fn with_object_id(&self, object_id: ObjectId) -> Self
    where
        N: Clone,
    {
        Atom {
            object: self.object.with_object_id(object_id),
            device: self.device,
            node: self.node.clone(),
        }
    }

    pub fn object_id(&self) -> ObjectId {
        self.object.id()
    }
}

impl<O> Atom<O, ValueNode> {
    pub fn object_size<Rt: RuntimeType>(&self) -> usize {
        use std::mem::size_of;
        match &self.node {
            Typ::ScalarArray { len, .. } => len * size_of::<Rt::Field>(),
            Typ::Scalar => size_of::<Rt::Field>(),
            Typ::Transcript => size_of::<Rt::Trans>(),
            Typ::PointBase { len } => len * size_of::<Rt::PointAffine>(),
            Typ::Point => size_of::<Rt::PointAffine>(),
            Typ::Tuple => panic!("a value should never point to a tuple"),
            Typ::Stream => panic!("a value should never point to a stream"),
            Typ::GpuBuffer(len, ..) => *len,
            Typ::Any(_, len) => *len,
        }
    }

    /// Try rotate the value, which must be a [`Typ::ScalarArray`].
    /// If the value is a rotated polynomial, i.e., slice length equals data length,
    /// then it is rotated by `delta`.
    /// Otherwise, [`None`] is returned.
    ///
    /// # Panics
    /// When the value is not a [`Typ::ScalarArray`].
    pub fn rotate(&self, delta: i32) -> Option<Self>
    where
        O: Clone,
    {
        match &self.node {
            Typ::ScalarArray { len, meta } => {
                if meta.len() == *len as u64 {
                    let offset = (meta.begin() as i64 + delta as i64).rem_euclid(*len as i64);
                    Some(self.with_node(Typ::ScalarArray {
                        len: *len,
                        meta: Slice::new(offset as u64, meta.len()),
                    }))
                } else {
                    None
                }
            }
            _ => panic!("can only rotate a scalar array"),
        }
    }

    /// Slice the value, which must be a [`Typ::ScalarArray`].
    /// A slice is still continuous in memory after slicing it, so this function always success.
    ///
    /// # Panics
    /// When the value is not a [`Typ::ScalarArray`].
    pub fn slice(&self, begin: u64, end: u64) -> Self
    where
        O: Clone,
    {
        match &self.node {
            Typ::ScalarArray { len: deg, meta } => {
                let offset = (meta.begin() + begin) % *deg as u64;
                self.with_node(Typ::ScalarArray {
                    len: *deg as usize,
                    meta: Slice::new(offset, end - begin),
                })
            }
            _ => panic!("can only slice a scalar array"),
        }
    }

    /// Check if the value is a sliced polynomial
    pub fn is_sliced(&self) -> bool {
        match &self.node {
            Typ::ScalarArray { len: deg, meta } => {
                *deg as usize != meta.len() as usize || meta.begin() != 0
            }
            _ => false,
        }
    }
}

/// One of the values outputed by a vertex.
/// If `.1` is [`Some`], it is the index of the input value taken inplace.
/// A input value is said to be taken inplace if the underlying memory space is then used by the output.
#[derive(Debug, Clone)]
pub struct OutputT<V, I> {
    v: V,
    inplace_of: Option<I>,
}

impl<V, I> std::ops::Deref for OutputT<V, I> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<V, I> std::ops::DerefMut for OutputT<V, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

impl<V, I> OutputT<V, I> {
    pub fn new(value: V, inplace: Option<I>) -> Self {
        OutputT {
            v: value,
            inplace_of: inplace,
        }
    }

    pub fn non_inplace(value: V) -> Self {
        OutputT {
            v: value,
            inplace_of: None,
        }
    }

    pub fn with_non_inplace(&self) -> Self
    where
        V: Clone,
    {
        OutputT {
            v: self.v.clone(),
            inplace_of: None,
        }
    }

    pub fn inplace_of(&self) -> Option<I>
    where
        I: Clone,
    {
        self.inplace_of.clone()
    }

    pub fn inplace_of_mut(&mut self) -> &mut Option<I> {
        &mut self.inplace_of
    }
}

pub type OutputValue<V, I> = Tree<OutputT<V, I>>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct InputT<V> {
    v: V,
    mutability: Mutability,
}

impl<V> InputT<V> {
    pub fn new(v: V, m: Mutability) -> Self {
        Self { v, mutability: m }
    }

    pub fn immutable(v: V) -> Self {
        Self::new(v, Mutability::Const)
    }

    pub fn mutability(&self) -> Mutability {
        self.mutability
    }
}

impl<V> std::ops::Deref for InputT<V> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<V> std::ops::DerefMut for InputT<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

pub type InputValue<V> = Tree<InputT<V>>;
