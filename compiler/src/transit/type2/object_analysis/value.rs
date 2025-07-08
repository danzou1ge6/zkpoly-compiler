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
/// Objects belonging to the same [`ObjectId`] can be concatenated to a complete object,
/// which now is a polynomial
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Object {
    id: ObjectId,
    slice: Option<Slice>,
}

impl Object {
    pub fn new(object_id: ObjectId, slice: Option<Slice>) -> Self {
        Object {
            id: object_id,
            slice,
        }
    }

    pub fn not_sliced(object_id: ObjectId) -> Self {
        Object {
            id: object_id,
            slice: None,
        }
    }

    pub fn id(&self) -> ObjectId {
        self.id
    }

    pub fn with_object_id(&self, object_id: ObjectId) -> Self {
        Object {
            id: object_id,
            slice: self.slice.clone(),
        }
    }
}

/// Represents what we know now about what's inside a runtime register
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Value<O, N> {
    /// The object it points to
    object: O,
    /// On which memory device the object resides.
    /// For [`super::Operation`]'s, this is determined by execution device of the operation.
    device: Device,
    /// Variant of the value
    node: N,
}

impl<O, N> Value<O, N> {
    pub fn new(object: O, device: Device, node: N) -> Self {
        Value {
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
        Value {
            node: self.node.clone(),
            object: self.object.clone(),
            device,
        }
    }

    pub fn with_object(&self, object: O) -> Self
    where
        N: Clone,
    {
        Value {
            object,
            device: self.device,
            node: self.node.clone(),
        }
    }

    pub fn with_node(&self, node: N) -> Self
    where
        O: Clone,
    {
        Value {
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

impl<N> Value<Object, N> {
    pub fn with_object_id(&self, object_id: ObjectId) -> Self
    where
        N: Clone,
    {
        Value {
            object: self.object.with_object_id(object_id),
            device: self.device,
            node: self.node.clone(),
        }
    }

    pub fn object_id(&self) -> ObjectId {
        self.object.id()
    }
}

impl<O> Value<O, ValueNode> {
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
pub struct OutputValue<V, I>(V, Option<I>);

impl<V, I> std::ops::Deref for OutputValue<V, I> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V, I> std::ops::DerefMut for OutputValue<V, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<V, I> OutputValue<V, I> {
    pub fn new(value: V, inplace: Option<I>) -> Self {
        OutputValue(value, inplace)
    }

    pub fn non_inplace(value: V) -> Self {
        OutputValue(value, None)
    }

    pub fn inplace_of(&self) -> Option<I>
    where
        I: Clone,
    {
        self.1.clone()
    }
}

/// A Type2 Vertex can either output a single value or a tuple of values.
/// They are distinguished here to recognize subsequent TupleGet's.
#[derive(Debug, Clone)]
pub enum VertexOutput<V> {
    Tuple(Vec<V>),
    Single(V),
}

impl<V> VertexOutput<V> {
    pub fn object_ids<'s, N>(&'s self) -> Box<dyn Iterator<Item = ObjectId> + 's>
    where
        V: Deref<Target = Value<Object, N>>,
    {
        use VertexOutput::*;
        match self {
            Tuple(ss) => Box::new(ss.iter().map(|x| x.deref().object_id())),
            Single(s) => Box::new([s.deref().object_id()].into_iter()),
        }
    }

    pub fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &'a V> + 'a> {
        use VertexOutput::*;
        match self {
            Tuple(ss) => Box::new(ss.iter()),
            Single(s) => Box::new([s].into_iter()),
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut V> + 'a> {
        use VertexOutput::*;
        match self {
            Tuple(ss) => Box::new(ss.iter_mut()),
            Single(s) => Box::new([s].into_iter()),
        }
    }

    pub fn try_unwrap_single(&self) -> Option<&V> {
        use VertexOutput::*;
        match self {
            Single(s) => Some(s),
            _ => None,
        }
    }

    pub fn unwrap_single(&self) -> &V {
        use VertexOutput::*;
        match self {
            Single(s) => s,
            _ => panic!("called unwrap_single on VertexValue::Tuple"),
        }
    }

    pub fn unwrap_single_mut(&mut self) -> &mut V {
        use VertexOutput::*;
        match self {
            Single(s) => s,
            _ => panic!("called unwrap_single on VertexValue::Tuple"),
        }
    }

    pub fn unwrap_tuple(&self) -> &Vec<V> {
        use VertexOutput::*;
        match self {
            Tuple(v) => v,
            _ => panic!("called unwrap_tuple on VertexValue::Single"),
        }
    }
}

impl<O, N, I> VertexOutput<OutputValue<Value<O, N>, I>>
where
    O: Clone,
    N: Clone,
{
    pub fn with_no_inplace(self) -> Self {
        use VertexOutput::*;
        match self {
            Tuple(ss) => VertexOutput::Tuple(
                ss.iter()
                    .map(|OutputValue(v, _)| OutputValue(v.clone(), None))
                    .collect(),
            ),
            Single(OutputValue(v, _)) => VertexOutput::Single(OutputValue(v.clone(), None)),
        }
    }

    pub fn with_device(&self, device: Device) -> Self
    where
        I: Clone,
    {
        use VertexOutput::*;
        match self {
            Tuple(ss) => VertexOutput::Tuple(
                ss.iter()
                    .map(|OutputValue(v, inplace)| {
                        OutputValue(v.with_device(device), inplace.as_ref().cloned())
                    })
                    .collect(),
            ),
            Single(OutputValue(v, inplace)) => VertexOutput::Single(OutputValue(
                v.with_device(device),
                inplace.as_ref().cloned(),
            )),
        }
    }
}

/// One of the input values to a vertex.
/// The ordering doesn't matter, it's just for using it as BTreeMap index.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum VertexInput<V> {
    Single(V, Mutability),
    Tuple(Vec<V>),
}

/// This is a meaning less default. Just to make the arith subgraph API work.
impl<V> Default for VertexInput<V> {
    fn default() -> Self {
        Self::Tuple(Vec::new())
    }
}

impl<V> VertexInput<V> {
    pub fn unwrap_single(&self) -> (&V, &Mutability) {
        use VertexInput::*;
        match self {
            Single(v, m) => (v, m),
            _ => panic!("called unwrap_single on VertexInput::Tuple"),
        }
    }

    pub fn unwrap_single_mut(&mut self) -> (&mut V, &mut Mutability) {
        use VertexInput::*;
        match self {
            Single(v, m) => (v, m),
            _ => panic!("called unwrap_single on VertexInput::Tuple"),
        }
    }

    pub fn v_into<V1>(self) -> VertexInput<V1>
    where
        V1: From<V>,
    {
        use VertexInput::*;
        match self {
            Single(v, m) => Single(V1::from(v), m),
            Tuple(tuple) => Tuple(tuple.into_iter().map(|v| V1::from(v)).collect()),
        }
    }

    pub fn try_map_v<V1, Er>(
        self,
        mut f: impl FnMut(V) -> Result<V1, Er>,
    ) -> Result<VertexInput<V1>, Er> {
        use VertexInput::*;
        Ok(match self {
            Single(v, m) => Single(f(v)?, m),
            Tuple(tuple) => Tuple(tuple.into_iter().map(f).collect::<Result<_, _>>()?),
        })
    }

    pub fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &'a V> + 'a> {
        use VertexInput::*;
        match self {
            Single(v, _) => Box::new(std::iter::once(v)),
            Tuple(tuple) => Box::new(tuple.iter()),
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut V> + 'a> {
        use VertexInput::*;
        match self {
            Single(v, _) => Box::new(std::iter::once(v)),
            Tuple(tuple) => Box::new(tuple.iter_mut()),
        }
    }

    pub fn mutable(&self) -> Option<&V> {
        match self {
            Self::Single(v, Mutability::Mut) => Some(v),
            _ => None,
        }
    }

    pub fn is_mutable(&self) -> bool {
        self.mutable().is_some()
    }

    pub fn single_mutable(v: V) -> Self {
        VertexInput::Single(v, Mutability::Mut)
    }
}

impl<N> VertexInput<Value<Object, N>>
where
    N: Clone,
{
    pub fn with_object_id(&self, object_id: ObjectId) -> Self {
        use VertexInput::*;
        match self {
            Single(v, m) => Single(v.with_object_id(object_id), *m),
            Tuple(tuple) => Tuple(tuple.iter().map(|v| v.with_object_id(object_id)).collect()),
        }
    }
}
