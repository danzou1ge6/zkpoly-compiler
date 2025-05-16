use crate::transit::type3::{self, Device};
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

/// Represents what we know now about what's inside a runtime register
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Value {
    /// The object it points to
    object_id: ObjectId,
    /// On which memory device the object resides.
    /// For [`super::Operation`]'s, this is determined by execution device of the operation.
    device: Device,
    /// Variant of the value
    node: ValueNode,
}

impl Value {
    pub fn new(object_id: ObjectId, device: Device, node: ValueNode) -> Self {
        Value {
            object_id,
            device,
            node,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn object_id(&self) -> ObjectId {
        self.object_id
    }

    pub fn object_id_mut(&mut self) -> &mut ObjectId {
        &mut self.object_id
    }

    pub fn with_object_id(&self, object_id: ObjectId) -> Self {
        Value {
            object_id,
            device: self.device,
            node: self.node.clone(),
        }
    }

    pub fn with_device(&self, device: Device) -> Self {
        Value {
            node: self.node.clone(),
            object_id: self.object_id,
            device,
        }
    }

    pub fn with_node(&self, node: ValueNode) -> Self {
        Value {
            object_id: self.object_id,
            device: self.device,
            node,
        }
    }

    pub fn node(&self) -> &ValueNode {
        &self.node
    }

    pub fn node_mut(&mut self) -> &mut ValueNode {
        &mut self.node
    }

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
            Typ::GpuBuffer(len) => *len,
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
    pub fn rotate(&self, delta: i32) -> Option<Self> {
        match &self.node {
            Typ::ScalarArray { len, meta } => {
                if meta.len() == *len as u64 {
                    let offset = (meta.begin() as i64 + delta as i64) % *len as i64;
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
    pub fn slice(&self, begin: u64, end: u64) -> Self {
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
            Typ::ScalarArray { len: deg, meta } => *deg as usize != meta.len() as usize,
            _ => false,
        }
    }
}

/// One of the values outputed by a vertex.
/// If `.1` is [`Some`], it is the index of the input value taken inplace.
/// A input value is said to be taken inplace if the underlying memory space is then used by the output.
#[derive(Debug, Clone)]
pub struct OutputValue(Value, Option<usize>);

impl std::ops::Deref for OutputValue {
    type Target = Value;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl OutputValue {
    pub fn new(value: Value, inplace: Option<usize>) -> Self {
        OutputValue(value, inplace)
    }

    pub fn non_inplace(value: Value) -> Self {
        OutputValue(value, None)
    }
}

/// A Type2 Vertex can either output a single value or a tuple of values.
/// They are distinguished here to recognize subsequent TupleGet's.
#[derive(Debug, Clone)]
pub enum VertexOutput {
    Tuple(Vec<OutputValue>),
    Single(OutputValue),
}

impl VertexOutput {
    pub fn object_ids<'s>(&'s self) -> Box<dyn Iterator<Item = ObjectId> + 's> {
        use VertexOutput::*;
        match self {
            Tuple(ss) => Box::new(ss.iter().map(|x| x.object_id())),
            Single(s) => Box::new([s.object_id()].into_iter()),
        }
    }

    pub fn with_device(&self, device: Device) -> Self {
        use VertexOutput::*;
        match self {
            Tuple(ss) => VertexOutput::Tuple(
                ss.iter()
                    .map(|OutputValue(v, inplace)| OutputValue(v.with_device(device), *inplace))
                    .collect(),
            ),
            Single(OutputValue(v, inplace)) => {
                VertexOutput::Single(OutputValue(v.with_device(device), *inplace))
            }
        }
    }

    pub fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &'a OutputValue> + 'a> {
        use VertexOutput::*;
        match self {
            Tuple(ss) => Box::new(ss.iter()),
            Single(s) => Box::new([s].into_iter()),
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut OutputValue> + 'a> {
        use VertexOutput::*;
        match self {
            Tuple(ss) => Box::new(ss.iter_mut()),
            Single(s) => Box::new([s].into_iter()),
        }
    }

    pub fn try_unwrap_single(&self) -> Option<&OutputValue> {
        use VertexOutput::*;
        match self {
            Single(s) => Some(s),
            _ => None,
        }
    }

    pub fn unwrap_single(&self) -> &OutputValue {
        use VertexOutput::*;
        match self {
            Single(s) => s,
            _ => panic!("called unwrap_single on VertexValue::Tuple"),
        }
    }

    pub fn unwrap_single_mut(&mut self) -> &mut OutputValue {
        use VertexOutput::*;
        match self {
            Single(s) => s,
            _ => panic!("called unwrap_single on VertexValue::Tuple"),
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

impl VertexInput<Value> {
    pub fn with_object_id(&self, object_id: ObjectId) -> Self {
        use VertexInput::*;
        match self {
            Single(v, m) => Single(v.with_object_id(object_id), *m),
            Tuple(tuple) => Tuple(tuple.iter().map(|v| v.with_object_id(object_id)).collect()),
        }
    }
}
