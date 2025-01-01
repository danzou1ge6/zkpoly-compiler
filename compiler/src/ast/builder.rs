use std::any;

use super::*;
use crate::external;

pub struct Builder {
    constants: external::ConstantTable,
    functions: external::FunctionTable,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            constants: external::ConstantTable::new(),
            functions: external::FunctionTable::new(),
        }
    }

    #[track_caller]
    pub fn entry(&self, name: String, typ: Typ) -> Vertex {
        VertexInner {
            node: VertexNode::Entry,
            typ: Some(typ),
            src: SourceInfo::new(*Location::caller(), name),
        }.into()
    }

    #[track_caller]
    pub fn hash_scalar(&self, transcript: Vertex, value: Vertex) -> Vertex {
        let name = transcript.inner().src.name.clone();
        VertexInner {
            node: VertexNode::HashTranscript {
                transcript,
                value,
                typ: HashTyp::HashScalar,
            },
            typ: Some(Typ::Transcript),
            src: SourceInfo::new(*Location::caller(), name),
        }.into()
    }

    #[track_caller]
    pub fn constant_scaler<T: 'static>(&mut self, name: String, value: T) -> Vertex {
        let constant_id = self.constants.push(external::Constant::new(
            name.clone(),
            Typ::Scalar,
            Box::new(value),
        ));
        VertexInner {
            node: VertexNode::Constant(constant_id),
            typ: Some(Typ::Scalar),
            src: SourceInfo::new(*Location::caller(), name),
        }.into()
    }

    #[track_caller]
    pub fn unpack_pair(
        &self,
        name1: String,
        name2: String,
        pair: Vertex,
    ) -> (Vertex, Vertex) {
        (
            VertexInner {
                node: VertexNode::TupleGet(pair.clone(), 0),
                typ: Some(Typ::Scalar),
                src: SourceInfo::new(*Location::caller(), name1),
            }.into(),
            VertexInner {
                node: VertexNode::TupleGet(pair, 0),
                typ: Some(Typ::Scalar),
                src: SourceInfo::new(*Location::caller(), name2),
            }.into(),
        )
    }

    #[track_caller]
    pub fn external_function(
        &mut self,
        name: String,
        f: external::Function,
        args: Vec<Vertex>,
    ) -> Vertex {
        let typ = f.ret_typ().clone();
        let function_id = self.functions.push(f);
        VertexInner {
            node: VertexNode::External(function_id, args),
            typ: Some(typ),
            src: SourceInfo::new(*Location::caller(), name),
        }.into()
    }

    #[track_caller]
    pub fn array(&self, name: String, typ: Typ, value: Vec<Vertex>) -> Vertex {
        let len = value.len();
        VertexInner {
            node: VertexNode::Array(value),
            typ: Some(Typ::Array(Box::new(typ), len)),
            src: SourceInfo::new(*Location::caller(), name),
        }.into()
    }

    #[track_caller]
    pub fn constant<T: 'static>(&mut self, name: String, value: T) -> Vertex {
        let typ = Typ::Any(any::TypeId::of::<T>(), std::mem::size_of::<T>());
        let constant_id = self.constants.push(external::Constant::new(
            name.clone(),
            typ.clone(),
            Box::new(value),
        ));
        VertexInner {
            node: VertexNode::Constant(constant_id),
            typ: Some(typ),
            src: SourceInfo::new(*Location::caller(), name),
        }.into()
    }
}

impl std::ops::Add for Vertex {
    type Output = Vertex;

    #[track_caller]
    fn add(self, rhs: Self) -> Self::Output {
        VertexInner {
            node: VertexNode::Arith(Arith::Bin(transit::ArithBinOp::Add, self, rhs)),
            typ: None,
            src: SourceInfo::new(*Location::caller(), "".to_string()),
        }.into()
    }
}

impl std::ops::Mul for Vertex {
    type Output = Vertex;

    #[track_caller]
    fn mul(self, rhs: Self) -> Self::Output {
        VertexInner {
            node: VertexNode::Arith(Arith::Bin(transit::ArithBinOp::Mul, self, rhs)),
            typ: None,
            src: SourceInfo::new(*Location::caller(), "".to_string()),
        }.into()
    }
}

impl std::ops::Sub for Vertex {
    type Output = Vertex;

    #[track_caller]
    fn sub(self, rhs: Self) -> Self::Output {
        VertexInner {
            node: VertexNode::Arith(Arith::Bin(transit::ArithBinOp::Sub, self, rhs)),
            typ: None,
            src: SourceInfo::new(*Location::caller(), "".to_string()),
        }.into()
    }
}

impl std::ops::Neg for Vertex {
    type Output = Vertex;

    #[track_caller]
    fn neg(self) -> Self::Output {
        VertexInner {
            node: VertexNode::Arith(Arith::Unr(transit::ArithUnrOp::Neg, self)),
            typ: None,
            src: SourceInfo::new(*Location::caller(), "".to_string()),
        }.into()
    }
}

impl std::ops::Div for Vertex {
    type Output = Vertex;

    #[track_caller]
    fn div(self, rhs: Self) -> Self::Output {
        VertexInner {
            node: VertexNode::Arith(Arith::Bin(transit::ArithBinOp::Div, self, rhs)),
            typ: None,
            src: SourceInfo::new(*Location::caller(), "".to_string()),
        }.into()
    }
}



