use core::panic;
use std::any;
use zkpoly_runtime::args::{Constant, ConstantTable, RuntimeType, Variable};
use zkpoly_runtime::functions as uf;

use super::*;

pub struct Builder<T: RuntimeType> {
    constants: ConstantTable<T>,
    functions: uf::FunctionTable,
}

impl<T: RuntimeType> Builder<T> {
    pub fn new() -> Self {
        Self {
            constants: ConstantTable::new(),
            functions: uf::FunctionTable::new(),
        }
    }

    #[track_caller]
    pub fn entry(&self, name: String, typ: Typ) -> Vertex {
        VertexInner {
            node: VertexNode::Entry,
            typ: Some(typ),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
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
        }
        .into()
    }

    #[track_caller]
    pub fn hash_point(&self, transcript: Vertex, value: Vertex) -> Vertex {
        let name = transcript.inner().src.name.clone();
        VertexInner {
            node: VertexNode::HashTranscript {
                transcript,
                value,
                typ: HashTyp::HashPoint,
            },
            typ: Some(Typ::Transcript),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn constant_scaler(&mut self, name: String, value: Variable<T>) -> Vertex {
        let constant_id = self.constants.push(Constant::new(name.clone(), value));
        VertexInner {
            node: VertexNode::Constant(constant_id),
            typ: Some(Typ::Scalar),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn constant_poly<T: 'static>(
        &mut self,
        name: String,
        repr: PolyRepr,
        value: Vec<T>,
    ) -> Vertex {
        let constant_id = self.constants.push(Constant::new(
            name.clone(),
            Typ::Poly(repr.clone()),
            Box::new(value),
        ));
        VertexInner {
            node: VertexNode::Constant(constant_id),
            typ: Some(Typ::Poly(repr)),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn unpack_pair(
        &self,
        name1: String,
        typ1: Typ,
        name2: String,
        typ2: Typ,
        pair: Vertex,
    ) -> (Vertex, Vertex) {
        (
            VertexInner {
                node: VertexNode::TupleGet(pair.clone(), 0),
                typ: Some(typ1),
                src: SourceInfo::new(*Location::caller(), name1),
            }
            .into(),
            VertexInner {
                node: VertexNode::TupleGet(pair, 0),
                typ: Some(typ2),
                src: SourceInfo::new(*Location::caller(), name2),
            }
            .into(),
        )
    }

    #[track_caller]
    pub fn user_function(&mut self, name: String, f: Function, args: Vec<Vertex>) -> Vertex {
        let typ = f.ret_typ().clone();
        let function_id = self.functions.push(f);
        VertexInner {
            node: VertexNode::UserFunction(function_id, args),
            typ: Some(typ),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn add_user_function(&mut self, f: Function) -> UFunctionId {
        self.functions.push(f)
    }

    #[track_caller]
    pub fn user_function_by_id(
        &mut self,
        name: String,
        function_id: UFunctionId,
        args: Vec<Vertex>,
    ) -> Vertex {
        VertexInner {
            node: VertexNode::UserFunction(function_id, args),
            typ: None,
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn array(&self, name: String, typ: Typ, value: Vec<Vertex>) -> Vertex {
        let len = value.len();
        VertexInner {
            node: VertexNode::Array(value),
            typ: Some(Typ::Array(Box::new(typ), len)),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn scatter_array(&self, name: &str, n: usize, value: Vertex) -> Vec<Vertex> {
        (0..n)
            .map(|i| {
                VertexInner {
                    node: VertexNode::ArrayGet(value.clone(), i),
                    typ: None,
                    src: SourceInfo::new(*Location::caller(), format!("{}_{}", name, i)),
                }
                .into()
            })
            .collect()
    }

    #[track_caller]
    pub fn constant<T: 'static>(&mut self, name: String, value: T) -> Vertex {
        let typ = Typ::Any(any::TypeId::of::<T>(), std::mem::size_of::<T>());
        let constant_id = self.constants.push(Constant::new(name.clone(), value));
        VertexInner {
            node: VertexNode::Constant(constant_id),
            typ: Some(typ),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn blind(&self, name: String, value: Vertex, begin: usize) -> Vertex {
        VertexInner {
            node: VertexNode::Blind(value.clone(), begin),
            typ: None,
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn array_get(&self, name: String, array: Vertex, index: usize) -> Vertex {
        VertexInner {
            node: VertexNode::ArrayGet(array, index),
            typ: None,
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn msm(&self, name: String, scalars: Vertex, points: Vertex) -> Vertex {
        VertexInner {
            node: VertexNode::Msm { scalars, points },
            typ: None,
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn squeeze_challenge_scalar(&self, name: String, value: Vertex) -> (Vertex, Vertex) {
        let transcript_scalar = VertexInner {
            node: VertexNode::SqueezeScalar(value),
            typ: Some(Typ::Scalar),
            src: SourceInfo::new(*Location::caller(), format!("transcript_{}_pair", name)),
        }
        .into();
        self.unpack_pair(
            "transcript".to_string(),
            Typ::Transcript,
            name,
            Typ::Scalar,
            transcript_scalar,
        )
    }

    #[track_caller]
    pub fn lagrange_to_coef(&self, name: String, value: Vertex, n: u64) -> Vertex {
        VertexInner {
            node: VertexNode::Ntt {
                s: value,
                to: PolyRepr::Coef(n),
                from: PolyRepr::Lagrange(n),
            },
            typ: Some(Typ::coef(n)),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn coef_to_extended(&self, name: String, value: Vertex, n: u64, n_extended: u64) -> Vertex {
        VertexInner {
            node: VertexNode::Ntt {
                s: value,
                to: PolyRepr::ExtendedLagrange(n_extended),
                from: PolyRepr::Coef(n),
            },
            typ: Some(Typ::extended_lagrange(n_extended)),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn rotate_idx(&self, name: String, value: Vertex, shift: i32) -> Vertex {
        VertexInner {
            node: VertexNode::RotateIdx(value, shift),
            typ: None,
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
    }

    #[track_caller]
    pub fn lagrange_zeros(&self, name: String, n: u64) -> Vertex {
        VertexInner {
            node: VertexNode::NewPoly(n, PolyInit::Zeros),
            typ: Some(Typ::lagrange(n)),
            src: SourceInfo::new(*Location::caller(), name),
        }
        .into()
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
        }
        .into()
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
        }
        .into()
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
        }
        .into()
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
        }
        .into()
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
        }
        .into()
    }
}
