use halo2curves::{bls12381, bn256};
use zkpoly_runtime::args::RuntimeType;
use zkpoly_runtime::transcript::{Blake2bWrite, Challenge255};

#[derive(Debug, Clone)]
pub struct MyRuntimeType;

impl RuntimeType for MyRuntimeType {
    type Field = bn256::Fr;
    type PointAffine = bn256::G1Affine;
    type Challenge = Challenge255<Self::PointAffine>;
    type Trans = Blake2bWrite<Vec<u8>, Self::PointAffine, Challenge255<Self::PointAffine>>;
}

pub type MyField = <MyRuntimeType as RuntimeType>::Field;

#[derive(Debug, Clone)]
pub struct BLS12381;

impl RuntimeType for BLS12381 {
    type Field = bls12381::Fr;
    type PointAffine = bls12381::G1Affine;
    type Challenge = Challenge255<Self::PointAffine>;
    type Trans = Blake2bWrite<Vec<u8>, Self::PointAffine, Challenge255<Self::PointAffine>>;
}
