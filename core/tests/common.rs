use halo2curves::bn256;
use zkpoly_runtime::args::RuntimeType;
use zkpoly_runtime::transcript::{Blake2bWrite, Challenge255};

#[derive(Debug, Clone)]
pub struct MyRuntimeType;

impl RuntimeType for MyRuntimeType {
    type Field = bn256::Fr;
    type PointAffine = bn256::G1Affine;
    type Challenge = Challenge255<bn256::G1Affine>;
    type Trans = Blake2bWrite<Vec<u8>, bn256::G1Affine, Challenge255<bn256::G1Affine>>;
}

pub type MyField = <MyRuntimeType as RuntimeType>::Field;
