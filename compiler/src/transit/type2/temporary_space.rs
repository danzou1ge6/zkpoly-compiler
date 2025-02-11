use zkpoly_common::msm_config::MsmConfig;
use zkpoly_runtime::args::RuntimeType;

use super::NttAlgorithm;

pub fn msm<Rt: RuntimeType>(msm_config: &MsmConfig) -> u64 {
    todo!("Returns the number of bytes used for MSM")
}

pub fn ntt<Rt: RuntimeType>(ntt_config: &NttAlgorithm) -> u64 {
    todo!("Returns the number of bytes used for NTT")
}
