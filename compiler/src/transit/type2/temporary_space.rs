use zkpoly_common::{load_dynamic::Libs, msm_config::MsmConfig};
use zkpoly_core::{msm::MSM, ntt::SsipNtt, poly::{KateDivision, PolyEval, PolyInvert, PolyScan}};
use zkpoly_runtime::args::RuntimeType;

use super::NttAlgorithm;

// msm can work on several cares, so its return type is Vec<usize>
pub fn msm<Rt: RuntimeType>(msm_config: &MsmConfig, len: usize, libs: &mut Libs) -> usize {
    let msm_impl = MSM::<Rt>::new(libs, msm_config.clone());
    assert_eq!(msm_config.cards.len(), 1);
    msm_impl.get_buffer_size(len)[0] // we don't support multi-card yet
}

pub fn poly_eval<Rt: RuntimeType>(len: usize, libs: &mut Libs) -> usize {
    let poly_eval_impl = PolyEval::<Rt>::new(libs);
    poly_eval_impl.get_buffer_size(len)
}

pub fn kate_division<Rt: RuntimeType>(len: usize, libs: &mut Libs) -> usize {
    let kate_division_impl = KateDivision::<Rt>::new(libs);
    assert!(len.is_power_of_two(), "kate_division: len must be a power of 2");
    let log_len = len.trailing_zeros() as u32;
    kate_division_impl.get_buffer_size(log_len)
}

pub fn poly_scan<Rt: RuntimeType>(len: usize, libs: &mut Libs) -> usize {
    let poly_scan_impl = PolyScan::<Rt>::new(libs);
    poly_scan_impl.get_buffer_size(len)
}

pub fn poly_invert<Rt: RuntimeType>(len: usize, libs: &mut Libs) -> usize {
    let poly_invert_impl = PolyInvert::<Rt>::new(libs);
    poly_invert_impl.get_buffer_size(len)
}
