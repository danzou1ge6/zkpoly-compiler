use zkpoly_common::{load_dynamic::Libs, msm_config::MsmConfig};
use zkpoly_core::{
    msm::MSM,
    poly::{KateDivision, PolyEval, PolyInvert, PolyScan},
};
use zkpoly_runtime::args::RuntimeType;

// msm can work on several cards, so its return type is Vec<usize>
pub fn msm<Rt: RuntimeType>(msm_config: &MsmConfig, len: usize, libs: &mut Libs) -> Vec<u64> {
    let msm_impl = MSM::<Rt>::new(libs, msm_config.clone());
    assert_eq!(msm_config.cards.len(), 1);
    vec![msm_impl.get_buffer_size(len)[0] as u64] // we don't support multi-card yet
}

pub fn poly_eval<Rt: RuntimeType>(len: usize, libs: &mut Libs) -> Vec<u64> {
    let poly_eval_impl = PolyEval::<Rt>::new(libs);
    vec![poly_eval_impl.get_buffer_size(len) as u64]
}

pub fn kate_division<Rt: RuntimeType>(len: usize, libs: &mut Libs) -> Vec<u64> {
    let kate_division_impl = KateDivision::<Rt>::new(libs);
    assert!(
        len.is_power_of_two(),
        "kate_division: len must be a power of 2"
    );
    let log_len = len.trailing_zeros() as u32;
    vec![kate_division_impl.get_buffer_size(log_len) as u64]
}

pub fn poly_scan<Rt: RuntimeType>(len: usize, libs: &mut Libs) -> Vec<u64> {
    let poly_scan_impl = PolyScan::<Rt>::new(libs);
    vec![poly_scan_impl.get_buffer_size(len) as u64]
}

pub fn poly_invert<Rt: RuntimeType>(len: usize, libs: &mut Libs) -> Vec<u64> {
    let poly_invert_impl = PolyInvert::<Rt>::new(libs);
    vec![poly_invert_impl.get_buffer_size(len) as u64]
}
