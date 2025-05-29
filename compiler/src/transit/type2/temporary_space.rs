use zkpoly_common::{arith::FusedType, load_dynamic::Libs, msm_config::MsmConfig};
use zkpoly_core::{
    fused_kernels::FusedOp,
    msm::MSM,
    poly::{KateDivision, PolyEval, PolyInvert, PolyPermute, PolyScan},
    poly_ptr::PolyPtr,
};
use zkpoly_runtime::args::RuntimeType;

use super::Arith;

// msm can work on several cards, so its return type is Vec<usize>
pub fn msm<Rt: RuntimeType>(msm_config: &MsmConfig, len: usize, libs: &mut Libs) -> Vec<u64> {
    let msm_impl = MSM::<Rt>::new(libs, msm_config.clone());
    assert_eq!(msm_config.cards.len(), 1);
    vec![msm_impl.get_buffer_size(len)[0] as u64] // we don't support multi-card yet
}

pub fn poly_permute<Rt: RuntimeType>(len: usize, libs: &mut Libs) -> Vec<u64> {
    let poly_permute_impl = PolyPermute::<Rt>::new(libs);
    vec![poly_permute_impl.get_buffer_size(len) as u64]
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

pub fn arith<Rt: RuntimeType>(arith: &Arith, chunking: Option<u64>) -> Vec<u64> {
    let local_size = FusedOp::new(
        arith.clone(),
        "temp".to_string(),
        size_of::<Rt::Field>() / size_of::<u32>(),
    )
    .get_temp_buffer_size()
        + 1; // add one to avoid zero size
    if chunking.is_none() {
        let (vars, mut_vars) = arith.gen_var_lists();
        vec![
            ((vars.len() + mut_vars.len()) * size_of::<PolyPtr>()) as u64,
            local_size as u64,
        ]
    } else {
        let mut temp_sizes = Vec::new();
        let poly_degree = arith.poly_degree.unwrap();
        let poly_chunk_size = poly_degree / chunking.unwrap() as usize * size_of::<Rt::Field>();
        let (vars, mut_vars) = arith.gen_var_lists();
        // place for arguments
        let arg_size = (2 * vars.len() + 3 * mut_vars.len()) * size_of::<PolyPtr>();
        temp_sizes.push(arg_size as u64);
        temp_sizes.push(local_size as u64); // place for local variables

        // place for chunking
        let (vars, mut_vars) = arith.gen_var_lists();

        for (typ, _) in vars.iter() {
            if typ == &FusedType::Scalar {
                temp_sizes.push(size_of::<Rt::Field>() as u64);
            } else {
                // double buffer
                temp_sizes.push(poly_chunk_size as u64);
                temp_sizes.push(poly_chunk_size as u64);
            }
        }

        for (typ, _) in mut_vars.iter() {
            if typ == &FusedType::Scalar {
                temp_sizes.push(size_of::<Rt::Field>() as u64);
            } else {
                // triple buffer
                temp_sizes.push(poly_chunk_size as u64);
                temp_sizes.push(poly_chunk_size as u64);
                temp_sizes.push(poly_chunk_size as u64);
            }
        }
        temp_sizes
    }
}
