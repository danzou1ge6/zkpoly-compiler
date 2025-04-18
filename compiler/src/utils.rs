use std::collections::BTreeMap;

use halo2curves::ff::PrimeField;

/// Computes smallest k such that
///   2^k >= [`x`]
pub fn min_power_of_2_above(n: usize) -> usize {
    if n == 0 {
        return 0;
    }

    let mut k = 0;
    let mut x = n;

    while x != 0 {
        k += 1;
        x >>= 1;
    }

    k
}

pub fn log2_ceil(x: u64) -> u32 {
    if x == 0 {
        panic!("log2(0) is undefined");
    }
    64 - (x - 1).leading_zeros()
}

pub fn log2(x: u64) -> Option<u32> {
    let r = log2_ceil(x);
    if 2u64.pow(r as u32) == x {
        Some(r)
    } else {
        None
    }
}

pub struct GenOmega<F> {
    omegas: BTreeMap<(u32, bool), F>,
}

impl<F: PrimeField> GenOmega<F> {
    pub fn new() -> Self {
        Self {
            omegas: BTreeMap::new(),
        }
    }

    // adopted from halo2
    pub fn get_omega(&mut self, k: u32, inv: bool) -> F {
        assert!(k <= F::S);

        let omega = self.omegas.get(&(k, inv));
        if omega.is_some() {
            return omega.unwrap().clone();
        }

        let mut omega = F::ROOT_OF_UNITY;

        // Get omega, the 2^{k}'th root of unity
        // The loop computes omega = omega^{2 ^ (S - k)}
        // Notice that omega ^ {2 ^ k} = omega ^ {2^S} = 1.
        for _ in k..F::S {
            omega = omega.square();
        }

        self.omegas.insert((k, false), omega);

        if inv {
            let inv_omega = omega.invert().unwrap();
            self.omegas.insert((k, true), inv_omega);
            inv_omega
        } else {
            omega
        }
    }
}
