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
    64 - x.leading_zeros()
}

pub fn log2(x: u64) -> Option<u32> {
    let r = log2_ceil(x);
    if 2u64.pow(r as u32) == x {
        Some(r)
    } else {
        None
    }
}
