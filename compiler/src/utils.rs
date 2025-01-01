
/// Computes smallest k such that
///   2^k >= [`x`]
fn min_power_of_2_above(n: usize) -> usize {
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
