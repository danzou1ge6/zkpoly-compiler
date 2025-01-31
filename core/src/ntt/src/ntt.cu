#include "ntt.h"
#include "ssip_ntt.cuh"
#include "precompute.cuh"
#include "recompute_ntt.cuh"

cudaError_t ssip_ntt(unsigned int *x, const unsigned int *twiddle, unsigned int log_len, cudaStream_t stream, const unsigned int max_threads_stage1_log, const unsigned int max_threads_stage2_log) {
    return detail::ssip_ntt<NTT_FIELD>(x, twiddle, log_len, stream, max_threads_stage1_log, max_threads_stage2_log);
}

cudaError_t ssip_precompute(unsigned int *twiddle, unsigned int len, const unsigned int *unit) {
    detail::gen_roots_cub<NTT_FIELD> gen_roots;
    return gen_roots(twiddle, len, NTT_FIELD::load(unit));
}

cudaError_t recompute_ntt(unsigned int *x, const unsigned int *pq_d, unsigned int pq_deg, const unsigned int *omegas_d, unsigned int log_len, cudaStream_t stream, const unsigned int max_threads_stage1_log, const unsigned int max_threads_stage2_log) {
    return detail::recompute_ntt<NTT_FIELD>(x, pq_d, pq_deg, omegas_d, log_len, stream, max_threads_stage1_log, max_threads_stage2_log);
}

void gen_pq_omegas(unsigned int *pq, unsigned int *omegas, unsigned int pq_deg, unsigned int len, unsigned int *unit) {
    detail::gen_pq_omegas<NTT_FIELD>(pq, omegas, pq_deg, len, NTT_FIELD::load(unit));
}