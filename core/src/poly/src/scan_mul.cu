#include "scan_mul.cuh"

cudaError_t scan_mul(void * temp_buffer, unsigned long *buffer_size, PolyPtr target, cudaStream_t stream) {
    return detail::scan_mul<POLY_FIELD>(temp_buffer, buffer_size, target, stream);
}