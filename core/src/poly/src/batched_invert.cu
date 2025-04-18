#include "batched_invert.cuh"
#include "poly.h"

cudaError_t batched_invert(void *temp_buffer, unsigned long *buffer_size, PolyPtr poly, cudaStream_t stream) {
    return detail::batched_invert<POLY_FIELD>(temp_buffer, buffer_size, poly, stream);
}
