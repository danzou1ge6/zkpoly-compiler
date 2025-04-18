#include "kate_division.cuh"
#include "poly.h"

cudaError_t kate_division(void* temp_buf, unsigned long *temp_buf_size, unsigned int log_p, ConstPolyPtr p, const unsigned int *b, PolyPtr q, cudaStream_t stream) {
    return detail::kate_division<POLY_FIELD>(temp_buf, temp_buf_size, log_p, p, reinterpret_cast<const POLY_FIELD*>(b), q, stream);
}