#include "poly_eval.cuh"
#include "poly.h"

cudaError_t poly_eval(void* temp_buf, unsigned long *temp_buf_size, ConstPolyPtr poly,  unsigned int* res, const unsigned int*x, cudaStream_t stream) {
    return detail::poly_eval<POLY_FIELD>(temp_buf, temp_buf_size, poly, res, reinterpret_cast<const POLY_FIELD*>(x), stream);
}