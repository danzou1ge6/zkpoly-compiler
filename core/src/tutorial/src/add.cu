#include "add.h"
#include "../../common/mont/src/field_impls.cuh"
template <typename Field>
void add_impl(const unsigned int *a, const unsigned int *b, unsigned int *c) {
    Field fa = Field::load(a);
    Field fb = Field::load(b);
    Field fc = fa + fb;
    fc.store(c);
}

void simple_add(const unsigned int *a, const unsigned int *b, unsigned int *c) {
    add_impl<FIELD>(a, b, c);
}