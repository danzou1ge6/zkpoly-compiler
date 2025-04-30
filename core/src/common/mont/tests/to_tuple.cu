#include "../src/field_impls.cuh"
#include <iostream>

using Number = bn254_fr::Number;
int main() {
    Number a = bn254_fr::Params::m();
    auto tuple = a.to_tuple();
    std::cout << "(" 
          << ::cuda::std::get<0>(tuple) << ", "
          << ::cuda::std::get<1>(tuple) << ", "
          // ...依此类推，直到最后一个元素
          << ::cuda::std::get<7>(tuple) 
          << ")" << std::endl;
    return 0;
}