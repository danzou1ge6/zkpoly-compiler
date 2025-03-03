#include <random>
#include <ctime>
#include "../src/recompute_ntt.cuh"
#include "small_field.cuh"
#include "../src/precompute.cuh"

inline unsigned long long qpow(unsigned long long x, unsigned long long y) {
    unsigned long long base = 1ll;
    while(y) {
        if (y & 1ll) base = (base * x) % P;
        x = (x * x) % P;
        y >>= 1ll;
    }
    return base;
}

inline unsigned long long inv(unsigned long long x) {
    return qpow(x, P - 2);
}

void swap(unsigned long long &a, unsigned long long &b) {
    long long tmp = a;
    a = b;
    b = tmp;
}

void ntt_cpu(unsigned long long data[], unsigned long long reverse[], long long len, unsigned long long omega) {
    // rearrange the coefficients
    for (unsigned long long i = 0; i < len; i++) {
        if (i < reverse[i]) swap(data[i], data[reverse[i]]);
    }

    for (unsigned long long stride = 1ll; stride < len; stride <<= 1ll) {
        unsigned long long gap = qpow(omega, (P - 1ll) / (stride << 1ll));
        for (unsigned long long start = 0; start < len; start += (stride << 1ll)) {
            for (unsigned long long offset = 0, w = 1ll; offset < stride; offset++, w = (gap * w) % P) {
                unsigned long long a = data[start + offset], b = w * data[start + offset + stride] % P;
                data[start + offset] = (a + b) % P;
                data[start + offset + stride] = (a - b + P) % P;
            }
        }
    }
}

const unsigned long long WORDS = small_field::Params::LIMBS;
typedef small_field::Element Field;
typedef small_field::Number Number;

int main() {
    unsigned long long *data, *reverse, *data_copy;
    unsigned long long l,length = 1ll;
    int bits = 0;

    l = 1 << 10;

    while (length < l) {
        length <<= 1ll;
        bits ++;
    }

    data = new unsigned long long[length];
    data_copy = new unsigned long long[length];
    reverse = new unsigned long long [length];

    reverse[0] = 0;
    for (long long i = 0; i < length; i++) {
        reverse[i] = (reverse[i >> 1ll] >> 1ll) | ((i & 1ll) << (bits - 1ll) ); //reverse the bits
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    for (long long i = 0; i < length; i++) {
        data[i] = i % P;
        data_copy[i] = data[i];
    }

    // cpu implementation
    {
        clock_t start = clock();

        ntt_cpu(data, reverse, length, root);

        clock_t end = clock();
        printf("cpu: %lfms\n",(double)(end - start) / CLOCKS_PER_SEC * 1000);
    }

    uint *data_gpu;
    data_gpu = new uint [length * WORDS];

    uint omega[WORDS];
    memset(omega, 0, sizeof(uint) * WORDS);
    omega[0] = root;
    
    auto unit = Field::from_number(Number::load(omega));
    auto one = Number::zero();
    one.limbs[0] = 1;
    Number exponent = (Field::ParamsType::m() - one).slr(bits);
    unit = unit.pow(exponent);

    // starting test ssip_ntt

    // precompute
    unsigned int max_threads_stage1_log = 8;
    unsigned int max_threads_stage2_log = 8;

    unsigned int total_deg_stage1 = (bits + 1) / 2;
    unsigned int total_deg_stage2 = bits / 2;

    unsigned int max_deg_stage1 = max_threads_stage1_log + 1;
    unsigned int max_deg_stage2 = (max_threads_stage2_log + 2) / 2; // 4 elements per thread

    unsigned int deg_stage1 = detail::get_deg(total_deg_stage1, max_deg_stage1);
    unsigned int deg_stage2 = detail::get_deg(total_deg_stage2, max_deg_stage2);

    unsigned int max_deg = std::max(deg_stage1, deg_stage2);
    
    uint *pq, *omegas;

    pq = new uint[(1 << max_deg >> 1) * WORDS];
    omegas = new uint[32 * WORDS];

    detail::gen_pq_omegas<Field>(pq, omegas, max_deg, length, unit);

    uint *d_pq, *d_omegas;
    CUDA_CHECK(cudaMalloc(&d_pq, (1 << max_deg >> 1) * WORDS * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_omegas, 32 * WORDS * sizeof(uint)));
    CUDA_CHECK(cudaMemcpy(d_pq, pq, (1 << max_deg >> 1) * WORDS * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_omegas, omegas, 32 * WORDS * sizeof(uint), cudaMemcpyHostToDevice));

    // ssip_ntt
    memset(data_gpu, 0, sizeof(uint) * length * WORDS);
    for (int i = 0; i < length; i++) {
        data_gpu[i * WORDS] = data_copy[i];
    }

    uint *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, length * WORDS * sizeof(uint)));
    CUDA_CHECK(cudaMemcpy(d_data, data_gpu, length * WORDS * sizeof(uint), cudaMemcpyHostToDevice));

    auto d_ptr = PolyPtr{d_data, length, 0, 0, length};
    CUDA_CHECK(detail::recompute_ntt<Field>(d_ptr, d_pq, max_deg, d_omegas, bits, 0, max_threads_stage1_log, max_threads_stage2_log));

    CUDA_CHECK(cudaMemcpy(data_gpu, d_data, length * WORDS * sizeof(uint), cudaMemcpyDeviceToHost));
    
    for (long long i = 0; i < length; i++) {
        if (data[i] != data_gpu[i * WORDS]) {
            printf("%lld %u %lld\n", data[i], data_gpu[i * WORDS], i);
            return 1;
        }
    }

    printf("recompute_ntt passed\n");

    CUDA_CHECK(cudaFree(d_pq));
    CUDA_CHECK(cudaFree(d_omegas));
    CUDA_CHECK(cudaFree(d_data));

    delete [] data;
    delete [] data_copy;
    delete [] reverse;
    delete [] data_gpu;
    delete [] pq;
    delete [] omegas;
    return 0;
}