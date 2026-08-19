// Focused test of the GPU directed-rounding primitives (rx_fma_rnd / rx_ddiv / rx_dsqrt)
// vs a CPU ground truth that uses fesetround + hardware FMA/div/sqrt -- the exact
// mechanism RandomX's x86 JIT depends on. Any bit-exact mismatch proves the GPU
// emulation is wrong.
//
// Usage: roundtest.exe <num_cases>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfenv>
#include <vector>
#include <hip/hip_runtime.h>
#include "cuda_device.hpp"
#include "cryptonight.h"
#include "randomx.h"

#include "randomx.cu"

namespace RandomX_Monero {
    #include "RandomX/monero/configuration.h"
    #define fillAes4Rx4 fillAes4Rx4_v104
    #include "RandomX/common.hpp"
}

static uint64_t xs_state = 0x12345678ULL;
static inline uint64_t xs() {
    uint64_t x = xs_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    xs_state = x;
    return x;
}
static inline uint64_t h_d2u(double x) { uint64_t u; memcpy(&u, &x, sizeof(u)); return u; }
static inline double   h_u2d(uint64_t u) { double x; memcpy(&x, &u, sizeof(x)); return x; }

static inline double rdouble() {
    // uniform in roughly [-1e300, 1e300], occasionally subnormal/inf.
    // NOTE: allow NEGATIVE inputs too -- RandomX generates signed floats
    // (entropy sign, NEGATIVE_SRC, FSUB), so the earlier positive-only generator
    // could not catch a sign-handling bug in the emulation.
    uint64_t bits = xs();
    if ((xs() & 63) == 0) {                  // occasionally force tiny/large
        int e = (int)(xs() & 2047);
        bits = (bits & 0x800FFFFFFFFFFFFFULL) | ((uint64_t)e << 52);
    }
    return h_u2d(bits);
}

__global__ void round_kernel(const double* a, const double* b, const double* c,
                              const int* mode, double* out_fma, double* out_div,
                              double* out_sqrt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_fma[i]  = RandomX_Monero::rx_fma_rnd(a[i], b[i], c[i], mode[i]);
    out_div[i]  = RandomX_Monero::rx_ddiv(a[i], b[i], mode[i]);
    out_sqrt[i] = RandomX_Monero::rx_dsqrt(a[i] >= 0.0 ? a[i] : 0.0, mode[i]);
}

int main(int argc, char** argv) {
    int n = (argc >= 2) ? (int)strtol(argv[1], nullptr, 10) : 200000;
    if (n <= 0) n = 200000;

    HIP_CHECK(0, hipSetDevice(0));

    std::vector<double> ha(n), hb(n), hc(n);
    std::vector<int>    hm(n);
    std::vector<double> ca(n), cb(n), cc(n);
    std::vector<int>    cm(n);

    for (int i = 0; i < n; ++i) {
        double a = rdouble(), b = rdouble(), c = rdouble();
        while (b == 0.0) b = rdouble();
        int m = (int)(xs() & 3);
        ha[i] = a; hb[i] = b; hc[i] = c; hm[i] = m;
        ca[i] = a; cb[i] = b; cc[i] = c; cm[i] = m;
    }

    double *d_a, *d_b, *d_c, *d_of, *d_od, *d_os; int* d_m;
    HIP_CHECK(0, hipMalloc(&d_a, n*sizeof(double)));
    HIP_CHECK(0, hipMalloc(&d_b, n*sizeof(double)));
    HIP_CHECK(0, hipMalloc(&d_c, n*sizeof(double)));
    HIP_CHECK(0, hipMalloc(&d_m, n*sizeof(int)));
    HIP_CHECK(0, hipMalloc(&d_of, n*sizeof(double)));
    HIP_CHECK(0, hipMalloc(&d_od, n*sizeof(double)));
    HIP_CHECK(0, hipMalloc(&d_os, n*sizeof(double)));
    HIP_CHECK(0, hipMemcpy(d_a, ha.data(), n*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(0, hipMemcpy(d_b, hb.data(), n*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(0, hipMemcpy(d_c, hc.data(), n*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(0, hipMemcpy(d_m, hm.data(), n*sizeof(int),   hipMemcpyHostToDevice));

    int threads = 256, blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(round_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_a, d_b, d_c, d_m, d_of, d_od, d_os, n);
    HIP_CHECK_KERNEL(0, round_kernel, blocks, threads);
    HIP_CHECK(0, hipDeviceSynchronize());

    std::vector<double> gof(n), god(n), gos(n);
    HIP_CHECK(0, hipMemcpy(gof.data(), d_of, n*sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(0, hipMemcpy(god.data(), d_od, n*sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(0, hipMemcpy(gos.data(), d_os, n*sizeof(double), hipMemcpyDeviceToHost));

    const int rm[4] = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
    int mismatch_fma = 0, mismatch_div = 0, mismatch_sqrt = 0;
    int nf_fma = 0, nf_div = 0, nf_sqrt = 0;
    auto expo = [](uint64_t u){ return (int)((u >> 52) & 0x7FF); };
    auto is_normal = [&](uint64_t u){ int e = expo(u); return e >= 1 && e <= 2046; };
    for (int i = 0; i < n; ++i) {
        int m = cm[i];
        fesetround(rm[m]);
        double rf = fma(ca[i], cb[i], cc[i]);
        double rd = ca[i] / cb[i];
        double rs = sqrt(ca[i] >= 0.0 ? ca[i] : 0.0);
        if (        h_d2u(gof[i]) != h_d2u(rf)) {
            if (mismatch_fma++ < 5)
                fprintf(stderr, "FMA mismatch i=%d m=%d a=%016llx b=%016llx c=%016llx gpu=%016llx cpu=%016llx\n",
                        i, m, h_d2u(ca[i]), h_d2u(cb[i]), h_d2u(cc[i]),
                        h_d2u(gof[i]), h_d2u(rf));
            if (is_normal(ca[i]) && is_normal(cb[i]) && is_normal(cc[i]) && is_normal(h_d2u(rf))) nf_fma++;
        }
        if (h_d2u(god[i]) != h_d2u(rd)) {
            if (mismatch_div++ < 5)
                fprintf(stderr, "DIV mismatch i=%d m=%d a=%016llx b=%016llx gpu=%016llx cpu=%016llx\n",
                        i, m, h_d2u(ca[i]), h_d2u(cb[i]),
                        h_d2u(god[i]), h_d2u(rd));
            if (is_normal(ca[i]) && is_normal(cb[i]) && is_normal(h_d2u(rd))) nf_div++;
        }
        if (h_d2u(gos[i]) != h_d2u(rs)) {
            if (mismatch_sqrt++ < 5)
                fprintf(stderr, "SQRT mismatch i=%d m=%d a=%016llx gpu=%016llx cpu=%016llx\n",
                        i, m, h_d2u(ca[i]),
                        h_d2u(gos[i]), h_d2u(rs));
            if (is_normal(ca[i]) && is_normal(h_d2u(rs))) nf_sqrt++;
        }
    }

    fprintf(stderr, "done: fma_mismatch=%d div_mismatch=%d sqrt_mismatch=%d (of %d)\n",
            mismatch_fma, mismatch_div, mismatch_sqrt, n);
    fprintf(stderr, "NORMAL-input/result mismatches: fma=%d div=%d sqrt=%d\n", nf_fma, nf_div, nf_sqrt);
    printf("RESULT fma=%d div=%d sqrt=%d\n", mismatch_fma, mismatch_div, mismatch_sqrt);
    return (mismatch_fma || mismatch_div || mismatch_sqrt) ? 1 : 0;
}
