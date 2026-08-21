// Test GPU __umul64hi / __mul64hi against exact __int128 ground truth.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>

__global__ void mulhi_kernel(const uint64_t* a, const uint64_t* b, uint64_t* out_umul, uint64_t* out_mul, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_umul[i] = __umul64hi(a[i], b[i]);
    out_mul[i] = (uint64_t)__mul64hi((int64_t)a[i], (int64_t)b[i]);
}

static uint64_t xs_state = 0x8877665544332211ULL;
static uint64_t xs() { uint64_t x = xs_state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; xs_state = x; return x; }

int main() {
    int n = 200000;
    uint64_t* ha = new uint64_t[n]; uint64_t* hb = new uint64_t[n];
    for (int i = 0; i < n; ++i) { ha[i] = xs(); hb[i] = xs(); }
    // also add edge cases
    ha[0]=0; hb[0]=0; ha[1]=UINT64_MAX; hb[1]=UINT64_MAX; ha[2]=UINT64_MAX; hb[2]=1;
    ha[3]=(uint64_t)INT64_MIN; hb[3]=(uint64_t)INT64_MIN; ha[4]=(uint64_t)-1; hb[4]=(uint64_t)INT64_MAX;

    uint64_t *d_a, *d_b, *d_um, *d_m;
    hipMalloc(&d_a, n*8); hipMalloc(&d_b, n*8); hipMalloc(&d_um, n*8); hipMalloc(&d_m, n*8);
    hipMemcpy(d_a, ha, n*8, hipMemcpyHostToDevice); hipMemcpy(d_b, hb, n*8, hipMemcpyHostToDevice);
    int blocks = (n + 255) / 256;
    hipLaunchKernelGGL(mulhi_kernel, dim3(blocks), dim3(256), 0, 0, d_a, d_b, d_um, d_m, n);
    hipError_t kerr = hipGetLastError();
    hipError_t serr = hipDeviceSynchronize();
    printf("launch err=%d sync err=%d\n", (int)kerr, (int)serr);
    uint64_t* gum = new uint64_t[n]; uint64_t* gm = new uint64_t[n];
    hipMemcpy(gum, d_um, n*8, hipMemcpyDeviceToHost); hipMemcpy(gm, d_m, n*8, hipMemcpyDeviceToHost);

    int um_mm = 0, m_mm = 0;
    for (int i = 0; i < n; ++i) {
        unsigned __int128 uex = (unsigned __int128)ha[i] * (unsigned __int128)hb[i];
        uint64_t uref = (uint64_t)(uex >> 64);
        __int128 sex = (__int128)(int64_t)ha[i] * (__int128)(int64_t)hb[i];
        uint64_t sref = (uint64_t)((uint64_t)(sex >> 64));
        if (gum[i] != uref) { if (um_mm < 5) printf("UMUL_MM i=%d a=%016llx b=%016llx gpu=%016llx ref=%016llx\n", i, ha[i], hb[i], gum[i], uref); um_mm++; }
        if (gm[i] != sref) { if (m_mm < 5) printf("MUL_MM i=%d a=%016llx b=%016llx gpu=%016llx ref=%016llx\n", i, ha[i], hb[i], gm[i], sref); m_mm++; }
    }
    printf("umul64hi mismatches=%d  mul64hi mismatches=%d of %d\n", um_mm, m_mm, n);
    return (um_mm || m_mm) ? 1 : 0;
}
