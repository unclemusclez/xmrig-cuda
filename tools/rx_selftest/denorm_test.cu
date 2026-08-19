#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>

__global__ void denorm_test(uint64_t* out) {
    // smallest normal double and a subnormal-producing product
    double x = __longlong_as_double(0x0010000000000000ULL); // ~2.2e-308 (smallest normal)
    double y = x * x;            // ~5e-616 -> subnormal
    double z = fma(x, x, 0.0);   // same via fma
    double w = sqrt(x * x * x);  // sqrt of tiny -> subnormal
    out[0] = __double_as_longlong(y);
    out[1] = __double_as_longlong(z);
    out[2] = __double_as_longlong(w);
    // also: is y exactly 0.0 (flushed) or subnormal?
    out[3] = (y == 0.0) ? 1 : 0;
    out[4] = (z == 0.0) ? 1 : 0;
    out[5] = (w == 0.0) ? 1 : 0;
}

int main() {
    uint64_t* d; hipMalloc(&d, 8 * sizeof(uint64_t));
    uint64_t h[8] = {0};
    hipLaunchKernelGGL(denorm_test, 1, 1, 0, 0, d);
    hipDeviceSynchronize();
    hipMemcpy(h, d, 8 * sizeof(uint64_t), hipMemcpyDeviceToHost);
    printf("y  (x*x)        = %016llx  flushed=%llu\n", (unsigned long long)h[0], (unsigned long long)h[3]);
    printf("z  (fma x*x 0)  = %016llx  flushed=%llu\n", (unsigned long long)h[1], (unsigned long long)h[4]);
    printf("w  (sqrt x^3)   = %016llx  flushed=%llu\n", (unsigned long long)h[2], (unsigned long long)h[5]);
    hipFree(d);
    return 0;
}
