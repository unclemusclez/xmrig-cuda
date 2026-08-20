#include <hip/hip_runtime.h>
__global__ void test_kernel() {
    double a = __fma_rd(1.0, 2.0, 3.0);
}