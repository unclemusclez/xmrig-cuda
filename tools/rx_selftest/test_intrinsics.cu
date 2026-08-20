#include <hip/hip_runtime.h>
__global__ void test_kernel() {
    double a = __ddiv_rd(1.0, 2.0);
    double b = __dsqrt_rd(1.0);
}