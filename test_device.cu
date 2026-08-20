#include <hip/hip_runtime.h>
__device__ double test_kernel() {
    return __fma_rd(1.0, 2.0, 3.0);
}