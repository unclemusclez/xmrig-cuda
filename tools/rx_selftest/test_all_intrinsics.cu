#include <hip/hip_runtime.h>

// Test all possible rounding intrinsics
__global__ void test_kernel() {
    double a = 1.0, b = 2.0, c = 3.0;
    
    // FMA intrinsics
    #ifdef __fma_rd
    double r1 = __fma_rd(a, b, c);
    #endif
    #ifdef __fma_ru
    double r2 = __fma_ru(a, b, c);
    #endif
    #ifdef __fma_rz
    double r3 = __fma_rz(a, b, c);
    #endif
    
    // Division intrinsics
    #ifdef __ddiv_rd
    double r4 = __ddiv_rd(a, b);
    #endif
    #ifdef __ddiv_ru
    double r5 = __ddiv_ru(a, b);
    #endif
    #ifdef __ddiv_rz
    double r6 = __ddiv_rz(a, b);
    #endif
    
    // Sqrt intrinsics
    #ifdef __dsqrt_rd
    double r7 = __dsqrt_rd(a);
    #endif
    #ifdef __dsqrt_ru
    double r8 = __dsqrt_ru(a);
    #endif
    #ifdef __dsqrt_rz
    double r9 = __dsqrt_rz(a);
    #endif
    
    // Addition/multiplication intrinsics
    #ifdef __dadd_rd
    double r10 = __dadd_rd(a, b);
    #endif
    #ifdef __dmul_rd
    double r11 = __dmul_rd(a, b);
    #endif
}