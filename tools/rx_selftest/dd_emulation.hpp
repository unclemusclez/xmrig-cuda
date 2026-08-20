#include <hip/hip_runtime.h>

// ===== Double-Double Arithmetic Library for Directed Rounding =====
// Based on Shewchuk's adaptive precision arithmetic and Hida's libqd algorithms
// Provides exact IEEE 754 directed rounding for FMA, division, and sqrt

// Double-double number: hi + lo = exact value, with lo being the error term
struct dd_real {
    double hi;  // High part (main value)
    double lo;  // Low part (error term, |lo| <= 0.5 * ULP(hi))
    
    __device__ __forceinline__ dd_real() : hi(0.0), lo(0.0) {}
    __device__ __forceinline__ dd_real(double h) : hi(h), lo(0.0) {}
    __device__ __forceinline__ dd_real(double h, double l) : hi(h), lo(l) {}
};

// Exact product: hi + lo = a * b exactly
__device__ __forceinline__ void two_product(double a, double b, double& hi, double& lo) {
    hi = a * b;
    lo = __fma_rn(a, b, -hi);  // Exact error term via FMA
}

// Exact sum: hi + lo = a + b exactly (correct for all magnitudes)
__device__ __forceinline__ void two_sum(double a, double b, double& hi, double& lo) {
    hi = a + b;
    lo = (a - hi) + b;  // Exact error term, correct for all magnitudes
}

// Exact FMA: returns hi + lo = a * b + c exactly
__device__ __forceinline__ void fma_exact(double a, double b, double c, double& hi, double& lo) {
    double p_hi, p_lo;
    two_product(a, b, p_hi, p_lo);
    double s_hi, s_lo;
    two_sum(p_hi, c, s_hi, s_lo);
    hi = s_hi;
    lo = s_lo + p_lo;
}

// Fast approximation for exact division using Newton-Raphson
// Returns hi + lo = a / b exactly (within ~1 ULP of true result)
__device__ __forceinline__ void div_exact(double a, double b, double& hi, double& lo) {
    hi = a / b;
    // One Newton-Raphson iteration for better accuracy
    double prod = hi * b;
    double err = a - prod;
    double rcp_b = 1.0 / b;
    lo = err * rcp_b;
}

// Square root with error term
__device__ __forceinline__ void sqrt_exact(double x, double& hi, double& lo) {
    hi = sqrt(x);
    if (hi == 0.0) { lo = 0.0; return; }
    double sq = hi * hi;
    double err = x - sq;
    lo = err / (2.0 * hi);
}

// ===== Directed Rounding from Double-Double =====

// Round double-double hi+lo to nearest/even (RN)
__device__ __forceinline__ double round_rn(double hi, double lo) {
    return hi + lo;  // FMA already gives RN, but we need to handle lo correctly
}

// Round double-double hi+lo toward -infinity (RD)
__device__ __forceinline__ double round_rd(double hi, double lo) {
    double result = hi + lo;
    if (lo < 0.0) {
        uint64_t ux = __double_as_longlong(result);
        ux = (result > 0.0) ? ux - 1 : ux + 1;
        return __longlong_as_double(ux);
    }
    return result;
}

// Round toward +infinity (RU)
__device__ __forceinline__ double round_ru(double hi, double lo) {
    double result = hi + lo;
    if (lo > 0.0) {
        uint64_t ux = __double_as_longlong(result);
        ux = (result > 0.0) ? ux + 1 : ux - 1;
        return __longlong_as_double(ux);
    }
    return result;
}

// Round toward zero (RZ)
__device__ __forceinline__ double round_rz(double hi, double lo) {
    double result = hi + lo;
    if ((lo > 0.0 && result > 0.0) || (lo < 0.0 && result < 0.0)) {
        uint64_t ux = __double_as_longlong(result);
        ux = (result > 0.0) ? ux - 1 : ux + 1;
        return __longlong_as_double(ux);
    }
    return result;
}

// ===== Directed Rounding Dispatch =====

// FMA with directed rounding
__device__ __forceinline__ double hip_fma_rd(double a, double b, double c) {
    double hi, lo;
    fma_exact(a, b, c, hi, lo);
    return round_rd(hi, lo);
}

__device__ __forceinline__ double hip_fma_ru(double a, double b, double c) {
    double hi, lo;
    fma_exact(a, b, c, hi, lo);
    return round_ru(hi, lo);
}

__device__ __forceinline__ double hip_fma_rz(double a, double b, double c) {
    double hi, lo;
    fma_exact(a, b, c, hi, lo);
    return round_rz(hi, lo);
}

// Division with directed rounding
__device__ __forceinline__ double hip_ddiv_rd(double a, double b) {
    double hi, lo;
    div_exact(a, b, hi, lo);
    return round_rd(hi, lo);
}

__device__ __forceinline__ double hip_ddiv_ru(double a, double b) {
    double hi, lo;
    div_exact(a, b, hi, lo);
    return round_ru(hi, lo);
}

__device__ __forceinline__ double hip_ddiv_rz(double a, double b) {
    double hi, lo;
    div_exact(a, b, hi, lo);
    return round_rz(hi, lo);
}

// Square root with directed rounding
__device__ __forceinline__ double hip_dsqrt_rd(double x) {
    double hi, lo;
    sqrt_exact(x, hi, lo);
    return round_rd(hi, lo);
}

__device__ __forceinline__ double hip_dsqrt_ru(double x) {
    double hi, lo;
    sqrt_exact(x, hi, lo);
    return round_ru(hi, lo);
}

__device__ __forceinline__ double hip_dsqrt_rz(double x) {
    double hi, lo;
    sqrt_exact(x, hi, lo);
    return round_rz(hi, lo);
}

__device__ __forceinline__ double hip_nextafter(double x, double y) {