/*
 * CUDA/HIP-specific vector intrinsics for RandomX GPU build.
 * Overrides the portable fallback with EFT-based correctly-rounded operations.
 */
#pragma once

#ifndef FORCE_INLINE
#define FORCE_INLINE __forceinline
#endif

#if defined(__HIP_DEVICE_COMPILE__)
#define HD_INLINE __host__ __device__ __forceinline__
#else
#define HD_INLINE FORCE_INLINE
#endif

// Type definitions (normally from intrin_portable.h)
typedef union {
    uint64_t u64[2];
    uint32_t u32[4];
    uint16_t u16[8];
    uint8_t u8[16];
} rx_vec_i128;

typedef union {
    struct {
        double lo;
        double hi;
    };
    rx_vec_i128 i;
} rx_vec_f128;

#include "common.hpp"
#include "randomx_cuda.hpp"

// Host/device helper functions (from intrin_portable.h)
FORCE_INLINE constexpr int32_t unsigned32ToSigned2sCompl(uint32_t x) {
    return (x > INT32_MAX) ? (-(int32_t)(UINT32_MAX - x) - 1) : (int32_t)x;
}

FORCE_INLINE uint32_t load32(const uint8_t* addr) {
    return *reinterpret_cast<const uint32_t*>(addr);
}

FORCE_INLINE uint64_t load64(const uint8_t* addr) {
    return *reinterpret_cast<const uint64_t*>(addr);
}

FORCE_INLINE void store32(uint8_t* addr, uint32_t val) {
    *reinterpret_cast<uint32_t*>(addr) = val;
}

FORCE_INLINE void store64(uint8_t* addr, uint64_t val) {
    *reinterpret_cast<uint64_t*>(addr) = val;
}

namespace randomx {

// rx_vec_f128 is defined in intrin_portable.h as:
//   typedef union { struct { double lo; double hi; }; rx_vec_i128 i; } rx_vec_f128;

// Basic operations that don't need rounding (bitwise/transformation)
FORCE_INLINE rx_vec_f128 rx_swap_vec_f128(rx_vec_f128 a) {
    double temp = a.lo;
    a.lo = a.hi;
    a.hi = temp;
    return a;
}

FORCE_INLINE rx_vec_f128 rx_set_vec_f128(uint64_t x1, uint64_t x0) {
    rx_vec_f128 v;
    v.i.u64[0] = x0;
    v.i.u64[1] = x1;
    return v;
}

FORCE_INLINE rx_vec_f128 rx_set1_vec_f128(uint64_t x) {
    rx_vec_f128 v;
    v.i.u64[0] = x;
    v.i.u64[1] = x;
    return v;
}

FORCE_INLINE rx_vec_f128 rx_xor_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
    rx_vec_f128 x;
    x.i.u64[0] = a.i.u64[0] ^ b.i.u64[0];
    x.i.u64[1] = a.i.u64[1] ^ b.i.u64[1];
    return x;
}

FORCE_INLINE rx_vec_f128 rx_and_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
    rx_vec_f128 x;
    x.i.u64[0] = a.i.u64[0] & b.i.u64[0];
    x.i.u64[1] = a.i.u64[1] & b.i.u64[1];
    return x;
}

FORCE_INLINE rx_vec_f128 rx_or_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
    rx_vec_f128 x;
    x.i.u64[0] = a.i.u64[0] | b.i.u64[0];
    x.i.u64[1] = a.i.u64[1] | b.i.u64[1];
    return x;
}

FORCE_INLINE rx_vec_f128 rx_cast_vec_i2f(rx_vec_i128 a) {
    rx_vec_f128 x;
    x.i = a;
    return x;
}

FORCE_INLINE rx_vec_i128 rx_cast_vec_f2i(rx_vec_f128 a) {
    return a.i;
}

FORCE_INLINE rx_vec_f128 rx_cvt_packed_int_vec_f128(const void* addr) {
    rx_vec_f128 x;
    x.lo = (double)unsigned32ToSigned2sCompl(load32((const uint8_t*)addr + 0));
    x.hi = (double)unsigned32ToSigned2sCompl(load32((const uint8_t*)addr + 4));
    return x;
}

// EFT-based correctly-rounded FPU operations (respect rounding mode via global MXCSR state)
// These replace the portable fallback's plain +, *, /, sqrt which ignore rounding mode.

// The VM maintains the current rounding mode in a global variable. We use thread-local
// storage to pass the mode to the EFT functions. The VM calls rx_set_rounding_mode
// before each FPU operation.

// Thread-local rounding mode (0=RN, 1=RD, 2=RU, 3=RZ)
#if __HIP_DEVICE_COMPILE__
__device__ __forceinline__ int& rx_gpu_rounding_mode() {
    thread_local int mode = 0;
    return mode;
}
#else
// Host version: use a static thread-local
inline int& rx_gpu_rounding_mode() {
    thread_local int mode = 0;
    return mode;
}
#endif

#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__
#endif
FORCE_INLINE void rx_set_rounding_mode(int mode) {
    rx_gpu_rounding_mode() = mode;
}

#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__
#endif
FORCE_INLINE int rx_get_rounding_mode() {
    return rx_gpu_rounding_mode();
}

// Host implementations of EFT functions (for round-to-nearest only)
// Device versions are in randomx_cuda.hpp
#if !__HIP_DEVICE_COMPILE__
inline double rx_fma_eft(double a, double b, double c, int mode) {
    (void)mode; // Host only uses round-to-nearest
    return std::fma(a, b, c);
}

inline double rx_ddiv(double a, double b, int mode) {
    (void)mode; // Host only uses round-to-nearest
    return a / b;
}

inline double rx_dsqrt(double a, int mode) {
    (void)mode; // Host only uses round-to-nearest
    return std::sqrt(a);
}
#else
__device__ __forceinline__ double rx_fma_eft(double a, double b, double c, int mode);
__device__ double rx_ddiv(double a, double b, int mode);
__device__ double rx_dsqrt(double a, int mode);
#endif

HD_INLINE rx_vec_f128 rx_add_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
    int mode = rx_get_rounding_mode();
    rx_vec_f128 x;
    x.lo = rx_fma_eft(a.lo, 1.0, b.lo, mode);
    x.hi = rx_fma_eft(a.hi, 1.0, b.hi, mode);
    return x;
}

HD_INLINE rx_vec_f128 rx_sub_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
    int mode = rx_get_rounding_mode();
    rx_vec_f128 x;
    x.lo = rx_fma_eft(a.lo, 1.0, -b.lo, mode);
    x.hi = rx_fma_eft(a.hi, 1.0, -b.hi, mode);
    return x;
}

HD_INLINE rx_vec_f128 rx_mul_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
    int mode = rx_get_rounding_mode();
    rx_vec_f128 x;
    x.lo = rx_fma_eft(a.lo, b.lo, 0.0, mode);
    x.hi = rx_fma_eft(a.hi, b.hi, 0.0, mode);
    return x;
}

HD_INLINE rx_vec_f128 rx_div_vec_f128(rx_vec_f128 a, rx_vec_f128 b) {
    int mode = rx_get_rounding_mode();
    rx_vec_f128 x;
    x.lo = rx_ddiv(a.lo, b.lo, mode);
    x.hi = rx_ddiv(a.hi, b.hi, mode);
    return x;
}

HD_INLINE rx_vec_f128 rx_sqrt_vec_f128(rx_vec_f128 a) {
    int mode = rx_get_rounding_mode();
    rx_vec_f128 x;
    x.lo = rx_dsqrt(a.lo, mode);
    x.hi = rx_dsqrt(a.hi, mode);
    return x;
}

} // namespace randomx