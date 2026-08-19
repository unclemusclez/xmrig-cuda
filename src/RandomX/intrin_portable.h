/*
 * CUDA/HIP-specific intrinsics for RandomX GPU build.
 * This file overrides the portable fallback (intrin_portable.h) with
 * EFT-based correctly-rounded operations for AMD/HIP.
 * Placed in src/RandomX/ to be picked up before .reference/RandomX/src/intrin_portable.h.
 */
#pragma once

#ifndef FORCE_INLINE
#define FORCE_INLINE __forceinline
#endif

#include "intrin_cuda.h"