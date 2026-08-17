#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <iostream>
#include <string>


#define HIP_THROW(error) throw std::runtime_error(std::string("<") + __FUNCTION__ + ">:" + std::to_string(__LINE__) + " \"" + (error) + "\"")


/** execute and check a HIP api command
 *
 * @param id gpu id (thread id)
 * @param ... HIP api command
 */
#define HIP_CHECK(id, ...) {                                                                             \
    hipError_t error = __VA_ARGS__;                                                                      \
    if (error != hipSuccess){                                                                            \
        HIP_THROW(hipGetErrorString(error));                                                            \
    }                                                                                                     \
}                                                                                                         \
( (void) 0 )

/** execute and check a HIP kernel
 *
 * @param id gpu id (thread id)
 * @param ... HIP kernel call
 */
#define HIP_CHECK_KERNEL(id, ...)      \
    __VA_ARGS__;                        \
    HIP_CHECK(id, hipGetLastError())

#define CUDA_CHECK_KERNEL HIP_CHECK_KERNEL
#define CUDA_CHECK HIP_CHECK
#define CU_CHECK HIP_DRV_CHECK

#if defined(XMRIG_ALGO_KAWPOW) || defined(XMRIG_ALGO_CN_R)
#define HIP_DRV_CHECK(id, ...) {                                                                             \
    hipError_t result = __VA_ARGS__;                                                                      \
    if(result != hipSuccess){                                                                         \
        HIP_THROW(hipGetErrorString(result));                                                            \
    }                                                                                                   \
}                                                                                                       \
( (void) 0 )
#endif