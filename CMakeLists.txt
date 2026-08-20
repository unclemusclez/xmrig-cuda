cmake_minimum_required(VERSION 3.16)
project(xmrig-rocm LANGUAGES C CXX HIP)
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocrand REQUIRED)


option(WITH_HIP_DRIVER_API "Enable HIP Driver API and HIPRTC, required for cn/r and kawpow algorithms" ON)

# Algorithm selection
option(WITH_CN_R       "Enable CryptoNight-R algorithm" ON)
option(WITH_CN_LITE    "Enable CryptoNight-Lite algorithms family" ON)
option(WITH_CN_HEAVY   "Enable CryptoNight-Heavy algorithms family" ON)
option(WITH_CN_PICO    "Enable CryptoNight-Pico algorithm" ON)
option(WITH_CN_FEMTO   "Enable CryptoNight-UPX2 algorithm" ON)
option(WITH_ARGON2     "Enable Argon2 algorithms family" OFF) #unsupported

option(WITH_RANDOMX  "Enable RandomX algorithms family" ON)
option(WITH_KAWPOW   "Enable KawPow algorithms family" ON)


include_directories(src)
add_compile_definitions(CUB_IGNORE_DEPRECATED_CPP_DIALECT)


include(cmake/cpu.cmake)
include(cmake/os.cmake)
include(cmake/flags.cmake)


set(SOURCES
    src/crypto/cn/c_blake256.c
    src/crypto/common/Algorithm.cpp
    src/crypto/common/Algorithm.h
    src/version.h
    src/xmrig-cuda.cpp
    src/xmrig-cuda.h
    src/cuda_core.cu
    src/cuda_extra.cu
    )


if (WITH_HIP_DRIVER_API AND WITH_CN_R)
    list(APPEND SOURCES src/CudaCryptonightR_gen.cpp)
endif()

if (WITH_RANDOMX)
    list(APPEND SOURCES src/RandomX/randomx.cu)
    list(APPEND SOURCES src/RandomX/arqma/randomx_arqma.cu)
    list(APPEND SOURCES src/RandomX/monero/randomx_monero.cu)
    list(APPEND SOURCES src/RandomX/wownero/randomx_wownero.cu)
    list(APPEND SOURCES src/RandomX/graft/randomx_graft.cu)
    list(APPEND SOURCES src/RandomX/yada/randomx_yada.cu)
endif()

if (WITH_KAWPOW)
    list(APPEND SOURCES src/KawPow/raven/KawPow.cu)
    list(APPEND SOURCES src/KawPow/raven/CudaKawPow_gen.cpp)
endif()

set_source_files_properties(
    src/cuda_core.cu
    src/cuda_extra.cu
    src/CryptonightR.cu
    src/RandomX/randomx.cu
    src/RandomX/arqma/randomx_arqma.cu
    src/RandomX/monero/randomx_monero.cu
    src/RandomX/wownero/randomx_wownero.cu
    src/RandomX/graft/randomx_graft.cu
    src/RandomX/yada/randomx_yada.cu
    src/KawPow/raven/KawPow.cu
    PROPERTIES LANGUAGE HIP
)

if (XMRIG_OS_WIN)
    if (NOT DEFINED ENV{WindowsSdkDir})
        set(WindowsSdkDir "C:/Program Files (x86)/Windows Kits/10" CACHE PATH "Fallback Windows SDK root")
    endif()
    if (NOT DEFINED ENV{WindowsSDKVersion})
        set(WindowsSDKVersion "10.0.26100.0" CACHE STRING "Fallback Windows SDK version")
    endif()
    set(RC_COMPILER "C:/Program Files (x86)/Windows Kits/10/bin/${WindowsSDKVersion}/x64/rc.exe")
    set(RC_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/app.res)
    add_custom_command(
        OUTPUT ${RC_OUTPUT}
        COMMAND ${RC_COMPILER}
            /I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            /I "${WindowsSdkDir}/Include/${WindowsSDKVersion}/um"
            /I "${WindowsSdkDir}/Include/${WindowsSDKVersion}/shared"
            /fo ${RC_OUTPUT}
            ${CMAKE_CURRENT_SOURCE_DIR}/res/app.rc
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/res/app.rc
        COMMENT "Compiling Windows resource file"
        VERBATIM
    )
    list(APPEND SOURCES ${RC_OUTPUT})
endif()

if (XMRIG_OS_APPLE)
    cmake_policy(SET CMP0042 NEW)
endif()

add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} hip::host hiprtc::hiprtc roc::rocblas rocrand)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if (WIN32)
    # RDNA3 (gfx1100) supports 32-lane wavefronts. The RandomX kernels assume
    # 32-lane warps (worker coordination, __launch_bounds__(32,16)), so we compile
    # in wave32 to make that assumption valid. Without it the 32-thread blocks run
    # as half of a 64-lane wavefront, which breaks worker coordination and cripples
    # occupancy (final occupancy 2), causing multi-second-per-batch stalls.
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE --offload-arch=gfx1100 -mno-wavefrontsize64 -O3)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --rocm-path=C:/ROCm72")
endif()

if (WITH_CN_LITE)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE XMRIG_ALGO_CN_LITE)
endif()

if (WITH_CN_HEAVY)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE XMRIG_ALGO_CN_HEAVY)
endif()

if (WITH_CN_PICO)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE XMRIG_ALGO_CN_PICO)
endif()

if (WITH_CN_FEMTO)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE XMRIG_ALGO_CN_FEMTO)
endif()

if (WITH_RANDOMX)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE XMRIG_ALGO_RANDOMX)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RANDOMX_FREQ_CFROUND=1)
endif()

if (WITH_ARGON2)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE XMRIG_ALGO_ARGON2)
endif()

if (WITH_KAWPOW)
    if (WITH_HIP_DRIVER_API)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE XMRIG_ALGO_KAWPOW)
    else()
        set(WITH_KAWPOW OFF)
        message(STATUS "HIP: KawPow disabled, requires WITH_HIP_DRIVER_API=ON for HIP Driver API and HIPRTC")
    endif()
endif()

if (WITH_CN_R)
    if (WITH_HIP_DRIVER_API)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE XMRIG_ALGO_CN_R)
    else()
        set(WITH_CN_R OFF)
        message(STATUS "HIP: CryptoNight-R disabled, requires WITH_HIP_DRIVER_API=ON for HIP Driver API and HIPRTC")
    endif()
endif()

if (XMRIG_OS_WIN)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        "$ENV{WindowsSdkDir}/Include/$ENV{WindowsSDKVersion}/um"
        "$ENV{WindowsSdkDir}/Include/$ENV{WindowsSDKVersion}/shared"
        "$ENV{WindowsSdkDir}/Include/$ENV{WindowsSDKVersion}/winrt"
        "$ENV{WindowsSdkDir}/Include/$ENV{WindowsSDKVersion}/cppwinrt"
    )
endif()

if (WITH_HIP_DRIVER_API AND WIN32)
    # HIPRTC DLLs are available in the ROCm bin directory; copy step disabled to avoid path mismatch during build
    # file(GLOB HIPRTCDLL "${ROCM_PATH}/bin/hiprtc*.dll")
    # add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${HIPRTCDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)

    # file(GLOB HIPRTCBUILTINDLL "${ROCM_PATH}/bin/hiprtc-builtins*.dll")
    # add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${HIPRTCBUILTINDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif()