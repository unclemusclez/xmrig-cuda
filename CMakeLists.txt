cmake_minimum_required(VERSION 3.16)
project(xmrig-rocm LANGUAGES C CXX HIP)
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocrand REQUIRED)

# Linux/other platforms: leave CMAKE_HIP_ARCHITECTURES unset so CMake auto-detects
# the native GPU via rocm_agent_enumerator; override with -DCMAKE_HIP_ARCHITECTURES=<gfx>.


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

# RandomX dispatch mode: legacy LDS ip/fprc round-trip (tag 0.0.2 behavior,
# best known performance; keeps VM_STATE_SIZE at 2048 B so execute_vm fits
# 8 blocks/CU on gfx1100) vs experimental fast dispatch.
option(WITH_RX_LEGACY_DISPATCH "Use legacy LDS dispatch for RandomX ip/fprc" ON)

# LDS occupancy: fetch the read-only compiled RandomX program from global
# memory (L1-hot) instead of staging it in LDS. Halves the per-block LDS
# footprint, doubling blocks/CU on wave64 targets (gfx906: 4 -> 8).
option(WITH_RX_PROGRAM_IN_GLOBAL "Keep RandomX compiled program in global cache instead of LDS" OFF)

# M3 register promotion: promote R[sub] to VGPR, cross-lane src via shfl.
option(WITH_RX_M3_PROMOTE "M3: promote integer register to VGPR with shfl cross-lane" OFF)
option(WITH_RX_M3_PROMOTE_FE "M3b: also promote F/E registers to VGPRs" OFF)

# RandomX wavefront size: gfx6xx-gfx9xx execute wave64, gfx10+ execute wave32.
# The RandomX VM kernels launch one wavefront per block (RX_WAVE_SIZE threads),
# and host-side launch geometry must match the device, so this is passed as a
# define for both host and device compilation. Pass -DRX_WAVE_SIZE=64 to force
# wave64 on RDNA3 (also add -mwavefrontsize64 to CMAKE_HIP_FLAGS then).
if (NOT DEFINED RX_WAVE_SIZE)
    if (CMAKE_HIP_ARCHITECTURES MATCHES "gfx[6789]")
        set(RX_WAVE_SIZE 64)
    else()
        set(RX_WAVE_SIZE 32)
    endif()
endif()
message(STATUS "RandomX: RX_WAVE_SIZE=${RX_WAVE_SIZE} for CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")


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
    if (NOT DEFINED ENV{WindowsSdkDir} OR "$ENV{WindowsSdkDir}" STREQUAL "")
        set(WindowsSdkDir "C:/Program Files (x86)/Windows Kits/10" CACHE PATH "Fallback Windows SDK root")
    else()
        set(WindowsSdkDir "$ENV{WindowsSdkDir}" CACHE PATH "Windows SDK root" FORCE)
    endif()
    if (NOT DEFINED ENV{WindowsSDKVersion} OR "$ENV{WindowsSDKVersion}" STREQUAL "")
        set(WindowsSDKVersion "10.0.26100.0" CACHE STRING "Fallback Windows SDK version" FORCE)
    else()
        set(WindowsSDKVersion "$ENV{WindowsSDKVersion}" CACHE STRING "Windows SDK version" FORCE)
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
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RX_WAVE_SIZE=${RX_WAVE_SIZE})

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if (WIN32)
    # Windows target is gfx1100 (RDNA3, native wave32). LAYOUT and CODEGEN wave
    # size are independent: RX_WAVE_SIZE controls kernel geometry (threads/block,
    # hashes/block, LDS staging) while codegen must stay wave32 -- RX_WAVE_SIZE=64
    # layouts run as 64-thread blocks of two wave32 halves and that is the
    # winning geometry (8 hashes/block x 2048 B = exactly 4 blocks/CU).
    # Compiling wave64 codegen on RDNA3 halves SIMD throughput (v0.0.5-era
    # experiments: coherent wave64 measured slower in every configuration).
    # Linux (gfx906) never enters this block; hipcc emits native wave64 there.
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
    if (WITH_RX_LEGACY_DISPATCH)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RX_LEGACY_DISPATCH)
    endif()
    if (WITH_RX_PROGRAM_IN_GLOBAL)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RX_PROGRAM_IN_GLOBAL)
    endif()
    if (WITH_RX_M3_PROMOTE)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RX_M3_PROMOTE)
    endif()
    if (WITH_RX_M3_PROMOTE_FE)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RX_M3_PROMOTE_FE)
    endif()
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