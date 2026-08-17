cmake_minimum_required(VERSION 3.10)
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

if (WITH_CN_LITE)
    add_definitions(/DXMRIG_ALGO_CN_LITE)
endif()

if (WITH_CN_HEAVY)
    add_definitions(/DXMRIG_ALGO_CN_HEAVY)
endif()

if (WITH_CN_PICO)
    add_definitions(/DXMRIG_ALGO_CN_PICO)
endif()

if (WITH_CN_FEMTO)
    add_definitions(/DXMRIG_ALGO_CN_FEMTO)
endif()

if (WITH_RANDOMX)
    add_definitions(/DXMRIG_ALGO_RANDOMX)
endif()

if (WITH_ARGON2)
    add_definitions(/DXMRIG_ALGO_ARGON2)
endif()

if (WITH_KAWPOW)
    if (WITH_HIP_DRIVER_API)
        add_definitions(/DXMRIG_ALGO_KAWPOW)
    else()
        set(WITH_KAWPOW OFF)
        message(STATUS "HIP: KawPow disabled, requires WITH_HIP_DRIVER_API=ON for HIP Driver API and HIPRTC")
    endif()
endif()

if (WITH_CN_R)
    if (WITH_HIP_DRIVER_API)
        add_definitions(/DXMRIG_ALGO_CN_R)
    else()
        set(WITH_CN_R OFF)
        message(STATUS "HIP: CryptoNight-R disabled, requires WITH_HIP_DRIVER_API=ON for HIP Driver API and HIPRTC")
    endif()
endif()


include_directories(src)
add_definitions(/DCUB_IGNORE_DEPRECATED_CPP_DIALECT)


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
    # list(APPEND SOURCES res/app.rc)
endif()

if (XMRIG_OS_APPLE)
    cmake_policy(SET CMP0042 NEW)
endif()

add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} hip::host hiprtc::hiprtc roc::rocblas rocrand)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)

if (WITH_HIP_DRIVER_API AND WIN32)
    # HIPRTC DLLs are available in the ROCm bin directory; copy step disabled to avoid path mismatch during build
    # file(GLOB HIPRTCDLL "${ROCM_PATH}/bin/hiprtc*.dll")
    # add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${HIPRTCDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)

    # file(GLOB HIPRTCBUILTINDLL "${ROCM_PATH}/bin/hiprtc-builtins*.dll")
    # add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${HIPRTCBUILTINDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif()