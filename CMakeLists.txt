cmake_minimum_required(VERSION 3.10)
project(xmrig-rocm LANGUAGES CXX HIP)
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocrand REQUIRED)
include(cmake/CUDA-Version.cmake)


option(WITH_DRIVER_API "Enable CUDA Driver API and NVRTC, required for cn/r and kawpow algorithms" ON)

# Algorithm selection
option(WITH_CN_R       "Enable CryptoNight-R algorithm" ON)
option(WITH_CN_LITE    "Enable CryptoNight-Lite algorithms family" ON)
option(WITH_CN_HEAVY   "Enable CryptoNight-Heavy algorithms family" ON)
option(WITH_CN_PICO    "Enable CryptoNight-Pico algorithm" ON)
option(WITH_CN_FEMTO   "Enable CryptoNight-UPX2 algorithm" ON)
option(WITH_ARGON2     "Enable Argon2 algorithms family" OFF) #unsupported

if (CUDA_VERSION VERSION_LESS 9.0)
    message(STATUS "CUDA ${CUDA_VERSION}: RandomX and KawPow disabled, they do not work with old CUDA")
    option(WITH_RANDOMX  "Enable RandomX algorithms family" OFF)
    option(WITH_KAWPOW   "Enable KawPow algorithms family" OFF)
else()
    option(WITH_RANDOMX  "Enable RandomX algorithms family" ON)
    option(WITH_KAWPOW   "Enable KawPow algorithms family" ON)
endif()

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
    if (WITH_DRIVER_API)
        add_definitions(/DXMRIG_ALGO_KAWPOW)
    else()
        set(WITH_KAWPOW OFF)
        message(STATUS "CUDA ${CUDA_VERSION}: KawPow disabled, requires WITH_DRIVER_API=ON for CUDA Driver API and NVRTC")
    endif()
endif()

if (WITH_CN_R)
    if (WITH_DRIVER_API)
        add_definitions(/DXMRIG_ALGO_CN_R)
    else()
        set(WITH_CN_R OFF)
        message(STATUS "CUDA ${CUDA_VERSION}: CryptoNight-R disabled, requires WITH_DRIVER_API=ON for CUDA Driver API and NVRTC")
    endif()
endif()


include_directories(src)
add_definitions(/DCUB_IGNORE_DEPRECATED_CPP_DIALECT)


include(cmake/cpu.cmake)
include(cmake/os.cmake)
include(cmake/flags.cmake)
# include(cmake/CUDA.cmake) # Removed as we're now using native CUDA support


set(SOURCES
    src/crypto/cn/c_blake256.c
    src/crypto/common/Algorithm.cpp
    src/crypto/common/Algorithm.h
    src/version.h
    src/xmrig-cuda.cpp
    src/xmrig-cuda.h
    )


if (WITH_DRIVER_API AND WITH_CN_R)
    list(APPEND SOURCES src/CudaCryptonightR_gen.cpp)
endif()

if (XMRIG_OS_WIN)
    list(APPEND SOURCES res/app.rc)
endif()

if (XMRIG_OS_APPLE)
    cmake_policy(SET CMP0042 NEW)
endif()

add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} xmrig-rocm ${LIBS} hip::host roc::rocblas rocrand)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)

if (WITH_DRIVER_API AND WIN32)
    if (CUDA_VERSION VERSION_LESS 10.0)
        file(GLOB NVRTCDLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvrtc64*.dll")
    else()
        file(GLOB NVRTCDLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvrtc64*_0.dll")
    endif()

    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NVRTCDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)

    file(GLOB NVRTCBUILTINDLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvrtc-builtins64*.dll")
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NVRTCBUILTINDLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif()
