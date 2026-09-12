@echo off
rem Pristine reference build (no RX_TRACE_VM instrumentation).
rem For traced CPU debugging: apply tools\rx_selftest\randomx_trace.patch to
rem .reference/RandomX and add "/DRX_TRACE_VM [/DRX_TRACE_ITER_ONLY]" below.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d P:\GitRepos\xmrig-cuda
cmake -B build_ref -G "Visual Studio 17 2022" -A x64 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_CXX_FLAGS="" .reference/RandomX
cmake --build build_ref --config Release --target randomx
