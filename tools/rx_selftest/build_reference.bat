@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d P:\GitRepos\xmrig-cuda
cmake -B build_ref -G "Visual Studio 17 2022" -A x64 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded .reference/RandomX
cmake --build build_ref --config Release --target randomx
