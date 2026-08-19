@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d P:\GitRepos\xmrig-cuda
set PATH=C:\Program Files\AMD\ROCm\7.2\bin;%PATH%
hipcc tools\rx_selftest\selftest.cu -o tools\rx_selftest\selftest.exe -O3 -fms-runtime-lib=dll --offload-arch=gfx1100 ^
  -I src -I src/RandomX -I src/RandomX/monero -I .reference/RandomX/src ^
  -L build_ref/Release -lrandomx -lamdhip64 -lhiprtc -lAdvapi32 -DRX_DEBUG_STAGE
if %errorlevel% neq 0 exit /b %errorlevel%
echo BUILD OK
