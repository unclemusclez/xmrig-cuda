@echo off
rem Wave64-layout selftest for the champion lineage (RX_WAVE_SIZE=64, legacy
rem dispatch). Extra defines are appended via arguments, e.g.:
rem   build_selftest_w64.bat -DRX_PROGRAM_IN_GLOBAL
rem Base gate (E1): no extra args.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d P:\repos\xmrig-cuda
set PATH=C:\Program Files\AMD\ROCm\7.2\bin;%PATH%
hipcc tools\rx_selftest\selftest.cu -o tools\rx_selftest\selftest_w64.exe -O3 -fms-runtime-lib=dll --offload-arch=gfx1100 -mno-wavefrontsize64 ^
  -I src -I src/RandomX -I src/RandomX/monero -I .reference/RandomX/src ^
  -L build_ref/Release -lrandomx -lamdhip64 -lhiprtc -lAdvapi32 -DRX_DEBUG_STAGE -DRANDOMX_FREQ_CFROUND=1 -DRX_WAVE_SIZE=64 -DRX_LEGACY_DISPATCH %*
if %errorlevel% neq 0 exit /b %errorlevel%
echo BUILD OK
