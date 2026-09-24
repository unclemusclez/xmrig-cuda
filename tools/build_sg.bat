@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d P:\repos\xmrig-cuda
set PATH=C:\Program Files\AMD\ROCm\7.2\bin;%PATH%
cmake -B build_sg -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DRX_WAVE_SIZE=64 -DWITH_RX_LEGACY_DISPATCH=ON -DWITH_RX_PROGRAM_IN_GLOBAL=OFF -DWITH_RX_M3_PROMOTE=OFF -DWITH_RX_M3_PROMOTE_FE=OFF -DWITH_ARGON2=OFF -DWITH_KAWPOW=OFF -DWITH_CN_LITE=OFF -DWITH_CN_HEAVY=OFF -DWITH_CN_PICO=OFF -DWITH_CN_FEMTO=OFF -DWITH_CN_R=OFF
if %errorlevel% neq 0 exit /b %errorlevel%
ninja -C build_sg xmrig-rocm
