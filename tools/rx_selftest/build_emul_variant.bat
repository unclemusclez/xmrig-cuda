@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d P:\GitRepos\xmrig-cuda
cl /O2 /EHsc /I src/RandomX tools\rx_selftest\emul_variant.cpp /Fe:tools\rx_selftest\emul_variant.exe
if %errorlevel% neq 0 exit /b %errorlevel%
echo BUILD OK
