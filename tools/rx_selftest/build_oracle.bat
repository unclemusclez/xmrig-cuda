@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d P:\GitRepos\xmrig-cuda
cl /I.reference/RandomX/src /MD /EHsc /O2 tools\rx_selftest\oracle.cpp build_ref\Release\randomx.lib Advapi32.lib /Fe:tools\rx_selftest\oracle.exe
if %errorlevel% neq 0 exit /b %errorlevel%
echo BUILD OK
