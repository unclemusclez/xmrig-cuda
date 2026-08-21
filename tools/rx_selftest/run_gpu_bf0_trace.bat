@echo off
setlocal EnableExtensions
cd /d P:\GitRepos\xmrig-cuda\tools\rx_selftest
set PATH=C:\Program Files\AMD\ROCm\7.2\bin;%PATH%

set SEED=0000000000000000000000000000000000000000000000000000000000000000
set BASE=000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b

rem bfactor=0; traced build (RX_TRACE_VM baked in)
selftest.exe %SEED% %BASE% 32 0 1> gpu_trace.out 2> gpu_trace.err
echo EXIT=%ERRORLEVEL% >> gpu_trace.err
exit /b %ERRORLEVEL%
