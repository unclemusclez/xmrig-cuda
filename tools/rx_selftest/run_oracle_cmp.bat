@echo off
setlocal EnableExtensions
cd /d P:\GitRepos\xmrig-cuda\tools\rx_selftest
set PATH=C:\Program Files\AMD\ROCm\7.2\bin;%PATH%

set SEED=0000000000000000000000000000000000000000000000000000000000000000
set BASE=000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b

echo === oracle.exe (tevador reference) ===
oracle.exe %SEED% %BASE% 32 1> oracle_now.txt 2> oracle_now.err
echo ORACLE_EXIT=%ERRORLEVEL%

echo === comparison vs gpu_out.txt ===
fc /L oracle_now.txt gpu_out.txt > nul && echo MATCH || echo MISMATCH
exit /b 0
