@echo off
echo #include ^<hip/hip_runtime.h^> > test_device.cu
echo __device__ double test_kernel() { >> test_device.cu
echo     return __fma_rd(1.0, 2.0, 3.0); >> test_device.cu
echo } >> test_device.cu