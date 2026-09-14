#include <hip/hip_runtime.h>
#include <cstdio>
#ifndef RX_WAVE_SIZE
#ifdef __AMDGCN_WAVEFRONT_SIZE
#define RX_WAVE_SIZE __AMDGCN_WAVEFRONT_SIZE
#else
#define RX_WAVE_SIZE (-777)
#endif
#endif
__global__ void probe(int* out) {
#ifdef __AMDGCN_WAVEFRONT_SIZE
    out[0] = __AMDGCN_WAVEFRONT_SIZE;
#else
    out[0] = -888;
#endif
    out[1] = RX_WAVE_SIZE;
    out[2] = (int)blockDim.x;
}
int main() {
    printf("host: __AMDGCN_WAVEFRONT_SIZE defined as RX_WAVE_SIZE=%d\n", RX_WAVE_SIZE);
    int* d; hipMalloc(&d, 3 * sizeof(int));
    probe<<<1, 32>>>(d);
    hipDeviceSynchronize();
    int h[3]; hipMemcpy(h, d, 3 * sizeof(int), hipMemcpyDeviceToHost);
    printf("device: WF=%d RX_WAVE_SIZE=%d blockDim=%d\n", h[0], h[1], h[2]);
    return 0;
}
