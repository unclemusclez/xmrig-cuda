#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

__global__ void k(int* out) {
	*out = (int)sizeof(long double);
}
int main() {
	int sz = 0;
	int* d; hipMalloc(&d, 4); hipMemcpy(d, &sz, 4, hipMemcpyHostToDevice);
	k<<<1,1>>>(d); hipDeviceSynchronize();
	hipMemcpy(&sz, d, 4, hipMemcpyDeviceToHost);
	printf("sizeof(long double) on GPU = %d bytes\n", sz);
	// precision check
	double a = 1.0;
	volatile double dres = a + 1e-20;
	long double la = 1.0L;
	volatile long double ldres = la + 1e-20L;
	printf("double  1.0+1e-20 = %.20g\n", (double)dres);
	printf("long double 1.0+1e-20 = %.20Lg\n", (long double)ldres);
	printf("long double has more precision: %s\n", ((long double)ldres != 1.0L) ? "YES" : "NO");
	return 0;
}
