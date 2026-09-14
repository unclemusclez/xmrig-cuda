#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void probe()
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		const int ws = static_cast<int>(warpSize);
		printf("warpSize=%d blockDim=%d\n", ws, (int)blockDim.x);
	}
}

int main()
{
	probe<<<1, 32>>>();
	hipDeviceSynchronize();
	return 0;
}
