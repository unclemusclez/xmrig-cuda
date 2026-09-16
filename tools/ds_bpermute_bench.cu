// Cross-lane primitive latency microbench for RandomX M2 (register promotion).
// Measures dependent-chain latency of ds_bpermute vs plain LDS load on the
// current device via host-side timing (single wave -> wall time per chain
// step IS the dependent latency).
//
// Build (Windows, ROCm 7.2):
//   hipcc tools/ds_bpermute_bench.cu -o tools/ds_bpermute_bench.exe -O3 --offload-arch=gfx1100 -mno-wavefrontsize64 -fms-runtime-lib=dll
#include <hip/hip_runtime.h>
#include <cstdio>

#define ITERS 2000000

__global__ void bench_bpermute(uint32_t* out)
{
	uint32_t peer = ((threadIdx.x + 3) & 31) << 2;
	uint32_t v = threadIdx.x + 1;

	#pragma unroll 4
	for (int i = 0; i < ITERS; ++i)
		v = __builtin_amdgcn_ds_bpermute(peer, v);

	if (threadIdx.x == 0) *out = v;
}

__global__ void bench_lds(uint32_t* out)
{
	__shared__ uint32_t buf[32];
	buf[threadIdx.x] = threadIdx.x + 1;
	__syncthreads();

	uint32_t v = threadIdx.x;

	#pragma unroll 4
	for (int i = 0; i < ITERS; ++i)
		v = buf[v & 31];

	if (threadIdx.x == 0) *out = v;
}

__global__ void bench_shfl(uint32_t* out)
{
	int peer = ((threadIdx.x + 3) & 31);
	uint32_t v = threadIdx.x + 1;

	#pragma unroll 4
	for (int i = 0; i < ITERS; ++i)
		v = __shfl(v, peer);

	if (threadIdx.x == 0) *out = v;
}

template<typename K>
double run(K k, uint32_t* d_out, const char* name)
{
	hipEvent_t a, b;
	hipEventCreate(&a); hipEventCreate(&b);
	k<<<1, 32>>>(d_out);
	hipEventRecord(a);
	k<<<1, 32>>>(d_out);
	hipEventRecord(b);
	hipEventSynchronize(b);
	float ms = 0; hipEventElapsedTime(&ms, a, b);
	printf("%-24s: %7.1f ns/dep-step (%.1f cyc @2.2GHz)\n", name, ms * 1e6 / ITERS, ms * 2.2);
	return ms;
}

int main()
{
	uint32_t* d_out = nullptr;
	hipMalloc(&d_out, 4);

	hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
	printf("device: %s (%s), %d CUs, wave %d\n", p.name, p.gcnArchName, p.multiProcessorCount, p.warpSize);

	run(bench_lds,      d_out, "LDS dependent chain");
	run(bench_shfl,     d_out, "__shfl chain");
	run(bench_bpermute, d_out, "ds_bpermute chain");

	hipFree(d_out);
	return 0;
}
