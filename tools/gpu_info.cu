// Portable GPU fact probe for xmrig-cuda RandomX (Windows/Linux, any HIP arch).
// Reports device facts + computes RandomX intensity limits and in-flight-hash
// capacity for both LDS layouts.
//
// Build (Windows, ROCm 7.2):
//   hipcc tools/gpu_info.cu -o tools/gpu_info.exe -O2 -fms-runtime-lib=dll --offload-arch=gfx1100 -mno-wavefrontsize64 -lamdhip64
// Build (Linux):
//   hipcc tools/gpu_info.cu -o tools/gpu_info -O2 --offload-arch=gfx906 -lamdhip64
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

#define HIP_OK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %d (%s) at line %d\n", (int)e, hipGetErrorString(e), __LINE__); return 1; } } while (0)

int main()
{
	int dev = 0;
	HIP_OK(hipSetDevice(dev));
	hipDeviceProp_t p;
	HIP_OK(hipGetDeviceProperties(&p, dev));

	printf("============= GPU INFO (HIP runtime) =============\n");
	printf("name                 : %s\n", p.name);
#if defined(__HIP_PLATFORM_AMD__)
	printf("gcnArchName          : %s\n", p.gcnArchName);
#endif
	printf("compute units        : %d\n", p.multiProcessorCount);
	printf("wavefront size       : %d\n", p.warpSize);
	printf("clock rate           : %d MHz\n", p.clockRate / 1000);
	printf("mem clock            : %d MHz\n", p.memoryClockRate / 1000);
	printf("mem bus width        : %d bit\n", p.memoryBusWidth);
	printf("total VRAM           : %.1f GB\n", p.totalGlobalMem / 1073741824.0);
	printf("shared mem / block   : %zu bytes\n", p.sharedMemPerBlock);
	printf("regs / block         : %zu\n", p.regsPerBlock);
	printf("max threads / block  : %d\n", p.maxThreadsPerBlock);
	printf("max threads / CU     : %d\n", p.maxThreadsPerMultiProcessor);
	printf("max waves / CU       : %d\n", p.maxThreadsPerMultiProcessor / p.warpSize);

	size_t free_b = 0, total_b = 0;
	HIP_OK(hipMemGetInfo(&free_b, &total_b));
	printf("VRAM now: free %.1f GB / total %.1f GB (runtime view)\n",
		free_b / 1073741824.0, total_b / 1073741824.0);

	// ---- RandomX arithmetic ----
	const double dataset_bytes = 2147483648.0;      // 2 GiB dataset
	const double scratch_bytes = 2097152.0;         // 2 MiB scratchpad per hash
	const double margin        = 2147483648.0;      // VA-aperture safety margin

	double usable = (double)free_b - dataset_bytes - margin;
	if (usable < 0) usable = 0;
	long long max_intensity = ((long long)(usable / scratch_bytes));
	max_intensity -= max_intensity % 32;            // batch must be multiple of 32
	printf("\n--- RandomX intensity budget ---\n");
	printf("max safe intensity   : %lld (free VRAM minus dataset minus %.0f GB margin)\n",
		max_intensity, margin / 1073741824.0);

	// ---- In-flight-hash capacity (execute_vm) ----
	// 8 lanes per hash; one block = one wavefront.
	const int cu = p.multiProcessorCount;
	const int wave = p.warpSize;
	const int hashes_per_block = wave / 8;
	const long long REG_IMM = 256 + 768;            // REGISTERS_SIZE + IMM_BUF_SIZE
	const long long prog = 1024;                    // compiled program bytes
	const long long flags = 128;                    // group flags (4 bits/word, Monero)
	const size_t lds_per_cu = 65536;

	long long state_full = REG_IMM + prog + flags;  // current default layout
	long long state_pig  = REG_IMM;                 // program-in-global layout
	auto cap = [&](long long bytes_per_hash, int max_waves_per_cu) {
		long long blocks_by_lds = (long long)lds_per_cu / (bytes_per_hash * hashes_per_block);
		long long blocks = blocks_by_lds < (long long)max_waves_per_cu ? blocks_by_lds : (long long)max_waves_per_cu;
		return cu * blocks * hashes_per_block;
	};
	int max_waves = p.maxThreadsPerMultiProcessor / p.warpSize;

	printf("\n--- in-flight hash capacity (before queueing) ---\n");
	printf("default LDS layout (%lld B/hash):  ~%lld hashes in flight (saturation ~intensity %lld)\n",
		state_full, cap(state_full, max_waves), cap(state_full, max_waves));
	printf("program-in-global  (%lld B/hash):   ~%lld hashes in flight (saturation ~intensity %lld)\n",
		state_pig, cap(state_pig, max_waves), cap(state_pig, max_waves));
	printf("(hardware max waves/CU = %d; LDS per CU = %zu B; hashes/block = %d)\n",
		max_waves, lds_per_cu, hashes_per_block);
	printf("============= END =============\n");
	return 0;
}
