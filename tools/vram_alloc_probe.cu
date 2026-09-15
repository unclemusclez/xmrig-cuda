#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    size_t free_b, total_b;
    hipMemGetInfo(&free_b, &total_b);
    printf("free %.2f GB / total %.2f GB\n", free_b/1e9, total_b/1e9);

    const size_t GB = 1073741824ull;
    for (size_t gb = 1; gb <= 28; ++gb) {
        void* p = nullptr;
        hipError_t e = hipMalloc(&p, gb * GB);
        printf("hipMalloc %2zu GB: %s\n", gb, e == hipSuccess ? "OK" : hipGetErrorString(e));
        if (p) hipFree(p);
        if (e != hipSuccess) {
            size_t lo = (gb-1)*GB, hi = gb*GB;
            for (int i = 0; i < 6 && hi - lo > 16*1024*1024; ++i) {
                size_t mid = (lo + hi) / 2;
                hipError_t e2 = hipMalloc(&p, mid);
                printf("  bisect %.3f GB: %s\n", mid/(double)GB, e2 == hipSuccess ? "OK" : hipGetErrorString(e2));
                if (p) { hipFree(p); p = nullptr; }
                if (e2 == hipSuccess) lo = mid; else hi = mid;
            }
            printf("=> single-allocation wall between %.3f and %.3f GB\n", lo/(double)GB, hi/(double)GB);
            break;
        }
    }
    return 0;
}