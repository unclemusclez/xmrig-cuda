// Host-side check: compute getSmallPositiveFloatBits(entropy[0..7]) and compare
// against cpu_rf_p0.bin / gpu_rf_p0.bin A-region (bytes 192..255).
#include <cstdint>
#include <cstdio>
#include <cstring>

static uint64_t gspfb(uint64_t entropy)
{
    const uint64_t mantissaMask = (1ULL << 52) - 1;
    const int exponentBias = 1023;
    const uint64_t exponentMask = (1ULL << 11) - 1;
    uint64_t exponent = entropy >> 59;
    uint64_t mantissa = entropy & mantissaMask;
    exponent += exponentBias;
    exponent &= exponentMask;
    exponent <<= 52;
    return exponent | mantissa;
}

int main()
{
    // The VM's actual program (entropyBuffer at bytes 0..127) — this is the true
    // program entropy the CPU uses (th evolves through initScratchpad before run).
    FILE* f = fopen("cpu_prog_p0.bin", "rb");
    if (!f) { fprintf(stderr, "cpu_prog_p0.bin missing\n"); return 1; }
    uint64_t ent[8];
    if (fread(ent, 8, 8, f) != 8) return 1;
    fclose(f);

    uint8_t cpu_rf[256], gpu_rf[256];
    FILE* fc = fopen("cpu_rf_p0.bin", "rb"); if (!fc) return 1;
    if (fread(cpu_rf, 1, 256, fc) != 256) return 1; fclose(fc);
    FILE* fg = fopen("tools/rx_selftest/gpu_rf_p0.bin", "rb"); if (!fg) return 1;
    if (fread(gpu_rf, 1, 256, fg) != 256) return 1; fclose(fg);

    printf("%-4s %-18s %-18s %-18s %-7s %-7s\n", "i", "expected", "cpu_a", "gpu_a", "cpuOK", "gpuOK");
    for (int i = 0; i < 8; ++i) {
        uint64_t exp = gspfb(ent[i]);
        uint64_t ca, ga;
        memcpy(&ca, cpu_rf + 192 + i * 8, 8);
        memcpy(&ga, gpu_rf + 192 + i * 8, 8);
        printf("a%d   %016llx   %016llx   %016llx   %-7s %-7s  (ent=%016llx)\n",
               i, (unsigned long long)exp, (unsigned long long)ca, (unsigned long long)ga,
               ca == exp ? "OK" : "BAD", ga == exp ? "OK" : "BAD", (unsigned long long)ent[i]);
    }

    // Also check the r region: GPU r regs should equal CPU r regs if execute_vm matched.
    int rmatch = memcmp(cpu_rf, gpu_rf, 64) == 0;
    printf("r-region cpu==gpu: %s\n", rmatch ? "YES" : "NO");
    return 0;
}
