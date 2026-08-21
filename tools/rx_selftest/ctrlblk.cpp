// Compare vmstate control blocks (gpu_vmstate_pX.bin) against entropy-derived
// CPU initialize() values (cpu_prog_pX.bin).
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

static std::vector<uint8_t> rd(const char* p)
{
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(sz);
    if (fread(v.data(), 1, sz, f) != (size_t)sz) exit(1);
    fclose(f);
    return v;
}

int main()
{
    const uint64_t CacheLineAlignMask = 0x7FFFFFC0ULL; // (2GB-1) & ~63
    const uint64_t ScratchpadL3Mask64 = 0x1FFFFFULL;
    for (int p = 0; p < 8; ++p) {
        char np[64], nv[64];
        snprintf(np, sizeof np, "cpu_prog_p%d.bin", p);
        snprintf(nv, sizeof nv, "gpu_vmstate_p%d.bin", p);
        FILE* t = fopen(np, "rb"); if (!t) { printf("p%d: (no file)\n", p); continue; } fclose(t);
        auto prog = rd(np);
        auto vm = rd(nv);
        uint64_t e[16];
        for (int i = 0; i < 16; ++i) e[i] = *(uint64_t*)(prog.data() + 8 * i);

        uint32_t ma  = *(uint32_t*)(vm.data() + 128);
        uint32_t mx  = *(uint32_t*)(vm.data() + 132);
        uint32_t adr = *(uint32_t*)(vm.data() + 136);
        uint32_t dso = *(uint32_t*)(vm.data() + 140);
        uint32_t plen = *(uint32_t*)(vm.data() + 160);

        uint64_t exp_ma = e[8] & CacheLineAlignMask;
        uint64_t exp_mx_raw = e[10];
        uint64_t exp_mx_masked = e[10] & CacheLineAlignMask;
        uint32_t exp_dso = (uint32_t)((e[13] & 0x7FFFF) << 6);
        printf("p%d:\n", p);
        printf("  e8 =%016llx e10=%016llx e12=%016llx e13=%016llx\n",
               (unsigned long long)e[8], (unsigned long long)e[10], (unsigned long long)e[12], (unsigned long long)e[13]);
        printf("  ma : vm=%08x exp=%016llx %s\n", ma, (unsigned long long)exp_ma,
               ((uint64_t)ma == (exp_ma & 0xFFFFFFFF)) ? "OK" : ( ((uint64_t)ma == exp_ma) ? "OK" : "DIFF"));
        printf("  mx : vm=%08x raw_lo=%08x masked_lo=%08x  %s\n", mx,
               (uint32_t)exp_mx_raw, (uint32_t)exp_mx_masked,
               (mx == (uint32_t)exp_mx_raw) ? "vm==raw" : ((mx == (uint32_t)exp_mx_masked) ? "vm==raw&0x7FFFFFC0" : "vm!=either"));
        printf("  adr: vm=%08x e12lo=%08x\n", adr, (uint32_t)e[12]);
        printf("  dso: vm=%08x exp=%08x %s\n", dso, exp_dso, (dso == exp_dso) ? "OK" : "DIFF");
        printf("  plen=%u spAddr0_raw=%08x spAddr0_masked=%08x\n", plen,
               (uint32_t)(exp_mx_raw & ScratchpadL3Mask64), (uint32_t)(exp_mx_masked & ScratchpadL3Mask64));
    }
    return 0;
}
