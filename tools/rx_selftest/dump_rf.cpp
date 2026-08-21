// Compare emulator's final R/F/E/A (from emul_out's [00..31] table) is not persisted,
// so instead re-derive: dump the emulator final words to emul_final.txt via a re-run flag.
// Simpler: compare gpu_rf_p0.bin (GPU real) against cpu_rf_p0.bin region-by-region,
// and print both as words so we can eyeball vs emulator output.
#include <cstdio>
#include <cstdint>
#include <vector>

static std::vector<uint8_t> rd(const char* n) {
    FILE* f = fopen(n, "rb"); if (!f) { fprintf(stderr, "missing %s\n", n); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(sz); fread(v.data(), 1, sz, f); fclose(f); return v;
}

int main() {
    auto gpu = rd("gpu_rf_p0.bin");
    auto cpu = rd("cpu_rf_p0.bin");
    printf("idx   gpu                 cpu                 match\n");
    for (int i = 0; i < 32; ++i) {
        uint64_t g = ((uint64_t*)gpu.data())[i], c = ((uint64_t*)cpu.data())[i];
        printf("[%02d] %016llx %016llx %s\n", i, (unsigned long long)g, (unsigned long long)c, g == c ? "SAME" : "DIFF");
    }
    return 0;
}
