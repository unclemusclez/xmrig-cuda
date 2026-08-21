// Analyze the IMUL_M divergence at pc=162 (cpu seq) / word 153 (emul).
// scratch = ic281 snapshot (identical cpu/emul). Find what memval each side
// effectively used, and where in the scratchpad that value lives.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>

static uint64_t mul_mod(uint64_t a, uint64_t b) { return a * b; }
static uint64_t mul_inv(uint64_t a) { // a must be odd; Newton lift
    uint64_t x = a; // works mod 8 if a odd? use standard: x = a (mod 8 ok for odd a? no—use 1)
    x = 1;
    for (int i = 0; i < 6; ++i) x = x * (2 - a * x);
    return x;
}

int main(int argc, char** argv)
{
    FILE* f = fopen("cpu_sp_ic.bin", "rb");
    std::vector<uint8_t> sp(2097152);
    if (!f || fread(sp.data(), 1, sp.size(), f) != sp.size()) { fprintf(stderr, "scratch read fail\n"); return 1; }
    fclose(f);

    uint64_t mem5d8 = *(uint64_t*)(sp.data() + 0x5D8);
    printf("scratch[0x5D8] = %016llx\n", (unsigned long long)mem5d8);

    uint64_t r0_old = strtoull("bccdc84c75bcf9ed", nullptr, 16);
    uint64_t r1 = strtoull("03bcac831e97833c", nullptr, 16); // r[1] at pc=162
    uint64_t cpu_new = strtoull("c26efcaaaa16e569", nullptr, 16);
    uint64_t emu_new = strtoull("146f80833f05ef24", nullptr, 16);

    printf("r0_old*mem5d8 = %016llx  (cpu_new=%016llx) %s\n",
           (unsigned long long)mul_mod(r0_old, mem5d8), (unsigned long long)cpu_new,
           mul_mod(r0_old, mem5d8) == cpu_new ? "MATCH" : "DIFF");

    uint64_t inv = mul_inv(r0_old);
    uint64_t mem_needed_cpu = cpu_new * inv;
    uint64_t mem_needed_emu = emu_new * inv;
    printf("memval implied by cpu result: %016llx\n", (unsigned long long)mem_needed_cpu);
    printf("memval implied by emu result: %016llx\n", (unsigned long long)mem_needed_emu);
    printf("(odd check r0_old&1=%llu)\n", (unsigned long long)(r0_old & 1));

    // verify inverse
    printf("inv check r0_old*inv=%016llx\n", (unsigned long long)(r0_old * inv));

    auto scan = [&](uint64_t v) {
        int found = 0;
        for (size_t a = 0; a + 8 <= sp.size(); a += 8) {
            if (*(uint64_t*)(sp.data() + a) == v) {
                if (found < 8) printf("  value found at addr %06zx\n", a);
                ++found;
            }
        }
        printf("  total occurrences: %d\n", found);
    };
    printf("scan for cpu-implied memval:\n"); scan(mem_needed_cpu);
    printf("scan for emu-implied memval:\n"); scan(mem_needed_emu);

    // also: what addr gives emu result if read there — check specific L1 window
    uint64_t addr_base = (uint32_t)r1 + (int32_t)0x50c342a0ULL;
    printf("cpu addr = %016llx & 0x3FF8 = %06llx\n",
           (unsigned long long)addr_base, (unsigned long long)(addr_base & 0x3FF8));
    uint64_t addr_gpu = (uint32_t)r1 + (int32_t)0x524342a0ULL;
    printf("gpu imm addr = %016llx & 0x3FF8 = %06llx\n",
           (unsigned long long)addr_gpu, (unsigned long long)(addr_gpu & 0x3FF8));
    return 0;
}
