// Confirm the store->load race: ISTORE raw155 writes cell that IMUL_M raw162 reads.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>

int main()
{
    FILE* f = fopen("cpu_ic281_seq.txt", "r");
    std::vector<std::vector<unsigned long long>> seq;
    char line[16384];
    while (fgets(line, sizeof line, f)) {
        std::vector<unsigned long long> w(24, 0);
        char* tok = strtok(line, " \r\n");
        tok = strtok(nullptr, " \r\n");
        int n = 0;
        while (tok && n < 24) { w[n++] = strtoull(tok, nullptr, 16); tok = strtok(nullptr, " \r\n"); }
        seq.push_back(w);
    }
    fclose(f);

    // ISTORE raw155: dst=r7 (addr base), src=r0 (value), mod=63, imm=5a9cb344
    // mask: modCond=3<14, mod&3=3 -> L1 0x3FF8
    uint64_t r7_155 = seq[155][7];      // r7 index=7
    uint64_t r0_155 = seq[155][0];      // r0 index=0
    uint64_t addr_store = ((uint32_t)r7_155 + (int32_t)0x5a9cb344) & 0x3FF8ULL;
    printf("ISTORE raw155: r7=%016llx imm=5a9cb344 -> addr=%06llx (want 5d8)\n",
           (unsigned long long)r7_155, (unsigned long long)addr_store);
    printf("ISTORE raw155 stores r0=%016llx\n", (unsigned long long)r0_155);

    // IMUL_M raw162: dst=r0, src=r1 (addr base), mod=34, imm=50c342a0
    // mask: mod&3=2 -> L1 0x3FF8
    uint64_t r1_162 = seq[162][1];
    uint64_t r0_162 = seq[162][0];
    uint64_t addr_load = ((uint32_t)r1_162 + (int32_t)0x50c342a0) & 0x3FF8ULL;
    printf("IMUL_M raw162: r1=%016llx imm=50c342a0 -> addr=%06llx (want 5d8)\n",
           (unsigned long long)r1_162, (unsigned long long)addr_load);
    printf("IMUL_M raw162: r0(old)=%016llx\n", (unsigned long long)r0_162);
    printf("IMUL_M raw162 cpu result=%016llx ; r0*r0_if_read_stored=%016llx\n",
           (unsigned long long)seq[163][0], (unsigned long long)(r0_162 * r0_155));
    printf("match-if-read-stored? %s\n", seq[163][0] == r0_162 * r0_155 ? "YES" : "no");
    printf("store-value r0_155 == r0_162(old)? %s\n", r0_155 == r0_162 ? "YES" : "no");

    // group membership check: is word150 (ISTORE) in same group as word153 (IMUL_M)?
    std::vector<uint8_t> v(2048);
    FILE* g = fopen("gpu_vmstate_p2.bin", "rb");
    fread(v.data(), 1, 2048, g); fclose(g);
    uint32_t* comp = (uint32_t*)(v.data() + 1024);
    int ip = 0;
    while (ip < 256) {
        uint32_t h = comp[ip];
        int nw = (h >> 24) & 7, nf = (h >> 28) & 7;
        int span = (nw - nf) + 1;
        if (ip <= 150 && 150 < ip + span) {
            printf("word150 (ISTORE) in group ip=%d span=%d (words %d..%d)\n", ip, span, ip, ip + span - 1);
        }
        if (ip <= 153 && 153 < ip + span) {
            printf("word153 (IMUL_M) in group ip=%d span=%d (words %d..%d)\n", ip, span, ip, ip + span - 1);
        }
        ip += span;
    }
    return 0;
}
