// Extract all CPU R-states (R[0..7]) of iteration 0 into a set, then walk the
// emulator's group-start R states (ic=0) and report the first one NOT present
// in the CPU set => the group whose INPUT state was already wrong.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

struct R8 {
    uint64_t r[8];
    bool operator<(const R8& o) const { return memcmp(r, o.r, sizeof r) < 0; }
};

int main() {
    // proper two-pass: collect EXEC blocks of iteration 0
    FILE* f = fopen("stagecmp_nt.err", "rb");
    if (!f) { fprintf(stderr, "stagecmp_nt.err missing\n"); return 1; }
    char line[512];
    std::set<R8> cpuset;
    std::vector<R8> seq;
    {
        R8 blk; int n = 0; int pc = -1; int seen_pc0 = 0; bool collect = false;
        while (fgets(line, sizeof line, f)) {
            if (strncmp(line, "[CPU_TRACE] EXEC pc=", 20) == 0) {
                if (collect && n == 8) seq.push_back(blk);
                pc = atoi(line + 20);
                if (pc == 0) { seen_pc0++; collect = (seen_pc0 == 1); }
                n = 0;
                continue;
            }
            unsigned long long v;
            if (collect && sscanf(line, "  R[%*d]=%llx", &v) == 1) {
                if (n < 8) blk.r[n++] = v;
            }
        }
        if (collect && n == 8) seq.push_back(blk);
    }
    fclose(f);
    for (auto& s : seq) cpuset.insert(s);
    fprintf(stderr, "CPU iter0 R-states: %zu blocks, %zu unique\n", seq.size(), cpuset.size());

    // --- Walk emulator group states ---
    FILE* g = fopen("emu_g0_groups.txt", "r");
    if (!g) { fprintf(stderr, "emu_g0_groups.txt missing\n"); return 1; }
    char gl[1024];
    int gidx = 0, firstbad = -1; uint64_t badip = 0; R8 badr;
    while (fgets(gl, sizeof gl, g)) {
        R8 e; uint64_t ip;
        int m = sscanf(gl, "%llu %llx %llx %llx %llx %llx %llx %llx %llx",
            (unsigned long long*)&ip,
            (unsigned long long*)&e.r[0], (unsigned long long*)&e.r[1],
            (unsigned long long*)&e.r[2], (unsigned long long*)&e.r[3],
            (unsigned long long*)&e.r[4], (unsigned long long*)&e.r[5],
            (unsigned long long*)&e.r[6], (unsigned long long*)&e.r[7]);
        if (m != 9) continue;
        if (cpuset.find(e) == cpuset.end()) {
            firstbad = gidx; badip = ip; badr = e; break;
        }
        gidx++;
    }
    fclose(g);
    if (firstbad < 0) { printf("ALL emulator group-start R states present in CPU set (%d groups)\n", gidx); }
    else {
        printf("FIRST group-start R NOT in CPU set: group#%d ip=%llu\n", firstbad, (unsigned long long)badip);
        printf("  R: ");
        for (int i = 0; i < 8; ++i) printf("%016llx ", (unsigned long long)badr.r[i]);
        printf("\n");
    }
    return 0;
}
