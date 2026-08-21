// Compare full 24-word (R+F+E) states: emulator group boundaries (ic=0) vs CPU
// per-instruction states of iteration 0. Reports first emulator state absent
// from the CPU set.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <vector>
#include <string>

struct S24 {
    uint64_t w[24];
    bool operator<(const S24& o) const { return memcmp(w, o.w, sizeof w) < 0; }
};

int main(int argc, char** argv) {
    const char* errfile = argc > 1 ? argv[1] : "stagecmp_nt.err";
    FILE* f = fopen(errfile, "rb");
    if (!f) { fprintf(stderr, "%s missing\n", errfile); return 1; }
    char line[512];
    std::vector<S24> seq;
    int seen_pc0 = 0;
    // accumulate one EXEC block
    S24 s; int rw = 0, fw = 0, ew = 0;  // counts (R words, F words, E words)
    bool collect = false;
    auto push = [&]() {
        if (collect && rw == 8 && fw == 8 && ew == 8) seq.push_back(s);
        rw = fw = ew = 0;
    };
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "[CPU_TRACE] EXEC pc=", 20) == 0) {
            push();
            int pc = atoi(line + 20);
            if (pc == 0) { seen_pc0++; collect = (seen_pc0 == 1); }
            continue;
        }
        if (!collect) continue;
        unsigned long long v;
        if (sscanf(line, "  R[%*d]=%llx", &v) == 1) { if (rw < 8) s.w[rw++] = v; continue; }
        unsigned long long fh, fl, eh, el;
        if (sscanf(line, "  F[%*d]=%16llx%16llx E[%*d]=%16llx%16llx", &fh, &fl, &eh, &el) == 4) {
            if (fw < 7) { s.w[8 + fw] = fl; s.w[8 + fw + 1] = fh; fw += 2; }
            if (ew < 7) { s.w[16 + ew] = el; s.w[16 + ew + 1] = eh; ew += 2; }
            continue;
        }
    }
    push();
    fclose(f);
    std::set<S24> cpuset(seq.begin(), seq.end());
    fprintf(stderr, "CPU iter0 states: %zu blocks, %zu unique\n", seq.size(), cpuset.size());

    FILE* g = fopen("emu_g0_groups.txt", "r");
    if (!g) { fprintf(stderr, "emu_g0_groups.txt missing\n"); return 1; }
    char gl[4096];
    int gi = 0;
    while (fgets(gl, sizeof gl, g)) {
        S24 e;
        char label[32] = {0};
        unsigned long long vals[24];
        int n = 0;
        char* p = gl;
        // first token: integer ip or ENDITER0
        char* sp = strchr(gl, ' ');
        if (!sp) continue;
        size_t ll = sp - gl; if (ll > 31) ll = 31;
        memcpy(label, gl, ll); label[ll] = 0;
        p = sp;
        while (n < 24) {
            char* end = nullptr;
            unsigned long long v = strtoull(p, &end, 16);
            if (end == p) break;
            vals[n++] = v;
            p = end;
        }
        if (n != 24) { fprintf(stderr, "short emul line (%d words) at group %d\n", n, gi); gi++; continue; }
        for (int i = 0; i < 24; ++i) e.w[i] = vals[i];
        bool in = cpuset.find(e) != cpuset.end();
        printf("group%-4d %-9s %s\n", gi, label, in ? "in-CPU-set" : "*** NOT in CPU set ***");
        if (!in) {
            printf("  R:"); for (int i = 0; i < 8; ++i) printf(" %016llx", e.w[i]); printf("\n");
            printf("  F:"); for (int i = 8; i < 16; ++i) printf(" %016llx", e.w[i]); printf("\n");
            printf("  E:"); for (int i = 16; i < 24; ++i) printf(" %016llx", e.w[i]); printf("\n");
            break;
        }
        gi++;
    }
    fclose(g);
    return 0;
}
