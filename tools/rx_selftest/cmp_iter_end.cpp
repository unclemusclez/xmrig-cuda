// Compare emulator ENDITER0 state vs CPU state at first instruction of iteration 1.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

struct S24 { int pc; uint64_t w[24]; };

int main() {
    FILE* f = fopen("stagecmp_nt.err", "rb");
    char line[512];
    std::vector<S24> seq;
    int seen_pc0 = 0;
    S24 s; int rw = 0, fw = 0, ew = 0; bool started = false;
    bool collect = true;  // collect iter0 and the first state of iter1
    auto push = [&]() {
        if (started && rw == 8 && fw == 8 && ew == 8) seq.push_back(s);
        rw = fw = ew = 0; started = false;
    };
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "[CPU_TRACE] EXEC pc=", 20) == 0) {
            push();
            int pc = atoi(line + 20);
            if (pc == 0) {
                seen_pc0++;
                if (seen_pc0 > 2) break;
            }
            s.pc = pc; started = true;
            continue;
        }
        if (!started) continue;
        unsigned long long v;
        if (sscanf(line, "  R[%*d]=%llx", &v) == 1) { if (rw < 8) s.w[rw++] = v; continue; }
        unsigned long long fh, fl, eh, el;
        if (sscanf(line, "  F[%*d]=%16llx%16llx E[%*d]=%16llx%16llx", &fh, &fl, &eh, &el) == 4) {
            if (fw < 7) { s.w[8 + fw] = fl; s.w[9 + fw] = fh; fw += 2; }
            if (ew < 7) { s.w[16 + ew] = el; s.w[17 + ew] = eh; ew += 2; }
            continue;
        }
    }
    push();
    fclose(f);
    fprintf(stderr, "collected %zu states (iter0 + iter1 start)\n", seq.size());
    if (seq.size() < 257) { fprintf(stderr, "not enough states\n"); return 1; }
    S24& cpu_iter1 = seq[256];  // first instruction state of iteration 1

    // emulator ENDITER0 line
    FILE* g = fopen("emu_g0_groups.txt", "r");
    char gl[4096];
    std::vector<uint64_t> emu;
    while (fgets(gl, sizeof gl, g)) {
        if (strncmp(gl, "ENDITER0", 8) != 0) continue;
        char* p = gl + 8;
        while (emu.size() < 24) {
            char* end = nullptr;
            unsigned long long v = strtoull(p, &end, 16);
            if (end == p) break;
            emu.push_back(v);
            p = end;
        }
        break;
    }
    fclose(g);
    if (emu.size() != 24) { fprintf(stderr, "ENDITER0 line bad (%zu words)\n", emu.size()); return 1; }

    const char* names[24] = {"r0","r1","r2","r3","r4","r5","r6","r7",
                             "f0lo","f0hi","f1lo","f1hi","f2lo","f2hi","f3lo","f3hi",
                             "e0lo","e0hi","e1lo","e1hi","e2lo","e2hi","e3lo","e3hi"};
    int nd = 0;
    for (int k = 0; k < 24; ++k) if (emu[k] != cpu_iter1.w[k]) nd++;
    printf("ENDITER0 vs CPU iter1-entry: %d/24 words differ\n", nd);
    for (int k = 0; k < 24; ++k) if (emu[k] != cpu_iter1.w[k])
        printf("  %-5s emul=%016llx cpu=%016llx\n", names[k], (unsigned long long)emu[k], (unsigned long long)cpu_iter1.w[k]);
    return 0;
}
