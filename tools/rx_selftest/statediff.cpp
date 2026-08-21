// Print full 24-word states for CPU seq indices a..b and diff consecutive.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>

struct S24 { int pc; uint64_t w[24]; };

int main(int argc, char** argv) {
    int a = argc > 1 ? atoi(argv[1]) : 14;
    int b = argc > 2 ? atoi(argv[2]) : 16;
    FILE* f = fopen("stagecmp_nt.err", "rb");
    char line[512];
    std::vector<S24> seq;
    int seen_pc0 = 0;
    S24 s; int rw = 0, fw = 0, ew = 0; bool collect = false, started = false;
    auto push = [&]() {
        if (collect && started && rw == 8 && fw == 8 && ew == 8) seq.push_back(s);
        rw = fw = ew = 0; started = false;
    };
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "[CPU_TRACE] EXEC pc=", 20) == 0) {
            push();
            int pc = atoi(line + 20);
            if (pc == 0) { seen_pc0++; collect = (seen_pc0 == 1); }
            if (collect) { s.pc = pc; started = true; }
            continue;
        }
        if (!collect || !started) continue;
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
    const char* names[24] = {"r0","r1","r2","r3","r4","r5","r6","r7",
                             "f0lo","f0hi","f1lo","f1hi","f2lo","f2hi","f3lo","f3hi",
                             "e0lo","e0hi","e1lo","e1hi","e2lo","e2hi","e3lo","e3hi"};
    for (int i = a; i <= b && i < (int)seq.size(); ++i) {
        printf("seq[%d] pc=%d\n", i, seq[i].pc);
        if (i > a) {
            for (int k = 0; k < 24; ++k) if (seq[i].w[k] != seq[i-1].w[k])
                printf("   %s: %016llx -> %016llx\n", names[k], (unsigned long long)seq[i-1].w[k], (unsigned long long)seq[i].w[k]);
        }
    }
    return 0;
}
