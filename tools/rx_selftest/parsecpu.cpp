// Parse stagecmp_nt.err CPU_TRACE output: extract the state at each "EXEC pc=0"
// (= iteration start, after scratchpad XOR + F/E load) for program 0, and write
// lines in the same word order as emul_vm.cpp's emul_iter_states.txt:
//   ic r0..r7 f0.lo f0.hi f1.lo f1.hi f2.lo f2.hi f3.lo f3.hi e0.lo e0.hi ...
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
    const int MAXB = argc > 1 ? atoi(argv[1]) : 2048;
    FILE* f = fopen("stagecmp_nt.err", "rb");
    if (!f) { fprintf(stderr, "stagecmp_nt.err missing\n"); return 1; }
    FILE* out = fopen("cpu_iter_states.txt", "w");
    char line[512];
    int blocks = 0;
    while (blocks < MAXB && fgets(line, sizeof line, f)) {
        if (strncmp(line, "[CPU_TRACE] EXEC pc=0", 21) != 0) continue;
        uint64_t r[8], fh[4], fl[4], eh[4], el[4];
        bool ok = true;
        for (int i = 0; i < 8; ++i) {
            if (!fgets(line, sizeof line, f)) { ok = false; break; }
            unsigned long long v;
            if (sscanf(line, "  R[%*d]=%llx", &v) != 1) { ok = false; break; }
            r[i] = v;
        }
        for (int i = 0; ok && i < 4; ++i) {
            if (!fgets(line, sizeof line, f)) { ok = false; break; }
            unsigned long long a, b, c, d;
            if (sscanf(line, "  F[%*d]=%16llx%16llx E[%*d]=%16llx%16llx", &a, &b, &c, &d) != 4) { ok = false; break; }
            fh[i] = a; fl[i] = b; eh[i] = c; el[i] = d;
        }
        if (!ok) { fprintf(stderr, "parse error at block %d\n", blocks); break; }
        fprintf(out, "%d", blocks);
        for (int i = 0; i < 8; ++i) fprintf(out, " %016llx", (unsigned long long)r[i]);
        for (int i = 0; i < 4; ++i) fprintf(out, " %016llx %016llx", (unsigned long long)fl[i], (unsigned long long)fh[i]);
        for (int i = 0; i < 4; ++i) fprintf(out, " %016llx %016llx", (unsigned long long)el[i], (unsigned long long)eh[i]);
        fprintf(out, "\n");
        blocks++;
    }
    fclose(out); fclose(f);
    fprintf(stderr, "parsed %d iteration-start states\n", blocks);
    return 0;
}
