// Compare emul_iter_states.txt (start lines only, skip "END" lines) against
// cpu_iter_states.txt, word by word. Also cross-check sp0/sp1/fprc of END lines
// against cpu_ctl2.txt (offset by window base, default 4096).
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>

static const char* names[25] = {"ic","r0","r1","r2","r3","r4","r5","r6","r7","f0","f1","f2","f3","f4","f5","f6","f7","e0","e1","e2","e3","e4","e5","e6","e7"};

int main(int argc, char** argv)
{
    long long base = (argc > 1) ? atoll(argv[1]) : 4096; // global iter of emulator ic=0
    FILE* fe = fopen("emul_iter_states.txt", "r");
    FILE* fc = fopen("cpu_iter_states.txt", "r");
    FILE* ct = fopen("cpu_ctl2.txt", "r");
    if (!fe || !fc) { fprintf(stderr, "missing files\n"); return 1; }

    char le[16384];
    int ic = -1;
    int diffs = 0;
    while (fgets(le, sizeof le, fe)) {
        if (strstr(le, "END")) {
            // parse sp0 sp1 fprc and compare with cpu_ctl2 for giter=base+ic
            unsigned esp0 = 0, esp1 = 0, efprc = 0;
            char* s = strstr(le, "sp0="); if (s) esp0 = strtoul(s + 4, nullptr, 16);
            s = strstr(le, "sp1="); if (s) esp1 = strtoul(s + 4, nullptr, 16);
            s = strstr(le, "fprc="); if (s) efprc = strtoul(s + 5, nullptr, 10);
            if (ct) {
                char lc[1024];
                char target[64]; snprintf(target, sizeof target, "%lld ", base + (long long)ic);
                rewind(ct);
                while (fgets(lc, sizeof lc, ct)) {
                    if (strncmp(lc, target, strlen(target)) == 0) {
                        char* t = strstr(lc, "sp0="); unsigned csp0 = t ? strtoul(t + 4, nullptr, 16) : 0;
                        t = strstr(lc, "sp1="); unsigned csp1 = t ? strtoul(t + 4, nullptr, 16) : 0;
                        t = strstr(lc, "fprc="); unsigned cfprc = t ? strtoul(t + 5, nullptr, 10) : 0;
                        if (csp0 != esp0 || csp1 != esp1 || cfprc != efprc) {
                            printf("ic=%d END ctl: emul sp0=%08x sp1=%08x fprc=%u | cpu(iter %lld) sp0=%08x sp1=%08x fprc=%u\n",
                                   ic, esp0, esp1, efprc, base + ic, csp0, csp1, cfprc);
                        }
                        break;
                    }
                }
            }
            continue;
        }
        ++ic;
        char lc[16384];
        if (!fgets(lc, sizeof lc, fc)) { printf("cpu file ended at ic=%d\n", ic); break; }
        unsigned long long ew[25], cw[25];
        int ne = 0, nc = 0;
        char* tok = strtok(le, " \r\n");
        while (tok && ne < 25) { ew[ne++] = strtoull(tok, nullptr, 16); tok = strtok(nullptr, " \r\n"); }
        tok = strtok(lc, " \r\n");
        while (tok && nc < 25) { cw[nc++] = strtoull(tok, nullptr, 16); tok = strtok(nullptr, " \r\n"); }
        int printed = 0;
        for (int i = 0; i < ne && i < nc; ++i) {
            if (ew[i] != cw[i]) {
                if (!printed) { printf("FIRST/next divergent iter-start ic=%d:\n", ic); printed = 1; ++diffs; }
                if (diffs <= 4)
                    printf("  %-3s emul=%016llx cpu=%016llx\n", names[i], ew[i], cw[i]);
            }
        }
        if (diffs > 4) break;
    }
    if (diffs == 0) printf("ALL %d iteration-start states MATCH\n", ic + 1);
    fclose(fe); fclose(fc); if (ct) fclose(ct);
    return 0;
}
