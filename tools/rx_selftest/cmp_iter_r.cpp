// Compare ONLY the 8 R registers (words 1..8) between emul and cpu iter states.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
    int maxprint = argc > 1 ? atoi(argv[1]) : 5;
    FILE* fe = fopen("emul_iter_states.txt", "r");
    FILE* fc = fopen("cpu_iter_states.txt", "r");
    if (!fe || !fc) { fprintf(stderr, "missing\n"); return 1; }
    char le[8192], lc[8192];
    int ic = 0, ndiff = 0;
    while (fgets(le, sizeof le, fe) && fgets(lc, sizeof lc, fc)) {
        unsigned long long ew[9], cw[9];
        int ne = 0, nc = 0;
        char* t = strtok(le, " \n"); while (t && ne < 9) { ew[ne++] = strtoull(t, nullptr, 16); t = strtok(nullptr, " \n"); }
        t = strtok(lc, " \n"); while (t && nc < 9) { cw[nc++] = strtoull(t, nullptr, 16); t = strtok(nullptr, " \n"); }
        bool diff = false;
        for (int i = 1; i <= 8; ++i) if (ew[i] != cw[i]) { diff = true; break; }
        if (diff) {
            if (ndiff < maxprint) {
                printf("R DIVERGE ic=%d:\n", ic);
                for (int i = 1; i <= 8; ++i)
                    printf("  r%d emul=%016llx cpu=%016llx %s\n", i - 1, ew[i], cw[i], ew[i] == cw[i] ? "" : "<DIFF");
            }
            ndiff++;
        }
        ic++;
    }
    printf("total iterations compared=%d, R-divergent=%d\n", ic, ndiff);
    fclose(fe); fclose(fc);
    return 0;
}
