// Compare emul_iter_states.txt vs cpu_iter_states.txt line by line.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>

int main() {
    FILE* fe = fopen("emul_iter_states.txt", "r");
    FILE* fc = fopen("cpu_iter_states.txt", "r");
    if (!fe || !fc) { fprintf(stderr, "missing iter state files\n"); return 1; }
    char le[8192], lc[8192];
    int ic = 0;
    while (fgets(le, sizeof le, fe) && fgets(lc, sizeof lc, fc)) {
        if (strcmp(le, lc) != 0) {
            printf("FIRST DIVERGENT ITERATION START: ic=%d\n", ic);
            // print word-by-word diff
            char* pe = le; char* pc2 = lc;
            // both start with "%d" then 24 words
            unsigned long long ew[25], cw[25];
            int nw_e = 0, nw_c = 0;
            char* tok;
            tok = strtok(le, " \n"); while (tok && nw_e < 25) { ew[nw_e++] = strtoull(tok, nullptr, 16); tok = strtok(nullptr, " \n"); }
            tok = strtok(lc, " \n"); while (tok && nw_c < 25) { cw[nw_c++] = strtoull(tok, nullptr, 16); tok = strtok(nullptr, " \n"); }
            const char* names[25] = {"ic","r0","r1","r2","r3","r4","r5","r6","r7","f0","f1","f2","f3","f4","f5","f6","f7","e0","e1","e2","e3","e4","e5","e6","e7"};
            for (int i = 0; i < nw_e && i < nw_c; ++i) {
                if (ew[i] != cw[i]) printf("  %-3s emul=%016llx cpu=%016llx\n", names[i], ew[i], cw[i]);
            }
            fclose(fe); fclose(fc);
            return 0;
        }
        ic++;
    }
    printf("all %d iteration-start states MATCH\n", ic);
    fclose(fe); fclose(fc);
    return 0;
}
