// Extract per-iteration start states from ITERSTART trace (iter_trace.txt).
// Emits cpu_iters2.txt: "<giter> R0..R7 F0..F7 E0..E7" (emulator convention:
// F[2i]=flo, F[2i+1]=fhi; same for E) and cpu_ctl2.txt: "<giter> fprc ma mx sp0 sp1".
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

static uint64_t hexparse(const char* s, size_t n)
{
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) {
        char c = s[i];
        int d = (c <= '9') ? (c - '0') : ((c <= 'F') ? (c - 'A' + 10) : ((c <= 'f') ? (c - 'a' + 10) : 0));
        v = (v << 4) | (uint64_t)d;
    }
    return v;
}

int main(int argc, char** argv)
{
    const char* path = (argc > 1) ? argv[1] : "iter_trace.txt";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    FILE* out = fopen("cpu_iters2.txt", "w");
    FILE* ctl = fopen("cpu_ctl2.txt", "w");

    char line[1024];
    uint64_t R[8], FW[8], EW[8];
    int nR = 0, nFE = 0;
    bool inRec = false;
    long long giter = -1;
    unsigned fprc = 99, ma = 0, mx = 0, sp0 = 0, sp1 = 0;
    bool haveCtl = false;

    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "ITERSTART ic=")) {
            if (haveCtl) {
                fprintf(out, "%lld", giter);
                for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)R[k]);
                for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)FW[k]);
                for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)EW[k]);
                fprintf(out, "\n");
                fprintf(ctl, "%lld fprc=%u ma=%08x mx=%08x sp0=%08x sp1=%08x\n", giter, fprc, ma, mx, sp0, sp1);
            }
            inRec = true; nR = 0; nFE = 0; haveCtl = false;
            ++giter;
            continue;
        }
        if (!inRec) continue;
        const char* p = strstr(line, "R[");
        if (p && nR < 8) {
            const char* eq = strchr(p, '=');
            R[nR++] = hexparse(eq + 1, 16);
            if (nR == 8 && nFE < 4) continue;
            continue;
        }
        const char* pf = strstr(line, "F[");
        const char* pe = strstr(line, "E[");
        if (pf && pe && nFE < 4) {
            const char* feq = strchr(pf, '=');
            const char* eeq = strchr(pe, '=');
            uint64_t fhi = hexparse(feq + 1, 16);
            uint64_t flo = hexparse(feq + 17, 16);
            uint64_t ehi = hexparse(eeq + 1, 16);
            uint64_t elo = hexparse(eeq + 17, 16);
            FW[nFE * 2] = flo; FW[nFE * 2 + 1] = fhi;
            EW[nFE * 2] = elo; EW[nFE * 2 + 1] = ehi;
            ++nFE;
            continue;
        }
        const char* pc = strstr(line, "CTL ");
        if (pc) {
            const char* s = strstr(line, "fprc=");
            fprc = s ? (unsigned)strtoul(s + 5, nullptr, 10) : 99;
            s = strstr(line, "ma="); ma = s ? (unsigned)strtoul(s + 3, nullptr, 16) : 0;
            s = strstr(line, "mx="); mx = s ? (unsigned)strtoul(s + 3, nullptr, 16) : 0;
            s = strstr(line, "sp0="); sp0 = s ? (unsigned)strtoul(s + 4, nullptr, 16) : 0;
            s = strstr(line, "sp1="); sp1 = s ? (unsigned)strtoul(s + 4, nullptr, 16) : 0;
            haveCtl = true;
            continue;
        }
        if (haveCtl) {
            fprintf(out, "%lld", giter);
            for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)R[k]);
            for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)FW[k]);
            for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)EW[k]);
            fprintf(out, "\n");
            fprintf(ctl, "%lld fprc=%u ma=%08x mx=%08x sp0=%08x sp1=%08x\n", giter, fprc, ma, mx, sp0, sp1);
            haveCtl = false;
        }
    }
    if (haveCtl) {
        fprintf(out, "%lld", giter);
        for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)R[k]);
        for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)FW[k]);
        for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)EW[k]);
        fprintf(out, "\n");
        fprintf(ctl, "%lld fprc=%u ma=%08x mx=%08x sp0=%08x sp1=%08x\n", giter, fprc, ma, mx, sp0, sp1);
    }
    fclose(f); fclose(out); fclose(ctl);
    fprintf(stderr, "total ITERSTART blocks: %lld\n", giter + 1);
    return 0;
}
