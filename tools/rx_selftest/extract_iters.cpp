// Extract per-iteration start states from the CPU trace (stagecmp_nt.err).
// Each "[CPU_TRACE] EXEC pc=0" marks an iteration start; the following lines carry
// R[0..7] (8 lines) and 4 lines "F[i]=<fhi 16><flo 16> E[i]=<ehi 16><elo 16>".
// Emulator convention: F[2i]=flo (low address), F[2i+1]=fhi; same for E.
// Output: "<iter> r0..r7 f0..f7 e0..e7" hex; program window via argv[2..3].
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

static uint64_t hexparse(const char* s, size_t n)
{
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) {
        char c = s[i];
        int d = (c <= '9') ? (c - '0') : ((c <= 'F') ? (c - 'A' + 10) : (c - 'a' + 10));
        v = (v << 4) | (uint64_t)d;
    }
    return v;
}

int main(int argc, char** argv)
{
    const char* path = (argc > 1) ? argv[1] : "stagecmp_nt.err";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    long long progStart = (argc > 2) ? atoll(argv[2]) : 4096; // program 2 default
    long long progEnd   = (argc > 3) ? atoll(argv[3]) : 6144;

    char line[512];
    uint64_t R[8], FW[8], EW[8];
    int nR = 0, nFE = 0;
    bool inRec = false;
    long long iter = -1;
    long long emitted = 0;
    long long rLines = 0, fLines = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "[CPU_TRACE] EXEC pc=0", 21) == 0) {
            ++iter;
            nR = 0; nFE = 0; inRec = true;
            continue;
        }
        if (!inRec) continue;
        if (nR < 8 && strncmp(line, "  R[", 4) == 0) {
            R[nR++] = hexparse(line + 7, 16);
            ++rLines;
            continue;
        }
        if (nR == 8 && nFE < 4 && strncmp(line, "  F[", 4) == 0) {
            uint64_t fhi = hexparse(line + 7, 16);
            uint64_t flo = hexparse(line + 23, 16);
            uint64_t ehi = hexparse(line + 45, 16);
            uint64_t elo = hexparse(line + 61, 16);
            FW[nFE * 2] = flo; FW[nFE * 2 + 1] = fhi;
            EW[nFE * 2] = elo; EW[nFE * 2 + 1] = ehi;
            ++nFE;
            ++fLines;
            if (nFE == 4) {
                if (iter >= progStart && iter < progEnd) {
                    printf("%lld", iter - progStart);
                    for (int k = 0; k < 8; ++k) printf(" %016llx", (unsigned long long)R[k]);
                    for (int k = 0; k < 8; ++k) printf(" %016llx", (unsigned long long)FW[k]);
                    for (int k = 0; k < 8; ++k) printf(" %016llx", (unsigned long long)EW[k]);
                    printf("\n");
                    ++emitted;
                }
                inRec = false;
            }
        }
    }
    fclose(f);
    fprintf(stderr, "total iterations seen: %lld, emitted: %lld (window %lld..%lld) rLines=%lld fLines=%lld\n", iter + 1, emitted, progStart, progEnd, rLines, fLines);
    return 0;
}
