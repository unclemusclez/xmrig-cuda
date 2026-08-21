// Extract CPU per-instruction states for program-2 iteration 281 from the full
// trace stagecmp_nt.err. Anchor on the iter-281 start state, stop after the
// iter-282 start state is emitted. Output: cpu_ic281_seq.txt lines
// "<pc> r0..r7 f0..f7 e0..e7" (F[2i]=flo, F[2i+1]=fhi convention).
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
    const char* path = (argc > 1) ? argv[1] : "stagecmp_nt.err";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    FILE* out = fopen("cpu_ic281_seq.txt", "w");

    // anchors from cpu_iters2.txt (giter 4377 / 4378)
    uint64_t A0[4] = { 0xc9983b6d9721ba53ULL, 0xb09f9851cc265df2ULL, 0xcc0c4e8b86cb74beULL, 0xe76451e776fbce40ULL };
    uint64_t B0[4] = { 0x97c79936d586bf1eULL, 0xe93b8718aa015074ULL, 0xbc3456bccbe71be9ULL, 0x2bbc2addcb61fe84ULL };

    char line[512];
    enum { IDLE, BLOCK } st = IDLE;
    int curPc = -1;
    uint64_t R[8], FW[8], EW[8];
    int nR = 0, nFE = 0;
    bool capturing = false;
    long long emitted = 0, seenBlocks = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "[CPU_TRACE] EXEC pc=", 20) == 0) {
            st = BLOCK;
            curPc = atoi(line + 20);
            nR = 0; nFE = 0;
            ++seenBlocks;
            continue;
        }
        if (st != BLOCK) continue;
        if (nR < 8 && strncmp(line, "  R[", 4) == 0) {
            R[nR++] = hexparse(line + 7, 16);
            continue;
        }
        if (nR == 8 && nFE < 4 && strncmp(line, "  F[", 4) == 0) {
            uint64_t fhi = hexparse(line + 7, 16);
            uint64_t flo = hexparse(line + 23, 16);
            const char* ep = strstr(line, "E[");
            const char* eeq = strchr(ep, '=');
            uint64_t ehi = hexparse(eeq + 1, 16);
            uint64_t elo = hexparse(eeq + 17, 16);
            FW[nFE * 2] = flo; FW[nFE * 2 + 1] = fhi;
            EW[nFE * 2] = elo; EW[nFE * 2 + 1] = ehi;
            ++nFE;
            if (nFE == 4) {
                st = IDLE;
                bool matchA = (R[0]==A0[0] && R[1]==A0[1] && R[2]==A0[2] && R[3]==A0[3]);
                bool matchB = (R[0]==B0[0] && R[1]==B0[1] && R[2]==B0[2] && R[3]==B0[3]);
                if (!capturing && matchA) capturing = true;
                if (capturing) {
                    fprintf(out, "%d", curPc);
                    for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)R[k]);
                    for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)FW[k]);
                    for (int k = 0; k < 8; ++k) fprintf(out, " %016llx", (unsigned long long)EW[k]);
                    fprintf(out, "\n");
                    ++emitted;
                    if (matchB) { capturing = false; fclose(out);
                        fprintf(stderr, "captured %lld blocks (stopped at iter-282 start); scan blocks seen=%lld\n", emitted, seenBlocks);
                        // continue scanning to see if anchor occurs again
                        out = fopen("cpu_ic281_seq_extra.txt", "a");
                    }
                }
            }
            continue;
        }
    }
    fclose(f);
    if (out) fclose(out);
    fprintf(stderr, "done: emitted %lld, blocks seen %lld\n", emitted, seenBlocks);
    return 0;
}
