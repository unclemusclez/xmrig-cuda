// Scan the CPU trace for structural anomalies: each iteration must show
// EXEC pc=0,1,2,... in order. Report the first gaps/duplicates.
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main(int argc, char** argv)
{
    const char* path = (argc > 1) ? argv[1] : "stagecmp_nt.err";
    FILE* f = fopen(path, "rb");
    if (!f) { return 1; }
    char line[256];
    long long iter = -1;      // iteration count (pc=0 occurrences)
    int expect = 0;           // next expected pc
    long long anomalies = 0;
    long long lines = 0;
    while (fgets(line, sizeof(line), f)) {
        ++lines;
        if (strncmp(line, "[CPU_TRACE] EXEC pc=", 20) != 0) continue;
        int pc = atoi(line + 20);
        if (pc == 0) {
            if (expect != 0 && iter >= 0) {
                ++anomalies;
                if (anomalies <= 8)
                    printf("ANOMALY: pc=0 after pc=%d (iteration %lld truncated, expected pc=%d)\n", expect - 1, iter + 1, expect);
            }
            ++iter;
            expect = 1;
        } else {
            if (pc != expect) {
                ++anomalies;
                if (anomalies <= 8)
                    printf("ANOMALY: pc=%d expected %d at iteration %lld (line %lld)\n", pc, expect, iter, lines);
            }
            expect = pc + 1;
        }
        if (anomalies > 100000) { printf("too many anomalies, stopping\n"); break; }
    }
    fclose(f);
    printf("lines=%lld iterations=%lld anomalies=%lld\n", lines, iter + 1, anomalies);
    return 0;
}
