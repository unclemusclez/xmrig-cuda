// Find the CPU pc where R == target state, then print the next N blocks (pc + R).
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>

struct Blk { int pc; uint64_t r[8]; };

int main(int argc, char** argv) {
    // target state (emulator group#6 start)
    uint64_t tgt[8] = {
        0xc09f4cd39e1d440aULL, 0xfadddf6b00e4a77eULL, 0x0f916ac7ebb284f8ULL, 0x56b0582303514abdULL,
        0x716c098a7dadb0dcULL, 0x7840ee24561cb983ULL, 0xf6eef64a6ce1ebfbULL, 0xb8f665e8da19b9d0ULL };
    FILE* f = fopen("stagecmp_nt.err", "rb");
    if (!f) return 1;
    char line[512];
    std::vector<Blk> seq;
    Blk blk; int n = 0; int pc = -1; int seen_pc0 = 0; bool collect = false;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "[CPU_TRACE] EXEC pc=", 20) == 0) {
            if (collect && n == 8) seq.push_back(blk);
            pc = atoi(line + 20);
            if (pc == 0) { seen_pc0++; collect = (seen_pc0 == 1); }
            n = 0; blk.pc = pc; continue;
        }
        unsigned long long v;
        if (collect && sscanf(line, "  R[%*d]=%llx", &v) == 1) { if (n < 8) blk.r[n++] = v; }
    }
    if (collect && n == 8) seq.push_back(blk);
    fclose(f);
    fprintf(stderr, "iter0 blocks: %zu\n", seq.size());
    int found = -1;
    for (size_t i = 0; i < seq.size(); ++i)
        if (memcmp(seq[i].r, tgt, sizeof tgt) == 0) { found = (int)i; break; }
    if (found < 0) { printf("target state NOT found in CPU iter0 trace\n"); return 0; }
    printf("target state found at seq[%d] pc=%d\n", found, seq[found].pc);
    for (int i = found; i < found + 12 && i < (int)seq.size(); ++i) {
        printf("pc=%3d R: ", seq[i].pc);
        for (int k = 0; k < 8; ++k) printf("%016llx ", (unsigned long long)seq[i].r[k]);
        printf("\n");
    }
    return 0;
}
