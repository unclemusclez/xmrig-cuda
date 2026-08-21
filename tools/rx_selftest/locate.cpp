// Find the CPU sequence index matching a given emulator state, then print the
// following CPU states + the emulator's next state, word-by-word diff.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

struct S24 {
    int pc;
    uint64_t w[24];
};

static int parse_cpu(const char* file, std::vector<S24>& seq) {
    FILE* f = fopen(file, "rb");
    if (!f) return -1;
    char line[512];
    int seen_pc0 = 0;
    S24 s; int rw = 0, fw = 0, ew = 0; bool collect = false; bool started = false;
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
    return 0;
}

int main(int argc, char** argv) {
    // emulator state lines to locate: which group index to match (default: previous good = 3)
    int match_group = argc > 1 ? atoi(argv[1]) : 3;
    int stop_group = argc > 2 ? atoi(argv[2]) : 4;

    std::vector<S24> seq;
    if (parse_cpu("stagecmp_nt.err", seq) < 0) { fprintf(stderr, "no trace\n"); return 1; }
    fprintf(stderr, "cpu states: %zu\n", seq.size());

    // read emulator group lines
    FILE* g = fopen("emu_g0_groups.txt", "r");
    if (!g) return 1;
    char gl[4096];
    std::vector<std::vector<uint64_t>> estates;
    std::vector<std::string> labels;
    while (fgets(gl, sizeof gl, g)) {
        std::vector<uint64_t> vals;
        char* sp2 = strchr(gl, ' ');
        labels.push_back(std::string(gl, sp2 ? (sp2 - gl) : (int)strlen(gl)));
        char* p = sp2;
        while (p && *p) {
            char* end = nullptr;
            unsigned long long v = strtoull(p, &end, 16);
            if (end == p) break;
            vals.push_back(v);
            p = end;
        }
        estates.push_back(vals);
    }
    fclose(g);

    if (match_group >= (int)estates.size()) { fprintf(stderr, "group out of range\n"); return 1; }
    auto& tgt = estates[match_group];
    int found = -1;
    for (size_t i = 0; i < seq.size(); ++i) {
        bool eq = true;
        for (int k = 0; k < 24 && eq; ++k) eq = (seq[i].w[k] == tgt[k]);
        if (eq) { found = (int)i; break; }
    }
    if (found < 0) { printf("group %d state not found in CPU sequence\n", match_group); return 0; }
    printf("emul group %d (ip label '%s') matches CPU seq[%d] pc=%d\n", match_group, labels[match_group].c_str(), found, seq[found].pc);

    int nshow = argc > 3 ? atoi(argv[3]) : 10;
    for (int i = found; i <= found + nshow && i < (int)seq.size(); ++i) {
        printf("cpu seq[%d] pc=%-4d R0=%016llx F2lo=%016llx F3hi=%016llx E1lo=%016llx E2lo=%016llx\n",
               i, seq[i].pc, (unsigned long long)seq[i].w[0], (unsigned long long)seq[i].w[10],
               (unsigned long long)seq[i].w[15], (unsigned long long)seq[i].w[18], (unsigned long long)seq[i].w[20]);
    }
    // full diff of emulator's stop_group state vs the cpu state at found+(distance)
    if (stop_group < (int)estates.size()) {
        auto& estop = estates[stop_group];
        // try to find estop in seq; report nearest candidate
        int best = -1;
        for (size_t i = found; i < seq.size(); ++i) {
            bool eq = true;
            for (int k = 0; k < 24 && eq; ++k) eq = (seq[i].w[k] == estop[k]);
            if (eq) { best = (int)i; break; }
        }
        if (best >= 0) printf("emul group %d state FOUND at CPU seq[%d] pc=%d\n", stop_group, best, seq[best].pc);
        else {
            printf("emul group %d state NOT in CPU seq after match; diff vs cpu seq[%d..]:\n", stop_group, found + (stop_group - match_group) * 2);
            int idx = found + (stop_group - match_group);
            if (idx < (int)seq.size()) {
                for (int k = 0; k < 24; ++k) if (seq[idx].w[k] != estop[k])
                    printf("  w[%02d] emul=%016llx cpu(seq[%d])=%016llx\n", k, estop[k], idx, (unsigned long long)seq[idx].w[k]);
            }
        }
    }
    return 0;
}
