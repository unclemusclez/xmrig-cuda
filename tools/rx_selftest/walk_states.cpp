// State-matching walker: no alignment assumption. For each emulator group
// boundary state, locate the identical full 24-word state in the CPU per-pc
// sequence. Report the mapping and the first emulator boundary whose state is
// NOT present in the CPU sequence (= first true divergence).
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>

struct State { unsigned long long w[24]; };

static bool eq(const State& a, const State& b) {
    for (int i = 0; i < 24; ++i) if (a.w[i] != b.w[i]) return false;
    return true;
}

int main()
{
    FILE* fc = fopen("cpu_ic281_seq.txt", "r");
    FILE* fe = fopen("emu_ic_boundary_states.txt", "r");
    if (!fc || !fe) { fprintf(stderr, "missing files\n"); return 1; }

    std::vector<State> seq; std::vector<int> pcs;
    char line[16384];
    while (fgets(line, sizeof line, fc)) {
        State s; int pc;
        char* tok = strtok(line, " \r\n"); pc = tok ? atoi(tok) : -1;
        tok = strtok(nullptr, " \r\n");
        int n = 0;
        while (tok && n < 24) { s.w[n++] = strtoull(tok, nullptr, 16); tok = strtok(nullptr, " \r\n"); }
        if (n == 24) { seq.push_back(s); pcs.push_back(pc); }
    }
    fprintf(stderr, "cpu states: %zu\n", seq.size());

    int prev_cpu = -1;
    int nb = 0;
    while (fgets(line, sizeof line, fe)) {
        char* sip = strstr(line, "ip=");
        char* sinst = strstr(line, "insts=");
        int ip = sip ? atoi(sip + 3) : -1;
        long long insts = sinst ? atoll(sinst + 6) : -1;
        State s; int n = 0;
        char* p = sinst; while (*p && *p != ' ') ++p;
        char* tok = strtok(p, " \r\n");
        while (tok && n < 24) { s.w[n++] = strtoull(tok, nullptr, 16); tok = strtok(nullptr, " \r\n"); }
        if (n < 24) continue;
        ++nb;
        int found = -1;
        // search in a window around prev_cpu forward
        int lo = (prev_cpu < 0) ? 0 : prev_cpu;
        for (size_t k = lo; k < seq.size() && k < (size_t)lo + 40; ++k) {
            if (eq(s, seq[k])) { found = (int)k; break; }
        }
        if (found < 0) {
            // fallback full search
            for (size_t k = 0; k < seq.size(); ++k) if (eq(s, seq[k])) { found = (int)k; break; }
        }
        printf("emul ip=%3d insts=%3lld -> cpu %s\n", ip, insts,
               found < 0 ? "NOT FOUND (divergence)" : ("pc=" + std::to_string(pcs[found]) + " idx=" + std::to_string(found)).c_str());
        if (found < 0) {
            // print the emul state and the nearest cpu state for diagnosis
            printf("  emul: "); for (int i=0;i<24;++i) printf("%016llx ", s.w[i]); printf("\n");
            if (prev_cpu >= 0 && prev_cpu + 1 < (int)seq.size()) {
                int k = prev_cpu + 1;
                printf("  cpu[pc=%d]: ", pcs[k]); for (int i=0;i<24;++i) printf("%016llx ", seq[k].w[i]); printf("\n");
                printf("  diff regs:");
                static const char* nm[24]={"r0","r1","r2","r3","r4","r5","r6","r7","f0","f1","f2","f3","f4","f5","f6","f7","e0","e1","e2","e3","e4","e5","e6","e7"};
                for (int i=0;i<24;++i) if (s.w[i]!=seq[k].w[i]) printf(" %s", nm[i]);
                printf("\n");
            }
            fclose(fc); fclose(fe);
            return 0;
        }
        prev_cpu = found;
    }
    printf("all %d emulator boundaries found in CPU sequence\n", nb);
    fclose(fc); fclose(fe);
    return 0;
}
