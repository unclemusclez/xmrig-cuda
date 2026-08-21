// Compare emulator group-boundary states (emu_ic_boundary_states.txt) against
// CPU per-instruction states (cpu_ic281_seq.txt) using the insts= counter as
// alignment. Report first mismatching boundary and differing registers.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>

static const char* names[24] = {"r0","r1","r2","r3","r4","r5","r6","r7","f0","f1","f2","f3","f4","f5","f6","f7","e0","e1","e2","e3","e4","e5","e6","e7"};

int main()
{
    FILE* fc = fopen("cpu_ic281_seq.txt", "r");
    FILE* fe = fopen("emu_ic_boundary_states.txt", "r");
    if (!fc || !fe) { fprintf(stderr, "missing files\n"); return 1; }

    std::vector<std::vector<unsigned long long>> seq; // [k][0..23]
    std::vector<int> pcs;
    char line[16384];
    while (fgets(line, sizeof line, fc)) {
        std::vector<unsigned long long> w;
        char* tok = strtok(line, " \r\n");
        int pc = tok ? atoi(tok) : -1;
        pcs.push_back(pc);
        tok = strtok(nullptr, " \r\n");
        while (tok && w.size() < 24) { w.push_back(strtoull(tok, nullptr, 16)); tok = strtok(nullptr, " \r\n"); }
        seq.push_back(w);
    }
    fprintf(stderr, "cpu seq lines: %zu (last pc=%d)\n", seq.size(), pcs.back());

    int prev_good_ip = -1; long long prev_good_insts = -1;
    int boundaries = 0;
    while (fgets(line, sizeof line, fe)) {
        // parse ip= insts=
        char* sip = strstr(line, "ip=");
        char* sinst = strstr(line, "insts=");
        int ip = sip ? atoi(sip + 3) : -1;
        long long insts = sinst ? atoll(sinst + 6) : -1;
        unsigned long long ew[24]; int ne = 0;
        // words start after "insts=<num>"
        char* p = sinst;
        while (*p && *p != ' ') ++p;
        char* tok = strtok(p, " \r\n");
        while (tok && ne < 24) { ew[ne++] = strtoull(tok, nullptr, 16); tok = strtok(nullptr, " \r\n"); }
        ++boundaries;
        size_t k = (size_t)insts;
        if (k >= seq.size()) { printf("boundary ip=%d insts=%lld beyond cpu seq (%zu)\n", ip, insts, seq.size()); return 0; }
        const auto& cw = seq[k];
        bool match = true;
        char diffbuf[512]; diffbuf[0] = 0;
        for (int i = 0; i < 24 && i < ne && i < (int)cw.size(); ++i) {
            if (ew[i] != cw[i]) {
                match = false;
                char tmp[64];
                snprintf(tmp, sizeof tmp, " %s", names[i]);
                strcat(diffbuf, tmp);
            }
        }
        if (!match) {
            printf("FIRST MISMATCH: boundary ip=%d insts=%lld (cpu line idx=%zu pc=%d)\n", ip, insts, k, pcs[k]);
            printf("  differing regs:%s\n", diffbuf);
            for (int i = 0; i < 24 && i < ne; ++i) {
                if (ew[i] != cw[i])
                    printf("  %-3s emul=%016llx cpu=%016llx\n", names[i], ew[i], cw[i]);
            }
            printf("last matching boundary: ip=%d insts=%lld\n", prev_good_ip, prev_good_insts);
            fclose(fc); fclose(fe);
            return 0;
        }
        prev_good_ip = ip; prev_good_insts = insts;
    }
    printf("ALL %d boundaries MATCH (last ip=%d insts=%lld)\n", boundaries, prev_good_ip, prev_good_insts);
    fclose(fc); fclose(fe);
    return 0;
}
