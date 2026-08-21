// Compare emulator write-deltas at ic=g_trace_ic against CPU per-instruction
// deltas derived from cpu_ic281_seq.txt (multiset by register+value), and
// compare ISTORE scratchpad writes (addr,value) against CPU expectations.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

static const char* wnames[24] = {"r0","r1","r2","r3","r4","r5","r6","r7","f0","f1","f2","f3","f4","f5","f6","f7","e0","e1","e2","e3","e4","e5","e6","e7"};

struct SRec { uint64_t addr, val; int ip; };

int main(int argc, char** argv)
{
    const char* emuf = argc > 1 ? argv[1] : "emu_ic_boundary_states.txt";
    FILE* fe = fopen(emuf, "r");
    FILE* fc = fopen("cpu_ic281_seq.txt", "r");
    FILE* fr = fopen("raw_listing_p2.txt", "r");
    if (!fe || !fc || !fr) { fprintf(stderr, "missing input files\n"); return 1; }

    // --- emulator deltas ---
    std::map<std::pair<int, uint64_t>, std::vector<int>> emuW; // (word,newval) -> ips
    std::multimap<std::pair<uint64_t, uint64_t>, int> emuS;     // (addr,val) -> ip
    char line[4096];
    while (fgets(line, sizeof line, fe)) {
        if (strncmp(line, "R off=", 6) == 0) {
            long off; unsigned long long oldv, newv; int ip;
            if (sscanf(line, "R off=%ld old=%llx new=%llx ip=%d", &off, &oldv, &newv, &ip) == 4) {
                int w = (off < 64) ? (int)(off / 8) : ((off < 128) ? 8 + (int)((off - 64) / 8) : 16 + (int)((off - 128) / 8));
                emuW[{w, newv}].push_back(ip);
            }
        } else if (strncmp(line, "S addr=", 7) == 0) {
            unsigned long long a, v; int ip;
            if (sscanf(line, "S addr=%llx val=%llx ip=%d", &a, &v, &ip) == 3)
                emuS.insert({{a, v}, ip});
        }
    }
    fclose(fe);

    // --- CPU states ---
    std::vector<std::vector<unsigned long long>> seq;
    while (fgets(line, sizeof line, fc)) {
        std::vector<unsigned long long> w(24, 0);
        char* tok = strtok(line, " \r\n"); // pc
        tok = strtok(nullptr, " \r\n");
        int n = 0;
        while (tok && n < 24) { w[n++] = strtoull(tok, nullptr, 16); tok = strtok(nullptr, " \r\n"); }
        seq.push_back(w);
    }
    fclose(fc);
    fprintf(stderr, "emu reg-writes=%zu  scr-writes=%zu ; cpu states=%zu\n", emuW.size(), emuS.size(), seq.size());

    // --- CPU deltas (multiset) ---
    std::map<std::pair<int, uint64_t>, std::vector<int>> cpuW; // (word,newval) -> pcs
    for (size_t k = 0; k + 1 < seq.size(); ++k) {
        for (int i = 0; i < 24; ++i) {
            if (seq[k][i] != seq[k + 1][i])
                cpuW[{i, seq[k + 1][i]}].push_back((int)k);
        }
    }

    // --- multiset diff, ordered ---
    int extraCpu = 0, extraEmu = 0;
    // collect ordered mismatch records
    struct CM { int pc; int w; uint64_t nv; };
    std::vector<CM> missCpu, missEmu;
    for (auto& kv : cpuW) {
        auto it = emuW.find(kv.first);
        int cCpu = (int)kv.second.size();
        int cEmu = (it == emuW.end()) ? 0 : (int)it->second.size();
        for (int i = cEmu; i < cCpu; ++i) missCpu.push_back({kv.second[i], kv.first.first, kv.first.second});
    }
    for (auto& kv : emuW) {
        auto it = cpuW.find(kv.first);
        int cEmu = (int)kv.second.size();
        int cCpu = (it == cpuW.end()) ? 0 : (int)it->second.size();
        for (int i = cCpu; i < cEmu; ++i) missEmu.push_back({kv.second[i], kv.first.first, kv.first.second});
    }
    std::sort(missCpu.begin(), missCpu.end(), [](const CM& a, const CM& b) { return a.pc < b.pc; });
    std::sort(missEmu.begin(), missEmu.end(), [](const CM& a, const CM& b) { return a.pc < b.pc; });
    printf("=== CPU register-writes missing in EMU (ordered by pc) ===\n");
    for (auto& m : missCpu)
        printf("  pc=%3d %s new=%016llx\n", m.pc, wnames[m.w], (unsigned long long)m.nv);
    printf("=== EMU register-writes missing in CPU (ordered by ip) ===\n");
    for (auto& m : missEmu)
        printf("  ip=%3d %s new=%016llx\n", m.pc, wnames[m.w], (unsigned long long)m.nv);
    printf("summary: extraCpu=%d extraEmu=%d\n", (int)missCpu.size(), (int)missEmu.size());

    // --- ISTORE check ---
    // raw lines: "<idx> ISTORE op=2NN dst=d src=s mod=m imm=XXXXXXXX"
    printf("=== ISTORE addr/val mismatches (cpu-expectation vs emu) ===\n");
    int shown = 0;
    while (fgets(line, sizeof line, fr)) {
        int idx, dst, src, mod, op; unsigned imm;
        if (sscanf(line, "%d ISTORE op=%d dst=%d src=%d mod=%d imm=%x", &idx, &op, &dst, &src, &mod, &imm) != 6)
            continue;
        uint64_t mask;
        int modCond = mod >> 4;
        if (modCond >= 14) mask = 0x1FFFF8ULL;
        else mask = ((mod & 3) ? 0x3FF8ULL : 0x3FFF8ULL);
        uint64_t addr = ((uint64_t)(seq[idx][dst] + (uint64_t)(int64_t)(int32_t)imm)) & mask;
        uint64_t val = seq[idx][src];
        auto it = emuS.find({addr, val});
        if (it == emuS.end()) {
            ++shown;
            if (shown <= 15) {
                printf("  cpu ISTORE pc=%d dst=r%d src=r%d mod=%d imm=%08x -> addr=%08llx val=%016llx  NOT in emu\n",
                       idx, dst, src, mod, imm, (unsigned long long)addr, (unsigned long long)val);
                // show what emu wrote near that addr
                auto lo = emuS.lower_bound({addr & ~0xFFULL, 0});
                int c = 0;
                for (auto jt = lo; jt != emuS.end() && c < 4 && jt->first.first <= (addr | 0xFFULL); ++jt, ++c)
                    printf("      emu nearby: addr=%016llx val=%016llx ip=%d\n", (unsigned long long)jt->first.first, (unsigned long long)jt->first.second, jt->second);
            }
        } else {
            emuS.erase(it); // consume one instance
        }
    }
    printf("ISTORE mismatches total: %d; unmatched emu scratch writes left: %zu\n", shown, emuS.size());
    fclose(fr);
    return 0;
}
