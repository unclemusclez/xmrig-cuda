// Decode ALL raw instructions of a RandomX program and ALL packed GPU words,
// reduce each to a semantic op name, and compare the multisets. A mismatch
// proves init_vm corrupted the instruction stream (added/dropped/re-typed).
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <map>

// frequency ceilings
static const char* rawName(int op) {
    if (op < 16) return "IADD_RS";
    if (op < 23) return "IADD_M";
    if (op < 39) return "ISUB_R";
    if (op < 46) return "ISUB_M";
    if (op < 62) return "IMUL_R";
    if (op < 66) return "IMUL_M";
    if (op < 70) return "IMULH_R";
    if (op < 71) return "IMULH_M";
    if (op < 75) return "ISMULH_R";
    if (op < 76) return "ISMULH_M";
    if (op < 84) return "IMUL_RCP";
    if (op < 86) return "INEG_R";
    if (op < 101) return "IXOR_R";
    if (op < 106) return "IXOR_M";
    if (op < 114) return "IROR_R";
    if (op < 116) return "IROL_R";
    if (op < 120) return "ISWAP_R";
    if (op < 124) return "FSWAP_R";
    if (op < 140) return "FADD_R";
    if (op < 145) return "FADD_M";
    if (op < 161) return "FSUB_R";
    if (op < 166) return "FSUB_M";
    if (op < 172) return "FSCAL_R";
    if (op < 204) return "FMUL_R";
    if (op < 206) return "FDIV_M";
    if (op < 214) return "FSQRT_R";
    if (op < 239) return "CBRANCH";
    if (op < 240) return "CFROUND";
    return "ISTORE";
}

// packed op -> coarse semantic name (mirrors emul_vm / randomx_cuda.hpp semantics)
static const char* packedName(int op, int loc, int neg, int shift) {
    switch (op) {
    case 0: return "IADD_RS";
    case 1: return neg ? "ISUB_R" : "IADD_R";
    case 2: return loc ? "IMUL_M" : "IMUL_R";
    case 3: return loc ? "IXOR_M" : "IXOR_R";
    case 4: return "IMULH";
    case 5: return "INEG_R";
    case 6: return "IMULH";
    case 7: return "IROR/IROL";
    case 8: return "ISWAP_R";
    case 9: return "CBRANCH";
    case 10: return "ISTORE";
    case 11: return "FSWAP_R";
    case 12: return shift ? "FMUL/FADD_E" : "FADD/FSUB_F";
    case 13: return "CFROUND";
    case 14: return "FSQRT_R";
    case 15: return "FDIV/FADD_M";
    default: return "?";
    }
}

int main(int argc, char** argv) {
    const char* prog = argc > 1 ? argv[1] : "cpu_prog_p2.bin";
    const char* vm = argc > 2 ? argv[2] : "gpu_vmstate_p2.bin";
    FILE* f = fopen(prog, "rb");
    std::vector<uint8_t> p(3200);
    if (!f || fread(p.data(), 1, 3200, f) != 3200) { fprintf(stderr, "prog read fail\n"); return 1; }
    fclose(f);
    f = fopen(vm, "rb");
    std::vector<uint8_t> v(2048);
    if (!f || fread(v.data(), 1, 2048, f) != 2048) { fprintf(stderr, "vm read fail\n"); return 1; }
    fclose(f);

    // raw histogram (broad buckets that are unambiguous)
    std::map<std::string, int> rawH;
    for (int i = 0; i < 256; ++i) {
        uint32_t x = *(uint32_t*)(p.data() + 128 + 8 * i);
        rawH[rawName(x & 0xff)]++;
    }

    // packed histogram: walk groups
    uint32_t* comp = (uint32_t*)(v.data() + 1024);
    std::map<std::string, int> pkH;
    int ip = 0, groups = 0, lanes = 0;
    while (ip < 256) {
        uint32_t h = comp[ip];
        int nw = (h >> 24) & 7, nf = (h >> 28) & 7;
        for (int sub = 0; sub <= nw; ++sub) {
            int io = sub - nf;
            bool isfp = io < nf;
            uint32_t w = comp[ip + (isfp ? (sub >> 1) : io)];
            int op = (w >> 20) & 15, loc = (w >> 14) & 1, neg = (w >> 19) & 1, sh = (w >> 15) & 3;
            pkH[packedName(op, loc, neg, sh)]++;
            ++lanes;
        }
        ip += (nw - nf) + 1;
        ++groups;
    }

    // full listings to files
    FILE* fr = fopen("raw_listing_p2.txt", "w");
    for (int i = 0; i < 256; ++i) {
        uint32_t x = *(uint32_t*)(p.data() + 128 + 8 * i);
        uint32_t y = *(uint32_t*)(p.data() + 132 + 8 * i);
        int op = x & 0xff, dst = (x >> 8) & 7, src = (x >> 16) & 7, mod = (x >> 24) & 255;
        fprintf(fr, "%3d %-9s op=%3d dst=%d src=%d mod=%3d imm=%08x\n", i, rawName(op), op, dst, src, mod, y);
    }
    fclose(fr);

    FILE* fpk = fopen("packed_listing_p2.txt", "w");
    ip = 0;
    while (ip < 256) {
        uint32_t h = comp[ip];
        int nw = (h >> 24) & 7, nf = (h >> 28) & 7;
        for (int sub = 0; sub <= nw; ++sub) {
            int io = sub - nf;
            bool isfp = io < nf;
            int wi = ip + (isfp ? (sub >> 1) : io);
            uint32_t w = comp[wi];
            fprintf(fpk, "grp=%3d sub=%d word=%3d %08x op=%2d dst=%d src=%d immo=%3d loc=%d sh=%d si32=%d si64=%d neg=%d %-14s\n",
                    ip, sub, wi, w, (w >> 20) & 15, w & 7, (w >> 3) & 7, (w >> 6) & 255,
                    (w >> 14) & 1, (w >> 15) & 3, (w >> 17) & 1, (w >> 18) & 1, (w >> 19) & 1,
                    packedName((w >> 20) & 15, (w >> 14) & 1, (w >> 19) & 1, (w >> 15) & 3));
        }
        ip += (nw - nf) + 1;
    }
    fclose(fpk);

    printf("groups=%d packed_lanes=%d\n\n", groups, lanes);
    printf("--- RAW (256) ---\n");
    for (auto& kv : rawH) printf("%-16s %6d\n", kv.first.c_str(), kv.second);
    printf("--- PACKED (lanes) ---\n");
    for (auto& kv : pkH) printf("%-16s %6d\n", kv.first.c_str(), kv.second);
    return 0;
}
