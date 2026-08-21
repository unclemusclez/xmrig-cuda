// Decode GPU compiled words [ip0..ip1] from a vmstate bin and print raw fields.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    int ip0 = argc > 1 ? atoi(argv[1]) : 0;
    int ip1 = argc > 2 ? atoi(argv[2]) : 16;
    const char* fname = argc > 3 ? argv[3] : "gpu_vmstate_p0.bin";
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "%s missing (run from tools/rx_selftest)\n", fname); return 1; }
    std::vector<uint8_t> v(2048);
    if (fread(v.data(), 1, 2048, f) != 2048) return 1;
    fclose(f);
    uint32_t* prog = (uint32_t*)(v.data() + 1024);
    uint32_t* imm = (uint32_t*)(v.data() + 256);
    for (int ip = ip0; ip <= ip1; ++ip) {
        uint32_t w = prog[ip];
        int dst = w & 7, src = (w >> 3) & 7, immo = (w >> 6) & 255, loc = (w >> 14) & 1;
        int shift = (w >> 15) & 3, si32 = (w >> 17) & 1, si64 = (w >> 18) & 1, neg = (w >> 19) & 1;
        int op = (w >> 20) & 15, nw = ((w >> 24) & 7), nfp = ((w >> 28) & 7);
        printf("ip=%3d %08x op=%2d dst=%d src=%d immo=%3d loc=%d sh=%d si32=%d si64=%d neg=%d workers=%d fp=%d | imm=%08x %08x\n",
               ip, w, op, dst, src, immo, loc, shift, si32, si64, neg, nw, nfp, imm[immo], imm[immo + 1]);
    }
    return 0;
}
