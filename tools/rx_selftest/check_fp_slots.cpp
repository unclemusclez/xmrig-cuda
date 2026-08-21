// Verify packed program p2 FP/int lane accounting: walk groups, sum nf, and
// list every FP-slot word whose opcode is NOT an FP opcode (11,12,14,15),
// plus every integer-slot word whose opcode IS FP.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>

int main(int argc, char** argv) {
    const char* vm = argc > 1 ? argv[1] : "gpu_vmstate_p2.bin";
    FILE* f = fopen(vm, "rb");
    std::vector<uint8_t> v(2048);
    if (!f || fread(v.data(), 1, 2048, f) != 2048) return 1;
    fclose(f);
    uint32_t* comp = (uint32_t*)(v.data() + 1024);
    int plen = (int)*(uint32_t*)(v.data() + 160);

    int ip = 0, sumNf = 0, sumNw1 = 0, groups = 0;
    int fpLanes = 0, intLanes = 0;
    while (ip < plen) {
        uint32_t h = comp[ip];
        int nw = (h >> 24) & 7, nf = (h >> 28) & 7;
        sumNf += nf; sumNw1 += nw + 1; ++groups;
        for (int sub = 0; sub <= nw; ++sub) {
            int io = sub - nf;
            bool isfp = io < nf;
            int wi = ip + (isfp ? (sub >> 1) : io);
            uint32_t w = comp[wi];
            int op = (w >> 20) & 15;
            bool fpOp = (op == 11 || op == 12 || op == 14 || op == 15);
            if (isfp) {
                ++fpLanes;
                if (!fpOp && sub % 2 == 0)
                    printf("FP-SLOT with INTEGER op: grp=%d sub=%d word=%d op=%d w=%08x (dst=%d src=%d neg=%d)\n",
                           ip, sub, wi, op, w, w & 7, (w >> 3) & 7, (w >> 19) & 1);
            } else {
                ++intLanes;
                if (fpOp)
                    printf("INT-SLOT with FP op: grp=%d sub=%d word=%d op=%d w=%08x\n", ip, sub, wi, op, w);
            }
        }
        ip += (nw - nf) + 1;
    }
    printf("plen=%d groups=%d sum(nw+1)=%d sum(nf)=%d fpLanes=%d intLanes=%d instructions=%d\n",
           plen, groups, sumNw1, sumNf, fpLanes, intLanes, sumNw1 - sumNf);
    return 0;
}
