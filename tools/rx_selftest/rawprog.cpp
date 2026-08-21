// Dump raw CPU program instructions (opcode dst src mod imm32) for pc range.
#include <cstdio>
#include <cstdint>
#include <vector>

int main(int argc, char** argv) {
    const char* fname = argc > 3 ? argv[3] : "cpu_prog_p0.bin";
    int pc0 = argc > 1 ? atoi(argv[1]) : 0;
    int pc1 = argc > 2 ? atoi(argv[2]) : 20;
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "%s missing\n", fname); return 1; }
    std::vector<uint8_t> v(3200);
    if (fread(v.data(), 1, 3200, f) != 3200) return 1;
    fclose(f);
    // raw program at bytes 128.., 8 bytes per instruction
    for (int pc = pc0; pc <= pc1; ++pc) {
        uint8_t* p = v.data() + 128 + pc * 8;
        uint32_t opcode = p[0], dst = p[1], src = p[2], mod = p[3];
        uint32_t imm32 = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);
        printf("pc=%3d op=%3u dst=%u src=%u mod=%02x imm32=%08x\n", pc, opcode, dst, src, mod, imm32);
    }
    return 0;
}
