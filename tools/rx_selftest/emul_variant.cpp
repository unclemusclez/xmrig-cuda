// Compute fillAes4Rx4 for BOTH v103 and v104 from the same seed (gpu_firsthash.bin)
// and report byte 0, to determine which variant the GPU actually ran.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "aes_table.inc"

static uint32_t get_byte(uint32_t a, uint32_t start_bit) { return (a >> start_bit) & 0xFF; }

static void run_variant(bool is_v104, const uint32_t* T, const uint8_t* seed, std::vector<uint8_t>& out, uint32_t* interm = nullptr) {
    const size_t OUTPUT_SIZE = 2176;
    out.assign(OUTPUT_SIZE, 0);
    for (uint32_t sub = 0; sub < 4; ++sub) {
        const uint32_t* s = ((const uint32_t*)seed) + sub * 4;
        uint32_t x[4] = { s[0], s[1], s[2], s[3] };
        const uint32_t s1 = (sub & 1) ? 8 : 24;
        const uint32_t s3 = (sub & 1) ? 24 : 8;
        const uint32_t* const t0 = (sub & 1) ? T : (T + 1024);
        const uint32_t* const t1 = (sub & 1) ? (T + 256) : (T + 1792);
        const uint32_t* const t2 = (sub & 1) ? (T + 512) : (T + 1536);
        const uint32_t* const t3 = (sub & 1) ? (T + 768) : (T + 1280);
        uint32_t k[16];
        if (is_v104) {
            const bool b = (sub < 2);
            k[ 0] = b ? 0x6421aaddu : 0xb5826f73u;
            k[ 1] = b ? 0xd1833ddbu : 0xe3d6a7a6u;
            k[ 2] = b ? 0x2f546d2bu : 0x3d518b6du;
            k[ 3] = b ? 0x99e5d23fu : 0x229effb4u;
            k[ 4] = b ? 0xb20e3450u : 0xc7566bf3u;
            k[ 5] = b ? 0xb6913f55u : 0x9c10b3d9u;
            k[ 6] = b ? 0x06f79d53u : 0xe9024d4eu;
            k[ 7] = b ? 0xa5dfcde5u : 0xb272b7d2u;
            k[ 8] = b ? 0x5c3ed904u : 0xf273c9e7u;
            k[ 9] = b ? 0x515e7bafu : 0xf765a38bu;
            k[10] = b ? 0x0aa4679fu : 0x2ba9660au;
            k[11] = b ? 0x171c02bfu : 0xf63befa7u;
            k[12] = b ? 0x85623763u : 0x7a7cd609u;
            k[13] = b ? 0xe78f5d08u : 0x915839deu;
            k[14] = b ? 0xcd673785u : 0x0c06d1fdu;
            k[15] = b ? 0xd8ded291u : 0xc0b0762du;
        }
        for (uint32_t blk = 0; blk < OUTPUT_SIZE / 64; ++blk) {
            uint32_t y[4];
            y[0] = t0[get_byte(x[0], 0)] ^ t1[get_byte(x[1], s1)] ^ t2[get_byte(x[2], 16)] ^ t3[get_byte(x[3], s3)] ^ (is_v104 ? k[0] : 0xf890465du);
            y[1] = t0[get_byte(x[1], 0)] ^ t1[get_byte(x[2], s1)] ^ t2[get_byte(x[3], 16)] ^ t3[get_byte(x[0], s3)] ^ (is_v104 ? k[1] : 0x7ffbe4a6u);
            y[2] = t0[get_byte(x[2], 0)] ^ t1[get_byte(x[3], s1)] ^ t2[get_byte(x[0], 16)] ^ t3[get_byte(x[1], s3)] ^ (is_v104 ? k[2] : 0x141f82b7u);
            y[3] = t0[get_byte(x[3], 0)] ^ t1[get_byte(x[0], s1)] ^ t2[get_byte(x[1], 16)] ^ t3[get_byte(x[2], s3)] ^ (is_v104 ? k[3] : 0xcf359e95u);
            if (interm && sub == 0 && blk == 0) { interm[0]=x[0]; interm[1]=x[1]; interm[2]=x[2]; interm[3]=x[3]; interm[4]=y[0]; }
            x[0] = t0[get_byte(y[0], 0)] ^ t1[get_byte(y[1], s1)] ^ t2[get_byte(y[2], 16)] ^ t3[get_byte(y[3], s3)] ^ (is_v104 ? k[4] : 0x6a55c450u);
            x[1] = t0[get_byte(y[1], 0)] ^ t1[get_byte(y[2], s1)] ^ t2[get_byte(y[3], 16)] ^ t3[get_byte(y[0], s3)] ^ (is_v104 ? k[5] : 0xfee8278au);
            x[2] = t0[get_byte(y[2], 0)] ^ t1[get_byte(y[3], s1)] ^ t2[get_byte(y[0], 16)] ^ t3[get_byte(y[1], s3)] ^ (is_v104 ? k[6] : 0xbd5c5ac3u);
            x[3] = t0[get_byte(y[3], 0)] ^ t1[get_byte(y[0], s1)] ^ t2[get_byte(y[1], 16)] ^ t3[get_byte(y[2], s3)] ^ (is_v104 ? k[7] : 0x6741ffdcu);
            if (interm && sub == 0 && blk == 0) { interm[5]=x[0]; }
            y[0] = t0[get_byte(x[0], 0)] ^ t1[get_byte(x[1], s1)] ^ t2[get_byte(x[2], 16)] ^ t3[get_byte(x[3], s3)] ^ (is_v104 ? k[8] : 0x114c47a4u);
            y[1] = t0[get_byte(x[1], 0)] ^ t1[get_byte(x[2], s1)] ^ t2[get_byte(x[3], 16)] ^ t3[get_byte(x[0], s3)] ^ (is_v104 ? k[9] : 0xd524fde4u);
            y[2] = t0[get_byte(x[2], 0)] ^ t1[get_byte(x[3], s1)] ^ t2[get_byte(x[0], 16)] ^ t3[get_byte(x[1], s3)] ^ (is_v104 ? k[10] : 0xa7279ad2u);
            y[3] = t0[get_byte(x[3], 0)] ^ t1[get_byte(x[0], s1)] ^ t2[get_byte(x[1], 16)] ^ t3[get_byte(x[2], s3)] ^ (is_v104 ? k[11] : 0x3d324aacu);
            if (interm && sub == 0 && blk == 0) { interm[6]=y[0]; }
            x[0] = t0[get_byte(y[0], 0)] ^ t1[get_byte(y[1], s1)] ^ t2[get_byte(y[2], 16)] ^ t3[get_byte(y[3], s3)] ^ (is_v104 ? k[12] : 0x810c3a2au);
            x[1] = t0[get_byte(y[1], 0)] ^ t1[get_byte(y[2], s1)] ^ t2[get_byte(y[3], 16)] ^ t3[get_byte(y[0], s3)] ^ (is_v104 ? k[13] : 0x99a9aeffu);
            x[2] = t0[get_byte(y[2], 0)] ^ t1[get_byte(y[3], s1)] ^ t2[get_byte(y[0], 16)] ^ t3[get_byte(y[1], s3)] ^ (is_v104 ? k[14] : 0x42d3dbd9u);
            x[3] = t0[get_byte(y[3], 0)] ^ t1[get_byte(y[0], s1)] ^ t2[get_byte(y[1], 16)] ^ t3[get_byte(y[2], s3)] ^ (is_v104 ? k[15] : 0x76f6db08u);
            if (interm && sub == 0 && blk == 0) { interm[7]=x[0]; }
            uint32_t* p = ((uint32_t*)out.data()) + blk * 16 + sub * 4;
            p[0] = x[0]; p[1] = x[1]; p[2] = x[2]; p[3] = x[3];
        }
    }
}

int main() {
    FILE* fh = fopen("gpu_firsthash.bin", "rb");
    if (!fh) { fprintf(stderr, "gpu_firsthash.bin missing\n"); return 1; }
    uint8_t seed[64]; fread(seed, 1, 64, fh); fclose(fh);
    const uint32_t* T = AES_TABLE;
    std::vector<uint8_t> v103, v104;
    uint32_t interm[8] = {0};
    run_variant(false, T, seed, v103);
    run_variant(true,  T, seed, v104, interm);
    auto rd = [](const char* n){
        FILE* f=fopen(n,"rb"); std::vector<uint8_t> v;
        if(f){ fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET); v.resize(sz); fread(v.data(),1,sz,f); fclose(f);} return v;
    };
    auto gpu = rd("gpu_entropy_p0.bin");
    printf("v103[0..7] = %02x %02x %02x %02x %02x %02x %02x %02x\n", v103[0],v103[1],v103[2],v103[3],v103[4],v103[5],v103[6],v103[7]);
    printf("v104[0..7] = %02x %02x %02x %02x %02x %02x %02x %02x\n", v104[0],v104[1],v104[2],v104[3],v104[4],v104[5],v104[6],v104[7]);
    printf("gpu [0..7] = %02x %02x %02x %02x %02x %02x %02x %02x\n", gpu[0],gpu[1],gpu[2],gpu[3],gpu[4],gpu[5],gpu[6],gpu[7]);
    printf("match gpu==v103: %s\n", (gpu==v103)?"YES":"no");
    printf("match gpu==v104: %s\n", (gpu==v104)?"YES":"no");
    printf("emul v104 intermediates (x0..3 in, y1, x2, y3, x4 out):\n");
    for (int i = 0; i < 8; ++i) printf("  emul[%d]=%08x\n", i, interm[i]);
    return 0;
}
