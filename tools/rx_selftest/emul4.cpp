// CPU emulation of the GPU's fillAes4Rx4_v104 round, to isolate whether the
// GPU's fillAes4Rx4 matches tevador's fillAes4Rx4 for an identical seed.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "aes_table.inc"

static uint32_t get_byte(uint32_t a, uint32_t start_bit) {
    return (a >> start_bit) & 0xFF;
}

int main(int argc, char** argv) {
    // seed = first hash (use gpu_firsthash.bin, identical to tevador's tempHash)
    FILE* fh = fopen("gpu_firsthash.bin", "rb");
    if (!fh) { fprintf(stderr, "gpu_firsthash.bin missing\n"); return 1; }
    uint8_t seed[64];
    fread(seed, 1, 64, fh); fclose(fh);

    const size_t OUTPUT_SIZE = 2176;
    std::vector<uint8_t> out(OUTPUT_SIZE, 0);

    const uint32_t* T = AES_TABLE;

    for (uint32_t sub = 0; sub < 4; ++sub) {
        const uint32_t* s = ((uint32_t*)seed) + sub * 4;
        uint32_t x[4] = { s[0], s[1], s[2], s[3] };

        const uint32_t s1 = (sub & 1) ? 8 : 24;
        const uint32_t s3 = (sub & 1) ? 24 : 8;

        const uint32_t* const t0 = (sub & 1) ? (T) : (T + 1024);
        const uint32_t* const t1 = (sub & 1) ? (T + 256) : (T + 1792);
        const uint32_t* const t2 = (sub & 1) ? (T + 512) : (T + 1536);
        const uint32_t* const t3 = (sub & 1) ? (T + 768) : (T + 1280);

        const bool b = (sub < 2);
        uint32_t k[16];
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

        for (uint32_t blk = 0; blk < OUTPUT_SIZE / 64; ++blk) {
            uint32_t y[4];
            y[0] = t0[get_byte(x[0], 0)] ^ t1[get_byte(x[1], s1)] ^ t2[get_byte(x[2], 16)] ^ t3[get_byte(x[3], s3)] ^ k[ 0];
            y[1] = t0[get_byte(x[1], 0)] ^ t1[get_byte(x[2], s1)] ^ t2[get_byte(x[3], 16)] ^ t3[get_byte(x[0], s3)] ^ k[ 1];
            y[2] = t0[get_byte(x[2], 0)] ^ t1[get_byte(x[3], s1)] ^ t2[get_byte(x[0], 16)] ^ t3[get_byte(x[1], s3)] ^ k[ 2];
            y[3] = t0[get_byte(x[3], 0)] ^ t1[get_byte(x[0], s1)] ^ t2[get_byte(x[1], 16)] ^ t3[get_byte(x[2], s3)] ^ k[ 3];

            x[0] = t0[get_byte(y[0], 0)] ^ t1[get_byte(y[1], s1)] ^ t2[get_byte(y[2], 16)] ^ t3[get_byte(y[3], s3)] ^ k[ 4];
            x[1] = t0[get_byte(y[1], 0)] ^ t1[get_byte(y[2], s1)] ^ t2[get_byte(y[3], 16)] ^ t3[get_byte(y[0], s3)] ^ k[ 5];
            x[2] = t0[get_byte(y[2], 0)] ^ t1[get_byte(y[3], s1)] ^ t2[get_byte(y[0], 16)] ^ t3[get_byte(y[1], s3)] ^ k[ 6];
            x[3] = t0[get_byte(y[3], 0)] ^ t1[get_byte(y[0], s1)] ^ t2[get_byte(y[1], 16)] ^ t3[get_byte(y[2], s3)] ^ k[ 7];

            y[0] = t0[get_byte(x[0], 0)] ^ t1[get_byte(x[1], s1)] ^ t2[get_byte(x[2], 16)] ^ t3[get_byte(x[3], s3)] ^ k[ 8];
            y[1] = t0[get_byte(x[1], 0)] ^ t1[get_byte(x[2], s1)] ^ t2[get_byte(x[3], 16)] ^ t3[get_byte(x[0], s3)] ^ k[ 9];
            y[2] = t0[get_byte(x[2], 0)] ^ t1[get_byte(x[3], s1)] ^ t2[get_byte(x[0], 16)] ^ t3[get_byte(x[1], s3)] ^ k[10];
            y[3] = t0[get_byte(x[3], 0)] ^ t1[get_byte(x[0], s1)] ^ t2[get_byte(x[1], 16)] ^ t3[get_byte(x[2], s3)] ^ k[11];

            x[0] = t0[get_byte(y[0], 0)] ^ t1[get_byte(y[1], s1)] ^ t2[get_byte(y[2], 16)] ^ t3[get_byte(y[3], s3)] ^ k[12];
            x[1] = t0[get_byte(y[1], 0)] ^ t1[get_byte(y[2], s1)] ^ t2[get_byte(y[3], 16)] ^ t3[get_byte(y[0], s3)] ^ k[13];
            x[2] = t0[get_byte(y[2], 0)] ^ t1[get_byte(y[3], s1)] ^ t2[get_byte(y[0], 16)] ^ t3[get_byte(y[1], s3)] ^ k[14];
            x[3] = t0[get_byte(y[3], 0)] ^ t1[get_byte(y[0], s1)] ^ t2[get_byte(y[1], 16)] ^ t3[get_byte(y[2], s3)] ^ k[15];

            uint32_t* p = ((uint32_t*)out.data()) + blk * 16 + sub * 4;
            p[0] = x[0]; p[1] = x[1]; p[2] = x[2]; p[3] = x[3];
        }
    }

    FILE* fo = fopen("emul_entropy.bin", "wb");
    fwrite(out.data(), 1, OUTPUT_SIZE, fo); fclose(fo);

    auto rd = [](const char* n) {
        FILE* f = fopen(n, "rb");
        std::vector<uint8_t> v;
        if (f) { fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f,0,SEEK_SET);
                v.resize(sz); fread(v.data(),1,sz,f); fclose(f); }
        return v;
    };
    auto A = rd("gpu_entropy.bin");
    auto B = rd("cpu_entropy.bin");
    auto E = out;
    auto cmp = [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const char* name){
        size_t n = a.size()<b.size()?a.size():b.size();
        size_t i=0; while(i<n && a[i]==b[i]) i++;
        printf("emul vs %-14s : %s", name, (a==b)?"MATCH\n":"DIFFER");
        if (a!=b) printf("  (first diff @%zu: emul=%02x %s=%02x)\n", i, E[i], name, (i<b.size()?b[i]:0));
    };
    cmp(E, A, "gpu_entropy");
    cmp(E, B, "cpu_entropy");
    return 0;
}
