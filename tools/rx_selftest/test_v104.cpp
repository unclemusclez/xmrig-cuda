#include <cstdint>
#include <cstdio>
#include "aes_table.inc"

static uint32_t get_byte(uint32_t a, uint32_t start_bit) { return (a >> start_bit) & 0xFF; }

int main() {
    uint32_t x[4] = { 0x0a9cd38b, 0x93647626, 0x5c213ed4, 0xfc0a3a20 };
    const uint32_t* T = AES_TABLE;
    uint32_t sub = 0;
    const uint32_t s1 = (sub & 1) ? 8 : 24;
    const uint32_t s3 = (sub & 1) ? 24 : 8;
    const uint32_t* const t0 = (sub & 1) ? T : (T + 1024);
    const uint32_t* const t1 = (sub & 1) ? (T + 256) : (T + 1792);
    const uint32_t* const t2 = (sub & 1) ? (T + 512) : (T + 1536);
    const uint32_t* const t3 = (sub & 1) ? (T + 768) : (T + 1280);

    uint32_t k[16] = {
        0x6421aaddu, 0xd1833ddbu, 0x2f546d2bu, 0x99e5d23fu,
        0xb20e3450u, 0xb6913f55u, 0x06f79d53u, 0xa5dfcde5u,
        0x5c3ed904u, 0x515e7bafu, 0x0aa4679fu, 0x171c02bfu,
        0x85623763u, 0xe78f5d08u, 0xcd673785u, 0xd8ded291u
    };

    printf("in:  %08x %08x %08x %08x\n", x[0], x[1], x[2], x[3]);

    uint32_t y[4];
    y[0] = t0[get_byte(x[0], 0)] ^ t1[get_byte(x[1], s1)] ^ t2[get_byte(x[2], 16)] ^ t3[get_byte(x[3], s3)] ^ k[0];
    y[1] = t0[get_byte(x[1], 0)] ^ t1[get_byte(x[2], s1)] ^ t2[get_byte(x[3], 16)] ^ t3[get_byte(x[0], s3)] ^ k[1];
    y[2] = t0[get_byte(x[2], 0)] ^ t1[get_byte(x[3], s1)] ^ t2[get_byte(x[0], 16)] ^ t3[get_byte(x[1], s3)] ^ k[2];
    y[3] = t0[get_byte(x[3], 0)] ^ t1[get_byte(x[0], s1)] ^ t2[get_byte(x[1], 16)] ^ t3[get_byte(x[2], s3)] ^ k[3];
    printf("r1y: %08x %08x %08x %08x\n", y[0], y[1], y[2], y[3]);

    x[0] = t0[get_byte(y[0], 0)] ^ t1[get_byte(y[1], s1)] ^ t2[get_byte(y[2], 16)] ^ t3[get_byte(y[3], s3)] ^ k[4];
    x[1] = t0[get_byte(y[1], 0)] ^ t1[get_byte(y[2], s1)] ^ t2[get_byte(y[3], 16)] ^ t3[get_byte(y[0], s3)] ^ k[5];
    x[2] = t0[get_byte(y[2], 0)] ^ t1[get_byte(y[3], s1)] ^ t2[get_byte(y[0], 16)] ^ t3[get_byte(y[1], s3)] ^ k[6];
    x[3] = t0[get_byte(y[3], 0)] ^ t1[get_byte(y[0], s1)] ^ t2[get_byte(y[1], 16)] ^ t3[get_byte(y[2], s3)] ^ k[7];
    printf("r2x: %08x %08x %08x %08x\n", x[0], x[1], x[2], x[3]);

    y[0] = t0[get_byte(x[0], 0)] ^ t1[get_byte(x[1], s1)] ^ t2[get_byte(x[2], 16)] ^ t3[get_byte(x[3], s3)] ^ k[8];
    y[1] = t0[get_byte(x[1], 0)] ^ t1[get_byte(x[2], s1)] ^ t2[get_byte(x[3], 16)] ^ t3[get_byte(x[0], s3)] ^ k[9];
    y[2] = t0[get_byte(x[2], 0)] ^ t1[get_byte(x[3], s1)] ^ t2[get_byte(x[0], 16)] ^ t3[get_byte(x[1], s3)] ^ k[10];
    y[3] = t0[get_byte(x[3], 0)] ^ t1[get_byte(x[0], s1)] ^ t2[get_byte(x[1], 16)] ^ t3[get_byte(x[2], s3)] ^ k[11];
    printf("r3y: %08x %08x %08x %08x\n", y[0], y[1], y[2], y[3]);

    x[0] = t0[get_byte(y[0], 0)] ^ t1[get_byte(y[1], s1)] ^ t2[get_byte(y[2], 16)] ^ t3[get_byte(y[3], s3)] ^ k[12];
    x[1] = t0[get_byte(y[1], 0)] ^ t1[get_byte(y[2], s1)] ^ t2[get_byte(y[3], 16)] ^ t3[get_byte(y[0], s3)] ^ k[13];
    x[2] = t0[get_byte(y[2], 0)] ^ t1[get_byte(y[3], s1)] ^ t2[get_byte(y[0], 16)] ^ t3[get_byte(y[1], s3)] ^ k[14];
    x[3] = t0[get_byte(y[3], 0)] ^ t1[get_byte(y[0], s1)] ^ t2[get_byte(y[1], 16)] ^ t3[get_byte(y[2], s3)] ^ k[15];
    printf("r4x: %08x %08x %08x %08x\n", x[0], x[1], x[2], x[3]);

    return 0;
}