// Host blake2b check for the selftest base input, replicating both embedding
// conventions, to determine which input the GPU blake2b_initial_hash hashed.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "blake2/blake2.h"
#include "aes_hash.hpp"

int main()
{
    uint8_t base[76];
    for (int i = 0; i < 76; ++i) base[i] = (uint8_t)i;

    uint8_t inG[76]; // GPU-style: keep byte32; nonce(=0) at bytes 39..42
    memcpy(inG, base, 76);
    inG[39] = inG[40] = inG[41] = inG[42] = 0;

    uint8_t inC[76]; // obsolete oracle-style: also clear byte32
    memcpy(inC, inG, 76);
    inC[32] = 0;

    uint8_t hG[64], hC[64];
    blake2b(hG, 64, inG, 76, nullptr, 0);
    blake2b(hC, 64, inC, 76, nullptr, 0);

    printf("keep_byte32 "); for (int i = 0; i < 64; ++i) printf("%02x", hG[i]); printf("\n");
    printf("clr_byte32  "); for (int i = 0; i < 64; ++i) printf("%02x", hC[i]); printf("\n");

    // CPU fillAes1Rx4 over the keep_byte32 first hash -> should equal the GPU's
    // gpu_chain_h0.bin (the d_rx_hashes value entering program 0's fillAes4Rx4).
    {
        uint8_t evolved[64];
        memcpy(evolved, hG, 64);
        std::vector<uint8_t> scratch(2097152);
        fillAes1Rx4<false>(evolved, scratch.size(), scratch.data());
        FILE* fg = fopen("gpu_chain_h0.bin", "rb");
        if (fg) {
            uint8_t g[64];
            bool ok = (fread(g, 1, 64, fg) == 64);
            fclose(fg);
            if (ok) {
                printf("fillAes1Rx4 evolved-state gpu-vs-cpu: %s\n",
                       memcmp(g, evolved, 64) == 0 ? "MATCH (chain start aligned)" : "DIFFER");
            }
        }
    }

    // Upstream CPU fillAes4Rx4 over the GPU's evolved first-hash, compared to the
    // GPU's v104 entropy dump: tells us if v104 differs from upstream AES fill.
    {
        FILE* fs = fopen("gpu_chain_h0.bin", "rb");
        FILE* fe = fopen("gpu_entropy_p0.bin", "rb");
        if (fs && fe) {
            uint8_t st[64];
            std::vector<uint8_t> ent(2176), up(2176);
            bool ok = (fread(st, 1, 64, fs) == 64) && (fread(ent.data(), 1, 2176, fe) == 2176);
            fclose(fs); fclose(fe);
            if (ok) {
                fillAes4Rx4<false>(st, 2176, up.data());
                size_t nd = 0;
                for (size_t i = 0; i < 2176; ++i) if (up[i] != ent[i]) ++nd;
                printf("fillAes4Rx4 upstream-vs-v104: %zu differing bytes of 2176%s\n", nd, nd == 0 ? " (IDENTICAL)" : " (DIFFER)");
            }
        }
    }

    FILE* f = fopen("gpu_firsthash.bin", "rb");
    if (f) {
        uint8_t g[64];
        if (fread(g, 1, 64, f) == 64) {
            printf("gpu_file    "); for (int i = 0; i < 64; ++i) printf("%02x", g[i]); printf("\n");
            printf("match=%s\n", memcmp(g, hG, 64) == 0 ? "KEEP_BYTE32" : (memcmp(g, hC, 64) == 0 ? "CLR_BYTE32" : "NEITHER"));
        }
        fclose(f);
    }
    return 0;
}
