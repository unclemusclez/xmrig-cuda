// CPU reference BLAKE2b-512 of the RandomX input (the "firstHash").
// Mirrors tevador randomx.cpp: blake2b(out, 64, input, inputLen, nullptr, 0).
// Usage: refhash.exe <seed_hex> <base_input_hex> <nonce>
//   Prints 8 uint64 hex words (64 bytes) of blake2b(input) to stdout.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "blake2/blake2.h"

static std::vector<uint8_t> hex2bin(const char* s)
{
    std::vector<uint8_t> out;
    size_t len = strlen(s);
    for (size_t i = 0; i + 1 < len; i += 2) {
        unsigned v = 0;
        if (sscanf(s + i, "%2x", &v) != 1) { fprintf(stderr, "bad hex at %zu\n", i); exit(1); }
        out.push_back((uint8_t)v);
    }
    return out;
}

// Same embedding as GPU blake2b_initial_hash / regcmp
static void embed_nonce(std::vector<uint8_t>& in, uint32_t nonce)
{
    in[39] = (uint8_t)(nonce & 0xFF);
    in[40] = (uint8_t)((nonce >> 8) & 0xFF);
    in[41] = (uint8_t)((nonce >> 16) & 0xFF);
    in[42] = (uint8_t)((nonce >> 24) & 0xFF);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <seed_hex> <base_input_hex> <nonce>\n", argv[0]);
        return 1;
    }
    auto base = hex2bin(argv[2]);
    uint32_t nonce = (uint32_t)strtoul(argv[3], nullptr, 10);

    std::vector<uint8_t> input = base;
    embed_nonce(input, nonce);

    uint8_t hash[64];
    int r = randomx_blake2b(hash, 64, input.data(), input.size(), nullptr, 0);
    if (r != 0) { fprintf(stderr, "blake2b failed %d\n", r); return 1; }

    const uint64_t* w = reinterpret_cast<const uint64_t*>(hash);
    for (int i = 0; i < 8; ++i)
        printf("%016llx ", (unsigned long long)w[i]);
    printf("\n");
    return 0;
}
