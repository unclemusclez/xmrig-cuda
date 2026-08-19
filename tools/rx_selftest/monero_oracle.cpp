// Second CPU oracle: faithful standalone port of monero's rx-slow-hash.c
// rx_slow_hash() core path (.reference/monero/src/crypto/rx-slow-hash.c).
//
// It uses the SAME tevador/RandomX library as oracle.cpp, but follows monero's
// *production* wiring (randomx_get_flags(), light VM by default, no prebuilt
// dataset, MONERO_RANDOMX_UMASK flag masking) so it independently validates the
// host orchestration our hand-written oracle.cpp applies (cache/VM setup and the
// nonce-embedding convention). If oracle.cpp and this agree, the CPU-side wiring
// that feeds the GPU is confirmed correct; the GPU output is then compared against
// both to localise any remaining GPU-pipeline divergence.
//
// Usage: monero_oracle.exe <seed_hex> <base_input_hex> <nonce_count>
//   Prints one line per nonce:  "<nonce> <hash_hex_64chars>"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <string>
#include <vector>
#include <limits.h>

#include "randomx.h"

#define HASH_SIZE RANDOMX_HASH_SIZE

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

// Mirror blake2b_initial_hash nonce embedding in blake2b_cuda.hpp EXACTLY. m[4] is
// bytes 32..39 (LE); `m[4] &= (uint64_t(-1) >> 8)` clears byte 32 (bits 0..7), then
// `| (nonce << 56)` sets byte 39. m[5] is bytes 40..47; `m[5] &= (uint64_t(-1) << 24)`
// clears bytes 40..42, then `| (nonce >> 8)` loads bytes 40..42 with nonce>>8..24.
// Bytes 33..38 and 43+ keep their base values. This is the canonical RandomX/Monero
// nonce location and matches the (pristine) CUDA reference, so GPU and CPU are apples-to-apples.
static void embed_nonce(std::vector<uint8_t>& in, uint32_t nonce)
{
    in[32] = 0;                                   // byte 32 cleared by GPU mask
    in[39] = (uint8_t)(nonce & 0xFF);
    in[40] = (uint8_t)((nonce >> 8) & 0xFF);
    in[41] = (uint8_t)((nonce >> 16) & 0xFF);
    in[42] = (uint8_t)((nonce >> 24) & 0xFF);
}

// Faithful port of monero rx-slow-hash.c disabled_flags() (MONERO_RANDOMX_UMASK).
static randomx_flags monero_disabled_flags(void)
{
    static int flags = -1;
    if (flags != -1) return (randomx_flags)flags;
    const char* env = getenv("MONERO_RANDOMX_UMASK");
    if (!env) {
        flags = 0;
    } else {
        char* endptr = nullptr;
        long int value = strtol(env, &endptr, 0);
        if (endptr != env && value >= 0 && value < INT_MAX) flags = (int)value;
        else flags = 0;
    }
    return (randomx_flags)flags;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <seed_hex> <base_input_hex> <nonce_count>\n", argv[0]);
        return 1;
    }

    auto seed = hex2bin(argv[1]);
    auto base = hex2bin(argv[2]);
    uint32_t count = (uint32_t)strtoul(argv[3], nullptr, 10);

    if (seed.size() != HASH_SIZE) {
        fprintf(stderr, "seed must be %d bytes (got %zu)\n", (int)HASH_SIZE, seed.size());
        return 1;
    }
    if (base.size() < 43) {
        fprintf(stderr, "base input must be >= 43 bytes (got %zu)\n", base.size());
        return 1;
    }

    // monero rx_slow_hash: flags = enabled_flags() & ~disabled_flags()
    randomx_flags flags = (randomx_flags)((int)randomx_get_flags() & ~(int)monero_disabled_flags());
    fprintf(stderr, "[monero_oracle] flags=0x%x\n", (int)flags);

    randomx_cache* cache = randomx_alloc_cache(flags);
    if (!cache) { fprintf(stderr, "randomx_alloc_cache failed\n"); return 1; }
    randomx_init_cache(cache, seed.data(), seed.size());
    fprintf(stderr, "[monero_oracle] init_cache ok\n");

    // monero default (no MONERO_RANDOMX_FULL_MEM): light VM, dataset computed on the fly.
    randomx_vm* vm = randomx_create_vm(flags, cache, nullptr);
    if (!vm) { fprintf(stderr, "randomx_create_vm failed\n"); return 1; }
    fprintf(stderr, "[monero_oracle] create_vm ok (light mode)\n");

    std::vector<uint8_t> input = base;
    std::vector<uint8_t> hash(RANDOMX_HASH_SIZE);

    for (uint32_t n = 0; n < count; ++n) {
        input = base;
        embed_nonce(input, n);
        randomx_calculate_hash(vm, input.data(), input.size(), hash.data());

        printf("%u ", n);
        for (size_t i = 0; i < hash.size(); ++i) printf("%02x", hash[i]);
        printf("\n");
    }

    randomx_destroy_vm(vm);
    randomx_release_cache(cache);
    return 0;
}
