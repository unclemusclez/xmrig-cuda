// CPU golden-reference oracle for xmrig-cuda RandomX GPU self-test.
//
// Builds a RandomX dataset from a fixed seed using the official tevador/RandomX
// library, then computes the authoritative hash for a range of nonces using the
// same nonce-embedding the GPU kernel applies (blake2b_initial_hash path:
// byte32 = nonce&0xFF, bytes40..42 = nonce>>8 low 24 bits).
//
// Usage: oracle.exe <seed_hex> <base_input_hex> <nonce_count>
//   Prints one line per nonce:  "<nonce> <hash_hex_64chars>"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <thread>

#include "randomx.h"

static std::vector<uint8_t> hex2bin(const char* s)
{
    std::vector<uint8_t> out;
    size_t len = strlen(s);
    for (size_t i = 0; i + 1 < len + 1 && i + 1 < len + 1; i += 2) {
        if (i + 1 >= len) break;
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

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <seed_hex> <base_input_hex> <nonce_count>\n", argv[0]);
        return 1;
    }

    auto seed = hex2bin(argv[1]);
    auto base = hex2bin(argv[2]);
    uint32_t count = (uint32_t)strtoul(argv[3], nullptr, 10);

    if (base.size() < 43) {
        fprintf(stderr, "base input must be >= 43 bytes (got %zu)\n", base.size());
        return 1;
    }

    randomx_cache* cache = randomx_alloc_cache(RANDOMX_FLAG_DEFAULT);
    if (!cache) { fprintf(stderr, "randomx_alloc_cache failed\n"); return 1; }
    fprintf(stderr, "[oracle] alloc_cache ok\n");
    randomx_init_cache(cache, seed.data(), seed.size());
    fprintf(stderr, "[oracle] init_cache ok\n");

    randomx_dataset* dataset = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
    if (!dataset) { fprintf(stderr, "randomx_alloc_dataset failed\n"); return 1; }
    unsigned long items = randomx_dataset_item_count();
    fprintf(stderr, "[oracle] dataset items=%lu\n", items);

    // Parallel dataset init across hardware threads.
    {
        unsigned long hw = std::thread::hardware_concurrency();
        if (hw < 1) hw = 1;
        unsigned long per = (items + hw - 1) / hw;
        std::vector<std::thread> ts;
        for (unsigned long t = 0; t < hw; ++t) {
            unsigned long start = t * per;
            unsigned long end = (start + per > items) ? items : start + per;
            if (start >= end) continue;
            ts.emplace_back(randomx_init_dataset, dataset, cache, start, end - start);
        }
        for (auto& th : ts) th.join();
    }
    fprintf(stderr, "[oracle] init_dataset ok\n");

    randomx_vm* vm = randomx_create_vm(RANDOMX_FLAG_FULL_MEM, nullptr, dataset);
    if (!vm) { fprintf(stderr, "randomx_create_vm failed\n"); return 1; }
    fprintf(stderr, "[oracle] create_vm ok\n");

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
    randomx_release_dataset(dataset);
    randomx_release_cache(cache);
    return 0;
}
