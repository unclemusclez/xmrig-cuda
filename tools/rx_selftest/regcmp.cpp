#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <randomx.h>
#include "virtual_machine.hpp"

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

static void embed_nonce(std::vector<uint8_t>& in, uint32_t nonce)
{
    in[32] = 0;
    in[39] = (uint8_t)(nonce & 0xFF);
    in[40] = (uint8_t)((nonce >> 8) & 0xFF);
    in[41] = (uint8_t)((nonce >> 16) & 0xFF);
    in[42] = (uint8_t)((nonce >> 24) & 0xFF);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <seed_hex> <base_hex> <nonce>\n", argv[0]);
        return 1;
    }
    auto seed = hex2bin(argv[1]);
    auto base = hex2bin(argv[2]);
    uint32_t nonce = (uint32_t)strtoul(argv[3], nullptr, 10);

    randomx_cache* cache = randomx_alloc_cache(RANDOMX_FLAG_DEFAULT);
    randomx_init_cache(cache, seed.data(), seed.size());
    randomx_dataset* dataset = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
    unsigned long items = randomx_dataset_item_count();
    unsigned long hw = std::thread::hardware_concurrency(); if (hw < 1) hw = 1;
    unsigned long per = (items + hw - 1) / hw;
    std::vector<std::thread> ts;
    for (unsigned long t = 0; t < hw; ++t) {
        unsigned long s = t * per, e = (s + per > items) ? items : s + per;
        if (s >= e) continue;
        ts.emplace_back(randomx_init_dataset, dataset, cache, s, e - s);
    }
    for (auto& th : ts) th.join();

    randomx_vm* vm = randomx_create_vm(RANDOMX_FLAG_FULL_MEM, nullptr, dataset);
    if (vm == nullptr) { fprintf(stderr, "vm create failed\n"); return 1; }

    std::vector<uint8_t> input = base;
    embed_nonce(input, nonce);
    std::vector<uint8_t> hash(RANDOMX_HASH_SIZE);

    // Run randomx_calculate_hash but dump register file after each program iteration
    // We need to hook into the VM execution. Since tevador doesn't expose per-iteration state,
    // we'll manually drive the VM by calling executeProgram and dumping registerFile after.
    // But randomx_calculate_hash does both programs internally.
    // Instead, let's just run calculate_hash once and dump final register file.
    // For per-iteration tracing, we'd need to modify tevador source - too invasive.
    // Alternative: use regcmp as-is for final state comparison.
    // The GPU trace is from a SINGLE program run (execute_vm_dbg on program 0).
    // But regcmp runs TWO programs. Can't directly compare.
    
    // For now, just run and print final hash and register file (matching regcmp)
    randomx_calculate_hash(vm, input.data(), input.size(), hash.data());

    auto* rf = vm->getRegisterFile();
    const uint64_t* w = reinterpret_cast<const uint64_t*>(rf);
    for (int i = 0; i < 32; ++i)
        printf("%016llx ", (unsigned long long)w[i]);
    printf("\n");

    printf("HASH ");
    for (int i = 0; i < RANDOMX_HASH_SIZE; ++i)
        printf("%02x", hash[i]);
    printf("\n");

    randomx_destroy_vm(vm);
    randomx_release_dataset(dataset);
    randomx_release_cache(cache);
    return 0;
}