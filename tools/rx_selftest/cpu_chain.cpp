// CPU reference chain dumper: replicates randomx_calculate_hash step by step
// and dumps every intermediate (tempHash before each program, RegisterFile
// after each program) so the GPU chain can be compared stage by stage.
//
// Usage: cpu_chain.exe <seed_hex> <base_hex>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>

#include "randomx.h"
#include "virtual_machine.hpp"
#include "intrin_portable.h"
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

static void embed_nonce(std::vector<uint8_t>& in, uint32_t nonce)
{
    // GPU blake2b_initial_hash keeps bytes 32..38 (mask (uint64_t(-1)>>8) clears only
    // byte 39) and writes nonce into bytes 39..42. Do NOT clear byte 32.
    in[39] = (uint8_t)(nonce & 0xFF);
    in[40] = (uint8_t)((nonce >> 8) & 0xFF);
    in[41] = (uint8_t)((nonce >> 16) & 0xFF);
    in[42] = (uint8_t)((nonce >> 24) & 0xFF);
}

static void dump(const char* name, const void* p, size_t n)
{
    FILE* f = fopen(name, "wb");
    if (f) { fwrite(p, 1, n, f); fclose(f); }
    fprintf(stderr, "[cpu_chain] dumped %s (%zu bytes)\n", name, n);
}

static void phex(const char* tag, const uint8_t* p, size_t n)
{
    fprintf(stderr, "%s ", tag);
    for (size_t i = 0; i < n; ++i) fprintf(stderr, "%02x", p[i]);
    fprintf(stderr, "\n");
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <seed_hex> <base_hex>\n", argv[0]); return 1; }
    auto seed = hex2bin(argv[1]);
    auto base = hex2bin(argv[2]);

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
    fprintf(stderr, "[cpu_chain] dataset ready\n");

    randomx_vm* vm = randomx_create_vm(RANDOMX_FLAG_FULL_MEM, nullptr, dataset);
    if (!vm) { fprintf(stderr, "vm create failed\n"); return 1; }
    int nprog = (argc > 3) ? atoi(argv[3]) : RANDOMX_PROGRAM_COUNT;
    if (nprog > RANDOMX_PROGRAM_COUNT) nprog = RANDOMX_PROGRAM_COUNT;

    std::vector<uint8_t> input = base;
    embed_nonce(input, 0);

    alignas(16) uint64_t tempHash[8];
    blake2b(tempHash, sizeof(tempHash), input.data(), input.size(), nullptr, 0);
    dump("cpu_chain_h0_input.bin", tempHash, 64);
    phex("[cpu_chain] h0_input ", (const uint8_t*)tempHash, 64);

    vm->initScratchpad(tempHash); // evolves tempHash (fillAes1Rx4 writes state back)
    dump("cpu_chain_h0_evolved.bin", tempHash, 64);
    phex("[cpu_chain] h0_evolved ", (const uint8_t*)tempHash, 64);

    vm->resetRoundingMode();
    for (int chain = 0; chain < nprog; ++chain) {
        {
            char name[64];
            snprintf(name, sizeof(name), "cpu_scratch_p%d.bin", chain);
            dump(name, vm->getScratchpad(), RANDOMX_SCRATCHPAD_L3);
        }
        vm->run(tempHash); // generateProgram(tempHash) + initialize + execute; evolves tempHash via fillAes4Rx4
        {
            char name[64];
            snprintf(name, sizeof(name), "cpu_prog_p%d.bin", chain);
            dump(name, &vm->getProgram(), sizeof(randomx::Program));
        }

        char name[64];
        snprintf(name, sizeof(name), "cpu_rf_p%d.bin", chain);
        dump(name, vm->getRegisterFile(), sizeof(randomx::RegisterFile));
        fprintf(stderr, "[cpu_chain] p%d fprc_after=%u\n", chain, (unsigned)rx_get_rounding_mode());

        if (chain < RANDOMX_PROGRAM_COUNT - 1) {
            blake2b(tempHash, sizeof(tempHash), vm->getRegisterFile(), sizeof(randomx::RegisterFile), nullptr, 0);
            snprintf(name, sizeof(name), "cpu_chain_h%d.bin", chain + 1);
            dump(name, tempHash, 64);
            phex(("[cpu_chain] h" + std::to_string(chain + 1) + " ").c_str(), (const uint8_t*)tempHash, 64);
        }
    }

    std::vector<uint8_t> out(RANDOMX_HASH_SIZE);
    vm->getFinalResult(out.data(), RANDOMX_HASH_SIZE);
    dump("cpu_final_hash.bin", out.data(), out.size());
    phex("[cpu_chain] final ", out.data(), out.size());

    randomx_destroy_vm(vm);
    randomx_release_dataset(dataset);
    randomx_release_cache(cache);
    return 0;
}
