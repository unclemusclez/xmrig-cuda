// CPU reference stage dumper for xmrig-cuda RandomX GPU self-test.
//
// Given the GPU's dumped first hash (gpu_firsthash.bin) and the selftest's
// dumped GPU stages (gpu_scratch.bin = post-fillAes1Rx4, gpu_entropy.bin =
// post-fillAes4Rx4), this tool reproduces tevador's equivalents:
//   * scratchpad  -> vm->initScratchpad(gpu_firsthash)  -> cpu_scratch.bin
//   * program     -> vm->getProgram() after calculate_hash -> cpu_program.bin
//
// Comparing gpu_*.bin to cpu_*.bin isolates fillAes1Rx4 / fillAes4Rx4.
//
// Usage: stagecmp.exe <seed_hex> <base_hex>
//   seed_hex  : same seed passed to selftest (builds the dataset)
//   base_hex  : same >=43-byte base passed to selftest (drives calculate_hash)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>

#include "randomx.h"
#include "virtual_machine.hpp"
#include "aes_hash.hpp"
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
    in[39] = (uint8_t)(nonce & 0xFF);
    in[40] = (uint8_t)((nonce >> 8) & 0xFF);
    in[41] = (uint8_t)((nonce >> 16) & 0xFF);
    in[42] = (uint8_t)((nonce >> 24) & 0xFF);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <seed_hex> <base_hex>\n", argv[0]);
        return 1;
    }
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

    randomx_vm* vm = randomx_create_vm(RANDOMX_FLAG_FULL_MEM, nullptr, dataset);
    if (vm == nullptr) { fprintf(stderr, "vm create failed\n"); return 1; }

    // Tevador's own first hash (blake2b of the base input) for seed-equality check.
    {
        std::vector<uint8_t> input = base;
        embed_nonce(input, 0);
        uint8_t th[64];
        blake2b(th, sizeof(th), input.data(), input.size(), nullptr, 0);
        FILE* f = fopen("cpu_firsthash.bin", "wb");
        if (f) { fwrite(th, 1, 64, f); fclose(f); }
        fprintf(stderr, "[stagecmp] dumped cpu_firsthash.bin (tevador blake2b of base)\n");

        // Tevador fillAes4Rx4 entropy (the program seed buffer), matching the GPU's
        // load_entropy stage which fills d_rx_entropy via fillAes4Rx4(firsthash).
        const size_t ENTROPY_SIZE = 128 + ((RANDOMX_PROGRAM_SIZE_V1 * 8 + 127) / 128) * 128;
        std::vector<uint8_t> entropy(ENTROPY_SIZE);
        fillAes4Rx4<false>(th, ENTROPY_SIZE, entropy.data());
        FILE* f2 = fopen("cpu_entropy.bin", "wb");
        if (f2) { fwrite(entropy.data(), 1, ENTROPY_SIZE, f2); fclose(f2); }
        fprintf(stderr, "[stagecmp] dumped cpu_entropy.bin (%zu B)\n", ENTROPY_SIZE);
    }

    // --- Scratchpad: fill from the GPU's actual first hash ---
    {
        FILE* fh = fopen("gpu_firsthash.bin", "rb");
        if (!fh) { fprintf(stderr, "gpu_firsthash.bin missing (run selftest first)\n"); return 1; }
        uint8_t firstHash[64];
        size_t nr = fread(firstHash, 1, 64, fh);
        fclose(fh);
        fprintf(stderr, "[stagecmp] read gpu_firsthash.bin: %zu bytes\n", nr);

        vm->initScratchpad(firstHash);
        const void* sp = vm->getScratchpad();
        FILE* f = fopen("cpu_scratch.bin", "wb");
        if (f) { fwrite(sp, 1, RANDOMX_SCRATCHPAD_L3, f); fclose(f); }
        fprintf(stderr, "[stagecmp] dumped cpu_scratch.bin (%u bytes)\n", (unsigned)RANDOMX_SCRATCHPAD_L3);
    }

    // --- Program: populate via calculate_hash, then dump raw Program ---
    {
        std::vector<uint8_t> input = base;
        embed_nonce(input, 0);
        std::vector<uint8_t> hash(RANDOMX_HASH_SIZE);
        randomx_calculate_hash(vm, input.data(), input.size(), hash.data());

        const randomx::Program& prog = vm->getProgram();
        FILE* fp = fopen("cpu_program.bin", "wb");
        if (fp) { fwrite(&prog, 1, sizeof(randomx::Program), fp); fclose(fp); }
        fprintf(stderr, "[stagecmp] dumped cpu_program.bin (%zu bytes)\n", sizeof(randomx::Program));

        printf("HASH ");
        for (int i = 0; i < RANDOMX_HASH_SIZE; ++i) printf("%02x", hash[i]);
        printf("\n");

        // Scratchpad AFTER calculate_hash (filled from tevador's OWN first hash).
        // Compare to cpu_scratch.bin (filled from gpu_firsthash.bin) to confirm
        // tevador's first hash == gpu_firsthash.bin (i.e. seeds are identical).
        const void* sp2 = vm->getScratchpad();
        FILE* f2 = fopen("cpu_scratch2.bin", "wb");
        if (f2) { fwrite(sp2, 1, RANDOMX_SCRATCHPAD_L3, f2); fclose(f2); }
        fprintf(stderr, "[stagecmp] dumped cpu_scratch2.bin (post-calculate_hash)\n");
    }

    // --- Dataset slice (validate the upload path) ---
    {
        const void* dm = randomx_get_dataset_memory(dataset);
        FILE* fd = fopen("cpu_dataset.bin", "wb");
        if (fd) { fwrite(dm, 1, (1u << 14), fd); fclose(fd); }
        fprintf(stderr, "[stagecmp] dumped cpu_dataset.bin (16384 bytes)\n");
    }

    randomx_destroy_vm(vm);
    randomx_release_dataset(dataset);
    randomx_release_cache(cache);
    return 0;
}
