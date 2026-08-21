// Dump the full reference RandomX dataset (memory) to cpu_dataset_full.bin.
// Usage: dump_dataset.exe <seed_hex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include "randomx.h"

static std::vector<uint8_t> hex2bin(const char* s) {
    std::vector<uint8_t> out;
    size_t len = strlen(s);
    for (size_t i = 0; i + 1 < len; i += 2) {
        unsigned v = 0; sscanf(s + i, "%2x", &v);
        out.push_back((uint8_t)v);
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: dump_dataset <seed_hex>\n"); return 1; }
    auto seed = hex2bin(argv[1]);
    randomx_cache* cache = randomx_alloc_cache(RANDOMX_FLAG_DEFAULT);
    randomx_init_cache(cache, seed.data(), seed.size());
    fprintf(stderr, "[dump_dataset] cache ok\n");
    randomx_dataset* ds = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
    unsigned long items = randomx_dataset_item_count();
    unsigned hw = std::thread::hardware_concurrency(); if (hw < 1) hw = 1;
    unsigned long per = (items + hw - 1) / hw;
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < hw; ++t) {
        unsigned long s = (unsigned long)t * per, e = s + per; if (e > items) e = items;
        if (s >= e) continue;
        ts.emplace_back(randomx_init_dataset, ds, cache, s, e - s);
    }
    for (auto& th : ts) th.join();
    fprintf(stderr, "[dump_dataset] dataset ok (%lu items)\n", items);
    const void* mem = randomx_get_dataset_memory(ds);
    size_t bytes = (size_t)items * RANDOMX_DATASET_ITEM_SIZE;
    FILE* f = fopen("cpu_dataset_full.bin", "wb");
    if (!f) return 1;
    fwrite(mem, 1, bytes, f);
    fclose(f);
    fprintf(stderr, "[dump_dataset] wrote cpu_dataset_full.bin (%zu bytes)\n", bytes);
    randomx_release_dataset(ds);
    randomx_release_cache(cache);
    return 0;
}
