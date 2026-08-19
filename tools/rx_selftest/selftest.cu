// GPU self-test for xmrig-cuda RandomX kernels.
//
// Reuses the exact RandomX pipeline from the plugin (blake2b_initial_hash ->
// fillAes1Rx4 -> fillAes4Rx4 -> init_vm -> execute_vm -> hashAes1Rx4 ->
// blake2b_hash_registers) compiled the same way as the shipping DLL, then prints
// the resulting 32-byte hash per work item. Compare against oracle.exe output.
//
// Usage: selftest.exe <seed_hex> <base_input_hex> <nonce_count>
//   Prints one line per item:  "<index> <hash_hex_64chars>"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <thread>
#include <chrono>

#include <hip/hip_runtime.h>
#include "cuda_device.hpp"
#include "cryptonight.h"
#include "randomx.h"

#include "randomx.cu"

namespace RandomX_Monero {
    #include "RandomX/monero/configuration.h"
    #define fillAes4Rx4 fillAes4Rx4_v104
    #include "RandomX/common.hpp"
}

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

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <seed_hex> <base_input_hex> <nonce_count>\n", argv[0]);
        return 1;
    }

    auto seed = hex2bin(argv[1]);
    auto base = hex2bin(argv[2]);
    uint32_t count = (uint32_t)strtoul(argv[3], nullptr, 10);
    if (base.size() < 43) { fprintf(stderr, "base input must be >= 43 bytes\n"); return 1; }

    HIP_CHECK(0, hipSetDevice(0));

    // --- Build dataset on CPU using the reference library (golden source) ---
    randomx_cache* cache = randomx_alloc_cache(RANDOMX_FLAG_DEFAULT);
    randomx_init_cache(cache, seed.data(), seed.size());
    fprintf(stderr, "[selftest] cache ok\n");
    randomx_dataset* dataset = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
    {
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
    }
    fprintf(stderr, "[selftest] dataset ok\n");
    void* ds_mem = randomx_get_dataset_memory(dataset);
    size_t ds_size = (size_t)randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE;

    // --- Set up the xmrig-cuda context exactly like the plugin host would ---
    nvid_ctx ctx;
    ctx.device_id = 0;
    ctx.algorithm = xmrig_cuda::Algorithm(xmrig_cuda::Algorithm::RX_0);
    ctx.rx_dataset_host = -1; // forces dataset upload via hipMalloc+copy
    // Optional 4th arg: bfactor (default 6). Lets us A/B the bfactor=12 clamp fix.
    ctx.device_bfactor = (argc >= 5) ? (int)strtol(argv[4], nullptr, 10) : 6;
    ctx.device_bsleep  = (argc >= 6) ? (int)strtol(argv[5], nullptr, 10) : 25;
    fprintf(stderr, "[selftest] bfactor=%d bsleep=%d\n", ctx.device_bfactor, ctx.device_bsleep);

    uint32_t batch_size = ((count + 31) / 32) * 32;
    if (batch_size < 32) batch_size = 32;

    randomx_prepare(&ctx, ds_mem, ds_size, batch_size);

    // input buffer (nonce bytes left zero; hash() embeds the nonce)
    HIP_CHECK(0, hipMalloc(&ctx.d_input, base.size()));
    HIP_CHECK(0, hipMemcpy(ctx.d_input, base.data(), base.size(), hipMemcpyHostToDevice));
    ctx.inputlen = (int)base.size();

    HIP_CHECK(0, hipMalloc(&ctx.d_result_count, sizeof(uint32_t)));
    HIP_CHECK(0, hipMalloc(&ctx.d_result_nonce, 10 * sizeof(uint32_t)));

    uint32_t rescount = 0;
    std::vector<uint32_t> resnonce(10, 0);

    // DEBUG: standalone first-hash dump for item 0 (nonce 0) to compare vs reference blake2b
    {
        std::vector<uint64_t> fh(8);
        RandomX_Monero::blake2b_initial_hash<<<1, 32>>>(ctx.d_rx_hashes, ctx.d_input, ctx.inputlen, (uint32_t)0);
        HIP_CHECK(0, hipDeviceSynchronize());
        HIP_CHECK(0, hipMemcpy(fh.data(), ctx.d_rx_hashes, 64, hipMemcpyDeviceToHost));
        FILE* f = fopen("gpu_firsthash.bin", "wb");
        if (f) { fwrite(fh.data(), 1, 64, f); fclose(f); }
        fprintf(stderr, "[selftest] GPU firstHash[0..7] (item0, nonce0):\n");
        for (int i = 0; i < 8; ++i) fprintf(stderr, "%016llx ", (unsigned long long)fh[i]);
        fprintf(stderr, "\n");
    }

    // DEBUG: dump fillAes4Rx4_v104 per-round intermediates for item 0, sub 0.
    {
        uint32_t* d_dbg = nullptr;
        HIP_CHECK(0, hipMalloc(&d_dbg, 8 * sizeof(uint32_t)));
        uint32_t* d_out = nullptr;
        HIP_CHECK(0, hipMalloc(&d_out, 32u * 2176u));
        RandomX_Monero::fillAes4Rx4_v104_dbg<2176, false><<<1, 128>>>(ctx.d_rx_hashes, d_out, (uint32_t)32, d_dbg);
        HIP_CHECK(0, hipDeviceSynchronize());
        std::vector<uint32_t> dbg(8);
        HIP_CHECK(0, hipMemcpy(dbg.data(), d_dbg, 8 * sizeof(uint32_t), hipMemcpyDeviceToHost));
        fprintf(stderr, "[dbg] fillAes4Rx4_v104 item0 intermediates (x0..3 in, y1, x2, y3, x4 out):\n");
        for (int i = 0; i < 8; ++i) fprintf(stderr, "  dbg[%d]=%08x\n", i, dbg[i]);
        FILE* f = fopen("gpu_fill_dbg.bin", "wb"); if (f) { fwrite(dbg.data(), 1, 32, f); fclose(f); }
        HIP_CHECK(0, hipFree(d_dbg)); HIP_CHECK(0, hipFree(d_out));
    }

    // DEBUG: execute_vm_dbg - run VM iterations with per-iteration state dump for item 0
    {
        fprintf(stderr, "[dbg] Running execute_vm_dbg for item 0...\n");
        uint64_t* d_dbg_vm = nullptr;
        uint32_t* d_dbg_idx = nullptr;
        const uint32_t max_snapshots = 8192;
        HIP_CHECK(0, hipMalloc(&d_dbg_vm, max_snapshots * 24 * sizeof(uint64_t)));
        HIP_CHECK(0, hipMalloc(&d_dbg_idx, sizeof(uint32_t)));
        HIP_CHECK(0, hipMemset(d_dbg_idx, 0, sizeof(uint32_t)));

        // Run fillAes4Rx4 first to populate entropy (same as hash() does)
        RandomX_Monero::fillAes4Rx4_v104<2176, false><<<ctx.rx_batch_size / 32, 32 * 4>>>(ctx.d_rx_hashes, ctx.d_rx_entropy, ctx.rx_batch_size);
        HIP_CHECK(0, hipDeviceSynchronize());

        // DEBUG: dump entropy buffer right after fillAes4Rx4 to verify layout
        {
            std::vector<uint8_t> ent(2304);  // ENTROPY_SIZE
            HIP_CHECK(0, hipMemcpy(ent.data(), ctx.d_rx_entropy, 2304, hipMemcpyDeviceToHost));
            FILE* f = fopen("gpu_entropy_dbg.bin", "wb");
            if (f) { fwrite(ent.data(), 1, 2304, f); fclose(f); }
            fprintf(stderr, "[dbg] GPU entropy first 16 uint64 after fillAes4Rx4:\n");
            for (int i = 0; i < 16; ++i) {
                uint64_t v = *(uint64_t*)(ent.data() + i * 8);
                fprintf(stderr, "  entropy[%d] = %016llx\n", i, (unsigned long long)v);
            }
        }

        // Run init_vm first (same as hash() does)
        RandomX_Monero::init_vm<8><<<ctx.rx_batch_size / 4, 4 * 8>>>(ctx.d_rx_entropy, ctx.d_rx_vm_states);
        HIP_CHECK(0, hipDeviceSynchronize());

        // DEBUG: dump item 0's compiled program (256 uint32 = 1024 bytes) for comparison with tevador
        {
            const size_t PROG_SIZE = 1024; // RANDOMX_PROGRAM_SIZE * 4
            std::vector<uint8_t> prog(PROG_SIZE);
            HIP_CHECK(0, hipMemcpy(prog.data(), (uint8_t*)ctx.d_rx_vm_states + 1024, PROG_SIZE, hipMemcpyDeviceToHost));
            FILE* f = fopen("gpu_program.bin", "wb");
            if (f) { fwrite(prog.data(), 1, PROG_SIZE, f); fclose(f); }
            fprintf(stderr, "[dbg] dumped gpu_program.bin (%zu bytes)\n", PROG_SIZE);
            // Print first 16 instructions
            fprintf(stderr, "  GPU program (first 16):\n");
            for (int i = 0; i < 16; ++i) {
                uint32_t inst = ((uint32_t*)prog.data())[i];
                uint32_t opcode = (inst >> 20) & 15;
                uint32_t dst = (inst >> 0) & 7;
                uint32_t src = (inst >> 3) & 7;
                uint32_t loc = (inst >> 14) & 1;
                uint32_t num_workers = (inst >> 24) & 15;
                uint32_t num_fp = (inst >> 28) & 15;
                fprintf(stderr, "    [%2d] %08x op=%u dst=%u src=%u loc=%u workers=%u fp=%u\n", i, inst, opcode, dst, src, loc, num_workers, num_fp);
            }
        }

        #if 0
        // Run execute_vm_dbg for all iterations (bfactor=6 -> 32 iterations per call, 64 calls = 2048 total)
        const int effective_bfactor = 6;  // matching ctx.device_bfactor default
        const int n = 1 << effective_bfactor;
        const int num_iterations = RANDOMX_PROGRAM_ITERATIONS >> effective_bfactor;
        for (int j = 0; j < n; ++j) {
            RandomX_Monero::execute_vm_dbg<8, false><<<ctx.rx_batch_size / 4, 4 * 8>>>(
                ctx.d_rx_vm_states, ctx.d_rx_rounding, ctx.d_long_state, ctx.d_rx_dataset,
                ctx.rx_batch_size, num_iterations, j == 0, j == n - 1, d_dbg_vm, d_dbg_idx);
            HIP_CHECK(0, hipDeviceSynchronize());
        }

        uint32_t h_dbg_idx = 0;
        HIP_CHECK(0, hipMemcpy(&h_dbg_idx, d_dbg_idx, sizeof(uint32_t), hipMemcpyDeviceToHost));
        fprintf(stderr, "[dbg] execute_vm_dbg captured %u VM state snapshots\n", h_dbg_idx);

        if (h_dbg_idx > 0) {
            std::vector<uint64_t> dbg_vm(h_dbg_idx * 24);
            HIP_CHECK(0, hipMemcpy(dbg_vm.data(), d_dbg_vm, h_dbg_idx * 24 * sizeof(uint64_t), hipMemcpyDeviceToHost));
            FILE* f = fopen("gpu_vm_dbg.bin", "wb");
            if (f) { fwrite(dbg_vm.data(), 1, h_dbg_idx * 24 * sizeof(uint64_t), f); fclose(f); }
            fprintf(stderr, "[dbg] dumped gpu_vm_dbg.bin (%zu bytes, %u snapshots)\n", h_dbg_idx * 24 * sizeof(uint64_t), h_dbg_idx);
            // Print first few snapshots
            for (uint32_t snap = 0; snap < std::min<uint32_t>(h_dbg_idx, 4); ++snap) {
                fprintf(stderr, "  snapshot %u:\n", snap);
                fprintf(stderr, "    R: ");
                for (int i = 0; i < 8; ++i) fprintf(stderr, "%016llx ", (unsigned long long)dbg_vm[snap * 24 + i]);
                fprintf(stderr, "\n    F: ");
                for (int i = 0; i < 8; ++i) fprintf(stderr, "%016llx ", (unsigned long long)dbg_vm[snap * 24 + 8 + i]);
                fprintf(stderr, "\n    E: ");
                for (int i = 0; i < 8; ++i) fprintf(stderr, "%016llx ", (unsigned long long)dbg_vm[snap * 24 + 16 + i]);
                fprintf(stderr, "\n");
            }
        }
#endif

        HIP_CHECK(0, hipFree(d_dbg_vm)); HIP_CHECK(0, hipFree(d_dbg_idx));
    }

    for (int iter = 0; iter < 10; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        RandomX_Monero::hash(&ctx, (uint32_t)(iter * batch_size), 39, 0, &rescount, resnonce.data(), batch_size);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "[selftest] hash iter %d done in %.1f ms (rescount=%u)\n", iter, ms, rescount);
    }

    // DEBUG: dump item 0's fillAes4Rx4 entropy (program) to compare vs tevador
    {
        const size_t ENT = 2176;
        std::vector<uint8_t> ent(ENT);
        HIP_CHECK(0, hipMemcpy(ent.data(), ctx.d_rx_entropy, ENT, hipMemcpyDeviceToHost));
        FILE* f = fopen("gpu_entropy.bin", "wb");
        if (f) { fwrite(ent.data(), 1, ENT, f); fclose(f); }
        fprintf(stderr, "[selftest] gpu_entropy[0..15]: ");
        for (int i = 0; i < 16; ++i) fprintf(stderr, "%02x ", ent[i]);
        fprintf(stderr, "\n");
    }

    // DEBUG: dump the folded register file (first 256 bytes of vm_states) for item 0
    {
        std::vector<uint8_t> rf(RandomX_Monero::VM_STATE_SIZE);
        HIP_CHECK(0, hipMemcpy(rf.data(), ctx.d_rx_vm_states, RandomX_Monero::VM_STATE_SIZE, hipMemcpyDeviceToHost));
        FILE* f = fopen("gpu_rf0.bin", "wb");
        if (f) { fwrite(rf.data(), 1, 256, f); fclose(f); }
        fprintf(stderr, "[selftest] GPU vm_state0 (256B register file):\n");
        for (int i = 0; i < 32; ++i) fprintf(stderr, "%016llx ", (unsigned long long)((uint64_t*)rf.data())[i]);
        fprintf(stderr, "\n");
    }

    // blake2b_hash_registers writes 32 bytes (8 uint32) per item at d_rx_hashes[i*8]
    std::vector<uint32_t> hashes(batch_size * 8);
    HIP_CHECK(0, hipMemcpy(hashes.data(), ctx.d_rx_hashes, batch_size * 8 * sizeof(uint32_t), hipMemcpyDeviceToHost));

    for (uint32_t i = 0; i < count; ++i) {
        printf("%u ", i);
        for (int j = 0; j < 8; ++j) printf("%08x", hashes[i * 8 + j]);
        printf("\n");
    }

    randomx_release_dataset(dataset);
    randomx_release_cache(cache);
    return 0;
}
