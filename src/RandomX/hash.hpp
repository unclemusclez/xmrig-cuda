#pragma once

/*
Copyright (c) 2019 SChernykh

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX CUDA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX CUDA.  If not, see<http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cstdint>
#include <cstdio>

void hash(nvid_ctx *ctx, uint32_t nonce, uint32_t nonce_offset, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size)
{
    if (ctx->inputlen <= 128) {
        CUDA_CHECK_KERNEL(ctx->device_id, blake2b_initial_hash<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce));
    }
    else if (ctx->inputlen <= 256) {
        CUDA_CHECK_KERNEL(ctx->device_id, blake2b_initial_hash_double<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce));
    }
    else {
        CUDA_CHECK_KERNEL(ctx->device_id, blake2b_initial_hash_big<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce, nonce_offset));
    }

    CUDA_CHECK_KERNEL(ctx->device_id, fillAes1Rx4<RANDOMX_SCRATCHPAD_L3, false, 64><<<batch_size / 32, 32 * 4>>>(ctx->d_rx_hashes, ctx->d_long_state, batch_size));
    CUDA_CHECK(ctx->device_id, hipMemset(ctx->d_rx_rounding, 0, batch_size * sizeof(uint32_t)));
    CUDA_CHECK(ctx->device_id, hipMemset(ctx->d_result_nonce, 0, 10 * sizeof(uint32_t)));

    for (size_t i = 0; i < RANDOMX_PROGRAM_COUNT; ++i) {
        CUDA_CHECK_KERNEL(ctx->device_id, fillAes4Rx4<ENTROPY_SIZE, false><<<batch_size / 32, 32 * 4>>>(ctx->d_rx_hashes, ctx->d_rx_entropy, batch_size));

#ifdef RX_DEBUG_STAGE
        {
            static bool dumped = false;
            if (!dumped) {
                dumped = true;
                std::vector<uint8_t> sp(RANDOMX_SCRATCHPAD_L3);
                CUDA_CHECK(ctx->device_id, hipMemcpy(sp.data(), ctx->d_long_state, RANDOMX_SCRATCHPAD_L3, hipMemcpyDeviceToHost));
                FILE* f = fopen("gpu_scratch.bin", "wb");
                if (f) { fwrite(sp.data(), 1, RANDOMX_SCRATCHPAD_L3, f); fclose(f); }
                std::vector<uint8_t> ent(ENTROPY_SIZE);
                CUDA_CHECK(ctx->device_id, hipMemcpy(ent.data(), ctx->d_rx_entropy, ENTROPY_SIZE, hipMemcpyDeviceToHost));
                FILE* f2 = fopen("gpu_entropy_p0.bin", "wb");
                if (f2) { fwrite(ent.data(), 1, ENTROPY_SIZE, f2); fclose(f2); }
                std::vector<uint8_t> ds(1u << 14);
                CUDA_CHECK(ctx->device_id, hipMemcpy(ds.data(), ctx->d_rx_dataset, ds.size(), hipMemcpyDeviceToHost));
                FILE* f3 = fopen("gpu_dataset.bin", "wb");
                if (f3) { fwrite(ds.data(), 1, ds.size(), f3); fclose(f3); }
                fprintf(stderr, "[hash] RX_DEBUG_STAGE: dumped gpu_scratch.bin (%llu B) + gpu_entropy.bin (%llu B) + gpu_dataset.bin (%llu B) for item 0\n",
                    (unsigned long long)RANDOMX_SCRATCHPAD_L3, (unsigned long long)ENTROPY_SIZE, (unsigned long long)ds.size());
            }
        }
#endif

        CUDA_CHECK_KERNEL(ctx->device_id, init_vm<8><<<batch_size / 4, 4 * 8>>>(ctx->d_rx_entropy, ctx->d_rx_vm_states));
        const int effective_bfactor = std::min(ctx->device_bfactor, 8);
        for (int j = 0, n = 1 << effective_bfactor; j < n; ++j) {
            CUDA_CHECK_KERNEL(ctx->device_id, execute_vm<8, false><<<batch_size / 4, 4 * 8>>>(ctx->d_rx_vm_states, ctx->d_rx_rounding, ctx->d_long_state, ctx->d_rx_dataset, batch_size, RANDOMX_PROGRAM_ITERATIONS >> effective_bfactor, j == 0, j == n - 1));
        }

        if (i == RANDOMX_PROGRAM_COUNT - 1) {
            CUDA_CHECK_KERNEL(ctx->device_id, hashAes1Rx4<RANDOMX_SCRATCHPAD_L3, 192, VM_STATE_SIZE, 64><<<batch_size / 32, 32 * 4>>>(ctx->d_long_state, ctx->d_rx_vm_states, batch_size));
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 32, true><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states, target, ctx->d_result_nonce));
        } else {
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 64, false><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states, target, ctx->d_result_nonce));
        }
    }

    CUDA_CHECK(ctx->device_id, hipDeviceSynchronize());
    CUDA_CHECK(ctx->device_id, hipMemcpy(resnonce, ctx->d_result_nonce, 10 * sizeof(uint32_t), hipMemcpyDeviceToHost));

    *rescount = resnonce[0];
    if (*rescount > 9) {
        *rescount = 9;
    }

    for (uint32_t i = 0; i < *rescount; i++) {
        resnonce[i] = resnonce[i + 1] + nonce;
    }
}
