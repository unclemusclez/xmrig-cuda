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

#include <cstdint>
#include <cstdio>

__global__ void find_shares(const void* hashes, uint64_t target, uint32_t* shares)
{
    const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t* p = (const uint64_t*)hashes;

    if (p[global_index * 4 + 3] < target) {
        const uint32_t idx = atomicInc(shares, 0xFFFFFFFF) + 1;
        if (idx < 10) {
            shares[idx] = global_index;
        }
    }
}

__global__ void debug_validate_hashes(const uint64_t* hashes, uint32_t batch_size, uint64_t target,
                                       uint32_t* total, uint32_t* valid, uint32_t* invalid_count,
                                       uint32_t* invalid_nonces, uint32_t max_invalid)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *total = batch_size;
        *valid = 0;
        *invalid_count = 0;

        for (uint32_t i = 0; i < batch_size; ++i) {
            const uint64_t* h = hashes + i * 4;
            if (h[3] < target) {
                (*valid)++;
            } else {
                uint32_t idx = *invalid_count;
                if (idx < max_invalid) {
                    invalid_nonces[idx] = i;
                }
                (*invalid_count)++;
            }
        }
    }
}

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

    for (size_t i = 0; i < RANDOMX_PROGRAM_COUNT; ++i) {
        CUDA_CHECK_KERNEL(ctx->device_id, fillAes4Rx4<ENTROPY_SIZE, false><<<batch_size / 32, 32 * 4>>>(ctx->d_rx_hashes, ctx->d_rx_entropy, batch_size));

        CUDA_CHECK_KERNEL(ctx->device_id, init_vm<8><<<batch_size / 4, 4 * 8>>>(ctx->d_rx_entropy, ctx->d_rx_vm_states));
        for (int j = 0, n = 1 << ctx->device_bfactor; j < n; ++j) {
            CUDA_CHECK_KERNEL(ctx->device_id, execute_vm<8, false><<<batch_size / 2, 2 * 8>>>(ctx->d_rx_vm_states, ctx->d_rx_rounding, ctx->d_long_state, ctx->d_rx_dataset, batch_size, RANDOMX_PROGRAM_ITERATIONS >> ctx->device_bfactor, j == 0, j == n - 1));
        }

        if (i == RANDOMX_PROGRAM_COUNT - 1) {
            CUDA_CHECK_KERNEL(ctx->device_id, hashAes1Rx4<RANDOMX_SCRATCHPAD_L3, 192, VM_STATE_SIZE, 64><<<batch_size / 32, 32 * 4>>>(ctx->d_long_state, ctx->d_rx_vm_states, batch_size));
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 32><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states));
        } else {
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 64><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states));
        }
    }

    CUDA_CHECK(ctx->device_id, hipMemset(ctx->d_result_nonce, 0, 10 * sizeof(uint32_t)));
    CUDA_CHECK_KERNEL(ctx->device_id, find_shares<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, target, ctx->d_result_nonce));

    if (ctx->d_rx_debug_total) {
        CUDA_CHECK_KERNEL(ctx->device_id, debug_validate_hashes<<<1, 1>>>(
            (const uint64_t*)ctx->d_rx_hashes, batch_size, target,
            ctx->d_rx_debug_total, ctx->d_rx_debug_valid,
            ctx->d_rx_debug_invalid_count, ctx->d_rx_debug_invalid_nonces, 16));

        uint32_t h_total = 0, h_valid = 0, h_invalid = 0;
        uint32_t h_invalid_nonces[16] = {0};

        CUDA_CHECK(ctx->device_id, hipMemcpy(&h_total, ctx->d_rx_debug_total, sizeof(uint32_t), hipMemcpyDeviceToHost));
        CUDA_CHECK(ctx->device_id, hipMemcpy(&h_valid, ctx->d_rx_debug_valid, sizeof(uint32_t), hipMemcpyDeviceToHost));
        CUDA_CHECK(ctx->device_id, hipMemcpy(&h_invalid, ctx->d_rx_debug_invalid_count, sizeof(uint32_t), hipMemcpyDeviceToHost));
        CUDA_CHECK(ctx->device_id, hipMemcpy(h_invalid_nonces, ctx->d_rx_debug_invalid_nonces, sizeof(h_invalid_nonces), hipMemcpyDeviceToHost));

        printf("[RX-DEBUG] batch=%u total=%u valid=%u invalid=%u",
               batch_size, h_total, h_valid, h_invalid);
        if (h_invalid > 0 && h_invalid <= 16) {
            printf(" nonces:");
            for (uint32_t i = 0; i < h_invalid; ++i) {
                printf(" %u", h_invalid_nonces[i]);
            }
        }
        printf("\n");
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
