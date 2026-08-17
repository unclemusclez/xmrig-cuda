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


#include "cryptonight.h"
#include "cuda_device.hpp"


#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <cstdint>


void randomx_prepare(nvid_ctx *ctx, const void *dataset, size_t dataset_size, uint32_t batch_size)
{
    ctx->rx_batch_size      = batch_size;
    ctx->d_scratchpads_size = batch_size * (ctx->algorithm.l3() + 64);

    if (ctx->rx_dataset_host > 0) {
        void* devPtr = nullptr;
        HIP_CHECK(ctx->device_id, hipHostGetDevicePointer(&devPtr, const_cast<void *>(dataset), 0));
        ctx->d_rx_dataset = static_cast<uint32_t*>(devPtr);
    }
    else {
        HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_rx_dataset, dataset_size));
        HIP_CHECK(ctx->device_id, hipMemcpy(ctx->d_rx_dataset, dataset, dataset_size, hipMemcpyHostToDevice));
    }

    HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_long_state, ctx->d_scratchpads_size));
    HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_rx_hashes, batch_size * 64));
    HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_rx_entropy, batch_size * (128 + 2560)));
    HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_rx_vm_states, batch_size * 2560));
    HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_rx_rounding, batch_size * sizeof(uint32_t)));

    HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_rx_debug_total, sizeof(uint32_t)));
    HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_rx_debug_valid, sizeof(uint32_t)));
    HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_rx_debug_invalid_count, sizeof(uint32_t)));
    HIP_CHECK(ctx->device_id, hipMalloc(&ctx->d_rx_debug_invalid_nonces, 16 * sizeof(uint32_t)));
}


void randomx_update_dataset(nvid_ctx* ctx, const void* dataset, size_t dataset_size)
{
    if (ctx->rx_dataset_host > 0) {
        return;
    }

    HIP_CHECK(ctx->device_id, hipMemcpy(ctx->d_rx_dataset, dataset, dataset_size, hipMemcpyHostToDevice));
}
