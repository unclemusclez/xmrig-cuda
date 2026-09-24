#pragma once

/*
Copyright (c) 2019 SChernykh
Portions Copyright (c) 2018-2019 tevador

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

#include <cmath>

#if defined(RX_TRACE_VM) && defined(__HIP_DEVICE_COMPILE__)
#define TRACE_VM_STATE(tag, ip, sub, R, fprc, ma, mx, spAddr0, spAddr1) \
    do { \
        if (threadIdx.x == 0 && blockIdx.x == 0) { \
            printf("[VM_TRACE] %s ip=%u sub=%u fprc=%u ma=%u mx=%u sp0=%u sp1=%u\n", \
                   tag, ip, sub, fprc, ma, mx, spAddr0, spAddr1); \
            for (int i = 0; i < 8; ++i) { \
                printf("  R[%d]=%016llx\n", i, (unsigned long long)R[i]); \
            } \
            double* F = (double*)(R + 8); \
            double* E = (double*)(R + 16); \
            for (int i = 0; i < 8; ++i) { \
                printf("  F[%d]=%016llx E[%d]=%016llx\n", i, (unsigned long long)__double_as_longlong(F[i]), i, (unsigned long long)__double_as_longlong(E[i])); \
            } \
        } \
    } while (0)
#else
#define TRACE_VM_STATE(tag, ip, sub, R, fprc, ma, mx, spAddr0, spAddr1) do {} while (0)
#endif

__device__ __forceinline__ double hip_longlong_as_double(uint64_t x) {
    union { uint64_t u; double d; } c;
    c.u = x;
    return c.d;
}

__device__ __forceinline__ uint64_t hip_double_as_longlong(double x) {
    union { uint64_t u; double d; } c;
    c.d = x;
    return c.u;
}

__device__ __forceinline__ double hip_int2double_rn(int value) {
    return static_cast<double>(value);
}

__device__ __forceinline__ double hip_fma_rn(double a, double b, double c) {
    return fma(a, b, c);
}

__device__ __forceinline__ double hip_ddiv_rn(double a, double b) {
    return a / b;
}

__device__ __forceinline__ double hip_dsqrt_rn(double x) {
    return sqrt(x);
}

#define __longlong_as_double(x) hip_longlong_as_double(x)
#define __double_as_longlong(x) hip_double_as_longlong(x)
#define __int2double_rn(x) hip_int2double_rn(x)
#define __fma_rn(a, b, c) hip_fma_rn(a, b, c)
#define __ddiv_rn(a, b) hip_ddiv_rn(a, b)
#define __dsqrt_rn(x) hip_dsqrt_rn(x)

__device__ double rx_ddiv(double a, double b, int mode);
__device__ double rx_dsqrt(double a, int mode);

constexpr size_t HASH_SIZE = 64;
constexpr size_t ENTROPY_SIZE = 128 + ((RANDOMX_PROGRAM_SIZE * 8 + 127) / 128) * 128;
constexpr size_t REGISTERS_SIZE = 256;
constexpr size_t IMM_BUF_SIZE = RANDOMX_PROGRAM_SIZE * 4 - REGISTERS_SIZE;
constexpr size_t IMM_INDEX_COUNT = (IMM_BUF_SIZE / 4) - 2;

#ifdef RX_LEGACY_DISPATCH
// Legacy dispatch (tag 0.0.2 layout): ip/fprc round-trip through imm_buf in
// LDS. VM state stays 2048 B per hash (Monero) so an execute_vm block of 4
// hashes fits exactly 8 blocks per CU in the 64 KB LDS of gfx1100.
constexpr size_t VM_STATE_SIZE = REGISTERS_SIZE + IMM_BUF_SIZE + RANDOMX_PROGRAM_SIZE * 4;
#else
// Group control flags (fast dispatch): 4 bits per compiled program word.
// Bit0 = word is a live CBRANCH, bit1 = word is a CFROUND. The inner loop
// scans the flags of its current worker group and recomputes branch/rounding
// updates uniformly on all lanes, which removes the per-slot LDS round-trip
// of ip/fprc through imm_buf.
// Bit2 (xlane) = word is a live ISWAP_R (cross-lane register write); bit3
// (scratch) = word touches the scratchpad. Reserved by the compiled-plan v2
// milestones (barrier gating / load pipelining); emitted by init_vm, ignored
// by execute in this revision. Max words = RANDOMX_PROGRAM_SIZE, so the
// region is RANDOMX_PROGRAM_SIZE/2 bytes (128 B for Monero).
//
// PERF NOTE: the extra 128 B/hash grows the LDS block footprint (8192 B ->
// 8704 B for 4 hashes on gfx1100); occupancy stays at 7 blocks per CU
// (65536/8704 = 7.5), same as the 2-bit layout. Fast dispatch is kept for
// experimentation only; builds ship with RX_LEGACY_DISPATCH.
constexpr size_t RX_GROUP_FLAGS_OFFSET = REGISTERS_SIZE + IMM_BUF_SIZE + RANDOMX_PROGRAM_SIZE * 4;
constexpr size_t RX_GROUP_FLAGS_SIZE = (RANDOMX_PROGRAM_SIZE * 4 + 7) / 8;

constexpr size_t VM_STATE_SIZE = RX_GROUP_FLAGS_OFFSET + RX_GROUP_FLAGS_SIZE;
#endif

// Target wavefront size: GCN (gfx6-9, e.g. gfx906) executes wave64, RDNA
// (gfx10+, e.g. gfx1100) executes wave32. The value MUST agree between host
// and device compilation: it sizes LDS buffers on the device and sets the
// launch block size on the host. CMake derives it from
// CMAKE_HIP_ARCHITECTURES and passes it for both passes.
//
// Do NOT fall back to __AMDGCN_WAVEFRONT_SIZE here: the host pass reports 64
// even for wave32 targets (and ROCm 7.2 deprecates the macro), which would
// launch 64-thread blocks into kernels whose LDS is sized for 4 hashes.
// Standalone builds for wave64 targets must pass -DRX_WAVE_SIZE=64. Never
// compile a gfx9 target with -mno-wavefrontsize64 -- the silicon still runs
// 64-wide lanes.
#ifndef RX_WAVE_SIZE
#define RX_WAVE_SIZE 32
#endif

// init_vm/execute_vm blocks are exactly one wavefront wide. With 8 lanes per
// hash, one block processes RX_VM_HASHES_PER_BLOCK hashes (4 on wave32, 8 on
// wave64); keeping the block a full wave avoids half-masked wavefronts on
// wave64 hardware. batch_size must be a multiple of this value.
constexpr int RX_VM_HASHES_PER_BLOCK = RX_WAVE_SIZE / 8;

// execute_vm occupancy hint (blocks per CU) for __launch_bounds__. Must state
// the TRUE LDS-limited residency: an inflated hint makes the compiler squeeze
// VGPRs for occupancy the launch can never achieve (pointless spills).
// Staged bytes/block = RX_VM_HASHES_PER_BLOCK * STAGED_SIZE, STAGED_SIZE =
// 2048 B with RX_PROGRAM_IN_GLOBAL (program served from global/L2) else the
// full 4096 B VM state. gfx906/gfx1100 both have 64 KB LDS per CU.
#ifdef RX_PROGRAM_IN_GLOBAL
#if RX_WAVE_SIZE == 64
constexpr int RX_VM_BLOCKS_PER_CU = 4;  // 8 hashes x 2048 B = 16 KB -> exactly 4
#else
constexpr int RX_VM_BLOCKS_PER_CU = 8;  // 4 hashes x 2048 B = 8 KB -> 8
#endif
#elif RX_WAVE_SIZE == 64
constexpr int RX_VM_BLOCKS_PER_CU = 2;  // 8 hashes x 4096 B = 32 KB -> 2
#else
constexpr int RX_VM_BLOCKS_PER_CU = 4;  // 4 hashes x 4096 B = 16 KB -> 4
#endif

constexpr uint32_t CacheLineSize = 64;
constexpr int ScratchpadL3Mask64 = RANDOMX_SCRATCHPAD_L3 - CacheLineSize;
constexpr uint32_t CacheLineAlignMask = (RANDOMX_DATASET_BASE_SIZE - 1) & ~(CacheLineSize - 1);

__device__ double getSmallPositiveFloatBits(uint64_t entropy)
{
	auto exponent = entropy >> 59;
	auto mantissa = entropy & randomx::mantissaMask;
	exponent += randomx::exponentBias;
	exponent &= randomx::exponentMask;
	exponent <<= randomx::mantissaSize;
	return __longlong_as_double(exponent | mantissa);
}

__device__ uint64_t getStaticExponent(uint64_t entropy)
{
	auto exponent = randomx::constExponentBits;
	exponent |= (entropy >> (64 - randomx::staticExponentBits)) << randomx::dynamicExponentBits;
	exponent <<= randomx::mantissaSize;
	return exponent;
}

__device__ uint64_t getFloatMask(uint64_t entropy)
{
	constexpr uint64_t mask22bit = (1ULL << 22) - 1;
	return (entropy & mask22bit) | getStaticExponent(entropy);
}

template<typename T>
__device__ T bit_cast(double value)
{
	return static_cast<T>(__double_as_longlong(value));
}

__device__ double load_F_E_groups(int value, uint64_t andMask, uint64_t orMask)
{
	uint64_t x = bit_cast<uint64_t>(__int2double_rn(value));
	x &= andMask;
	x |= orMask;
	return __longlong_as_double(static_cast<int64_t>(x));
}

__device__ void set_byte(uint64_t& a, uint32_t position, uint64_t value)
{
	a = (a & ~(0xFFULL << (position << 3))) | (value << (position << 3));
}

__device__ uint32_t get_byte(uint64_t a, uint32_t position)
{
	return static_cast<uint32_t>((a >> (position << 3)) & 0xFF);
}

template<typename T, typename U, size_t N>
__device__ void set_buffer(T (&dst_buf)[N], const U value)
{
	uint32_t i = threadIdx.x * sizeof(T);
	const uint32_t step = blockDim.x * sizeof(T);
	uint8_t* dst = ((uint8_t*) dst_buf) + i;
	while (i < sizeof(T) * N)
	{
		*(T*)(dst) = static_cast<T>(value);
		dst += step;
		i += step;
	}
}

template<typename T, typename U>
__device__ void update_max(T& value, const U next_value)
{
	if (value < next_value)
		value = static_cast<T>(next_value);
}

constexpr int ScratchpadL1Mask64 = RANDOMX_SCRATCHPAD_L1 - CacheLineSize;
constexpr int ScratchpadL2Mask64 = RANDOMX_SCRATCHPAD_L2 - CacheLineSize;

#define DST_OFFSET			0
#define SRC_OFFSET			3
#define IMM_OFFSET			6
#define LOC_OFFSET			14
#define SHIFT_OFFSET		15
#define SRC_IS_IMM32_OFFSET	17
#define SRC_IS_IMM64_OFFSET	18
#define NEGATIVE_SRC_OFFSET	19
#define OPCODE_OFFSET		20
#define NUM_INSTS_OFFSET	24
#define NUM_FP_INSTS_OFFSET	28

#define INST_NOP			(8 << OPCODE_OFFSET)

// Group header word: the first word of every group, read by all participating
// lanes for num_workers/num_fp. Bits 27 and 31 are otherwise unused
// (num_insts sits at 24-26, num_fp at 28-30). M2c: bit 27 = group contains a
// live CBRANCH, bit 31 = group contains a CFROUND. Lets execution_end resolve
// ip/fprc with predicated imm_buf reads only -- no per-word flag scan.
#define GROUP_CB_BIT		(1u << 27)
#define GROUP_CF_BIT		(1u << 31)

template<size_t N> struct get_power_of_2 { enum { Value = get_power_of_2<N / 2>::Value + 1 }; };
template<> struct get_power_of_2<1> { enum { Value = 0 }; };

#define LOC_L1 (32 - get_power_of_2<RANDOMX_SCRATCHPAD_L1>::Value)
#define LOC_L2 (32 - get_power_of_2<RANDOMX_SCRATCHPAD_L2>::Value)
#define LOC_L3 (32 - get_power_of_2<RANDOMX_SCRATCHPAD_L3>::Value)

__device__ uint64_t imul_rcp_value(uint32_t divisor)
{
	if ((divisor & (divisor - 1)) == 0)
	{
		return 1ULL;
	}

	const uint64_t p2exp63 = 1ULL << 63;

	uint64_t quotient = p2exp63 / divisor;
	uint64_t remainder = p2exp63 % divisor;

	uint32_t bsr = 31 - __clz(divisor);

	for (uint32_t shift = 0; shift <= bsr; ++shift)
	{
		const bool b = (remainder >= divisor - remainder);
		quotient = (quotient << 1) | (b ? 1 : 0);
		remainder = (remainder << 1) - (b ? divisor : 0);
	}

	return quotient;
}

template<int WORKERS_PER_HASH>
__global__ void __launch_bounds__(RX_WAVE_SIZE, RX_WAVE_SIZE == 64 ? 8 : 16) init_vm(void* entropy_data, void* vm_states)
{
#if RANDOMX_PROGRAM_SIZE <= 256
	typedef uint8_t exec_t;
#else
	typedef uint16_t exec_t;
#endif

	enum { INIT_IDX_WIDTH = (WORKERS_PER_HASH == 16) ? 16 : 8 };
	enum { INIT_HASHES_PER_BLOCK = RX_WAVE_SIZE / INIT_IDX_WIDTH };

	__shared__ uint32_t execution_plan_buf[RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH * INIT_HASHES_PER_BLOCK * sizeof(exec_t) / sizeof(uint32_t)];

	set_buffer(execution_plan_buf, 0);

	__syncthreads();

	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx = global_index / INIT_IDX_WIDTH;
	const uint32_t sub = global_index % INIT_IDX_WIDTH;

	exec_t* execution_plan = (exec_t*)(execution_plan_buf + (threadIdx.x / INIT_IDX_WIDTH) * RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH * sizeof(exec_t) / sizeof(uint32_t));

	uint64_t* R = ((uint64_t*) vm_states) + idx * VM_STATE_SIZE / sizeof(uint64_t);
	R[sub] = 0;

	const uint64_t* entropy = ((const uint64_t*) entropy_data) + idx * ENTROPY_SIZE / sizeof(uint64_t);

	double* A = (double*)(R + 24);
	A[sub] = getSmallPositiveFloatBits(entropy[sub]);

	if (sub == 0)
	{
		uint2* src_program = (uint2*)(entropy + 128 / sizeof(uint64_t));

#if RANDOMX_PROGRAM_SIZE <= 256
		uint64_t registerLastChanged = 0;
		uint64_t registerWasChanged = 0;
#else
		int32_t registerLastChanged[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
#endif

		for (uint32_t i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		{
			*(uint32_t*)(src_program + i) &= ~(0xF8U << 8);

			const uint2 src_inst = src_program[i];
			uint2 inst = src_inst;

			uint32_t opcode = inst.x & 0xff;
			const uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;

			if (opcode < RANDOMX_FREQ_IADD_RS + RANDOMX_FREQ_IADD_M + RANDOMX_FREQ_ISUB_R + RANDOMX_FREQ_ISUB_M + RANDOMX_FREQ_IMUL_R + RANDOMX_FREQ_IMUL_M + RANDOMX_FREQ_IMULH_R + RANDOMX_FREQ_IMULH_M + RANDOMX_FREQ_ISMULH_R + RANDOMX_FREQ_ISMULH_M)
			{
#if RANDOMX_PROGRAM_SIZE <= 256
				set_byte(registerLastChanged, dst, i);
				set_byte(registerWasChanged, dst, 1);
#else
				registerLastChanged[dst] = i;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_RS + RANDOMX_FREQ_IADD_M + RANDOMX_FREQ_ISUB_R + RANDOMX_FREQ_ISUB_M + RANDOMX_FREQ_IMUL_R + RANDOMX_FREQ_IMUL_M + RANDOMX_FREQ_IMULH_R + RANDOMX_FREQ_IMULH_M + RANDOMX_FREQ_ISMULH_R + RANDOMX_FREQ_ISMULH_M;

			if (opcode < RANDOMX_FREQ_IMUL_RCP)
			{
				if (inst.y & (inst.y - 1))
				{
#if RANDOMX_PROGRAM_SIZE <= 256
					set_byte(registerLastChanged, dst, i);
					set_byte(registerWasChanged, dst, 1);
#else
					registerLastChanged[dst] = i;
#endif
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_RCP;

			if (opcode < RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R + RANDOMX_FREQ_IXOR_M + RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
			{
#if RANDOMX_PROGRAM_SIZE <= 256
				set_byte(registerLastChanged, dst, i);
				set_byte(registerWasChanged, dst, 1);
#else
				registerLastChanged[dst] = i;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R + RANDOMX_FREQ_IXOR_M + RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

			if (opcode < RANDOMX_FREQ_ISWAP_R)
			{
				if (src != dst)
				{
#if RANDOMX_PROGRAM_SIZE <= 256
					set_byte(registerLastChanged, dst, i);
					set_byte(registerWasChanged, dst, 1);
					set_byte(registerLastChanged, src, i);
					set_byte(registerWasChanged, src, 1);
#else
					registerLastChanged[dst] = i;
					registerLastChanged[src] = i;
#endif
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_ISWAP_R;

			if (opcode < RANDOMX_FREQ_FSWAP_R + RANDOMX_FREQ_FADD_R + RANDOMX_FREQ_FADD_M + RANDOMX_FREQ_FSUB_R + RANDOMX_FREQ_FSUB_M + RANDOMX_FREQ_FSCAL_R + RANDOMX_FREQ_FMUL_R + RANDOMX_FREQ_FDIV_M + RANDOMX_FREQ_FSQRT_R)
			{
				*(uint32_t*)(src_program + i) |= 0x20 << 8;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSWAP_R + RANDOMX_FREQ_FADD_R + RANDOMX_FREQ_FADD_M + RANDOMX_FREQ_FSUB_R + RANDOMX_FREQ_FSUB_M + RANDOMX_FREQ_FSCAL_R + RANDOMX_FREQ_FMUL_R + RANDOMX_FREQ_FDIV_M + RANDOMX_FREQ_FSQRT_R;

			if (opcode < RANDOMX_FREQ_CBRANCH)
			{
				const uint32_t creg = dst;
#if RANDOMX_PROGRAM_SIZE <= 256
				const uint32_t change = get_byte(registerLastChanged, dst);
				const int32_t lastChanged = (get_byte(registerWasChanged, dst) == 0) ? -1 : static_cast<int32_t>(change);

				*(uint32_t*)(src_program + i) = (src_inst.x & 0xFF0000FFU) | ((creg | ((lastChanged == -1) ? 0x90 : 0x10)) << 8) | ((static_cast<uint32_t>(lastChanged) & 0xFF) << 16);
#else
				const int32_t lastChanged = registerLastChanged[dst];

				*(uint32_t*)(src_program + i) = (src_inst.x & 0xFF0000FFU) | ((creg | 0x10) << 8);
#endif

				*(uint32_t*)(src_program + lastChanged + 1) |= 0x40 << 8;

#if RANDOMX_PROGRAM_SIZE <= 256
				uint32_t tmp = i | (i << 8);
				registerLastChanged = tmp | (tmp << 16);
				registerLastChanged = registerLastChanged | (registerLastChanged << 32);

				registerWasChanged = 0x0101010101010101ULL;
#else
				registerLastChanged[0] = i;
				registerLastChanged[1] = i;
				registerLastChanged[2] = i;
				registerLastChanged[3] = i;
				registerLastChanged[4] = i;
				registerLastChanged[5] = i;
				registerLastChanged[6] = i;
				registerLastChanged[7] = i;
#endif
			}
		}

		uint64_t registerLatency = 0;
		uint64_t registerReadCycle = 0;
		uint64_t registerLatencyFP = 0;
		uint64_t registerReadCycleFP = 0;
		uint32_t ScratchpadHighLatency = 0;
		uint32_t ScratchpadLatency = 0;

		int32_t first_available_slot = 0;
		int32_t first_allowed_slot_cfround = 0;
		int32_t last_used_slot = -1;
		int32_t last_memory_op_slot = -1;
		int32_t last_memory_store_slot = -1;

		uint32_t num_slots_used = 0;
		uint32_t num_instructions = 0;

		int32_t first_instruction_slot = -1;
		bool first_instruction_fp = false;

		bool update_branch_target_mark = false;
		bool first_available_slot_is_branch_target = false;
		for (uint32_t i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		{
			const uint2 inst = src_program[i];

			uint32_t opcode = inst.x & 0xff;
			uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;
			const uint32_t mod = (inst.x >> 24);

			bool is_branch_target = (inst.x & (0x40 << 8)) != 0;
			if (is_branch_target)
			{
				first_available_slot = last_used_slot + 1;
				first_available_slot_is_branch_target = true;
			}

			const uint32_t dst_latency = get_byte(registerLatency, dst);
			const uint32_t src_latency = get_byte(registerLatency, src);
			const uint32_t reg_read_latency = (dst_latency > src_latency) ? dst_latency : src_latency;
			const uint32_t mem_read_latency = ((dst == src) && ((inst.y & ScratchpadL3Mask64) >= RANDOMX_SCRATCHPAD_L2)) ? ScratchpadHighLatency : ScratchpadLatency;

			uint32_t full_read_latency = mem_read_latency;
			update_max(full_read_latency, reg_read_latency);

			uint32_t latency = 0;
			bool is_memory_op = false;
			bool is_memory_store = false;
			bool is_nop = false;
			bool is_branch = false;
			bool is_swap = false;
			bool is_src_read = true;
			bool is_fp = false;
			bool is_cfround = false;

			do {
				if (opcode < RANDOMX_FREQ_IADD_RS)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IADD_RS;

				if (opcode < RANDOMX_FREQ_IADD_M)
				{
					latency = full_read_latency;
					// A scratchpad READ must not be grouped with an earlier ISTORE:
					// group execution performs all reads before any write, while the
					// CPU executes the store first. Reads only need separation from
					// the last store (read-read pairs are safe); the ISTORE branch's
					// own constraint keeps later stores out of this read's group.
					update_max(latency, (last_memory_store_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IADD_M;

				if (opcode < RANDOMX_FREQ_ISUB_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_ISUB_R;

				if (opcode < RANDOMX_FREQ_ISUB_M)
				{
					latency = full_read_latency;
					update_max(latency, (last_memory_store_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISUB_M;

				if (opcode < RANDOMX_FREQ_IMUL_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IMUL_R;

				if (opcode < RANDOMX_FREQ_IMUL_M)
				{
					latency = full_read_latency;
					update_max(latency, (last_memory_store_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IMUL_M;

				if (opcode < RANDOMX_FREQ_IMULH_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IMULH_R;

				if (opcode < RANDOMX_FREQ_IMULH_M)
				{
					latency = full_read_latency;
					update_max(latency, (last_memory_store_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IMULH_M;

				if (opcode < RANDOMX_FREQ_ISMULH_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_ISMULH_R;

				if (opcode < RANDOMX_FREQ_ISMULH_M)
				{
					latency = full_read_latency;
					update_max(latency, (last_memory_store_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISMULH_M;

				if (opcode < RANDOMX_FREQ_IMUL_RCP)
				{
					is_src_read = false;
					if (inst.y & (inst.y - 1))
						latency = dst_latency;
					else
						is_nop = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IMUL_RCP;

				if (opcode < RANDOMX_FREQ_INEG_R)
				{
					is_src_read = false;
					latency = dst_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_INEG_R;

				if (opcode < RANDOMX_FREQ_IXOR_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IXOR_R;

				if (opcode < RANDOMX_FREQ_IXOR_M)
				{
					latency = full_read_latency;
					update_max(latency, (last_memory_store_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IXOR_M;

				if (opcode < RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

				if (opcode < RANDOMX_FREQ_ISWAP_R)
				{
					is_swap = true;
					if (dst != src)
						latency = reg_read_latency;
					else
						is_nop = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISWAP_R;

				if (opcode < RANDOMX_FREQ_FSWAP_R)
				{
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSWAP_R;

				if (opcode < RANDOMX_FREQ_FADD_R)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FADD_R;

				if (opcode < RANDOMX_FREQ_FADD_M)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					update_max(latency, src_latency);
					update_max(latency, ScratchpadLatency);
					update_max(latency, (last_memory_store_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_fp = true;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_FADD_M;

				if (opcode < RANDOMX_FREQ_FSUB_R)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSUB_R;

				if (opcode < RANDOMX_FREQ_FSUB_M)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					update_max(latency, src_latency);
					update_max(latency, ScratchpadLatency);
					update_max(latency, (last_memory_store_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_fp = true;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_FSUB_M;

				if (opcode < RANDOMX_FREQ_FSCAL_R)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSCAL_R;

				if (opcode < RANDOMX_FREQ_FMUL_R)
				{
					dst = (dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FMUL_R;

				if (opcode < RANDOMX_FREQ_FDIV_M)
				{
					dst = (dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					update_max(latency, src_latency);
					update_max(latency, ScratchpadLatency);
					update_max(latency, (last_memory_store_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_fp = true;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_FDIV_M;

				if (opcode < RANDOMX_FREQ_FSQRT_R)
				{
					dst = (dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSQRT_R;

				if (opcode < RANDOMX_FREQ_CBRANCH)
				{
					is_src_read = false;
					is_branch = true;
					latency = dst_latency;

					first_available_slot = last_used_slot + 1;
					break;
				}
				opcode -= RANDOMX_FREQ_CBRANCH;

				if (opcode < RANDOMX_FREQ_CFROUND)
				{
					latency = src_latency;
					is_cfround = true;
					break;
				}
				opcode -= RANDOMX_FREQ_CFROUND;

				if (opcode < RANDOMX_FREQ_ISTORE)
				{
					latency = reg_read_latency;
					update_max(latency, (last_memory_op_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_memory_op = true;
					is_memory_store = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISTORE;

				is_nop = true;
			} while (false);

			if (is_nop)
			{
				if (is_branch_target)
					update_branch_target_mark = true;
				continue;
			}

			if (update_branch_target_mark)
			{
				*(uint32_t*)(src_program + i) |= 0x40 << 8;
				update_branch_target_mark = false;
				is_branch_target = true;
			}

			int32_t first_allowed_slot = first_available_slot;
			update_max(first_allowed_slot, latency * WORKERS_PER_HASH);
			if (is_cfround)
				update_max(first_allowed_slot, first_allowed_slot_cfround);
			else
				update_max(first_allowed_slot, get_byte(is_fp ? registerReadCycleFP : registerReadCycle, dst) * WORKERS_PER_HASH);

			if (is_swap)
				update_max(first_allowed_slot, get_byte(registerReadCycle, src) * WORKERS_PER_HASH);

			int32_t slot_to_use = last_used_slot + 1;
			update_max(slot_to_use, first_allowed_slot);

			if (is_fp)
			{
				slot_to_use = -1;
				for (int32_t j = first_allowed_slot; slot_to_use < 0; ++j)
				{
					if ((execution_plan[j] == 0) && (execution_plan[j + 1] == 0) && ((j + 1) % WORKERS_PER_HASH))
					{
						bool blocked = false;
						for (int32_t k = (j / WORKERS_PER_HASH) * WORKERS_PER_HASH; k < j; ++k)
						{
							if (execution_plan[k] || (k == first_instruction_slot))
							{
								const uint32_t inst = src_program[execution_plan[k]].x;

								if (((inst & (0x20 << 8)) == 0) && (((inst & (0x50 << 8)) != 0) || is_branch_target))
								{
									blocked = true;
									continue;
								}
							}
						}

						if (!blocked)
						{
							for (int32_t k = (j / WORKERS_PER_HASH) * WORKERS_PER_HASH; k < j; ++k)
							{
								if (execution_plan[k] || (k == first_instruction_slot))
								{
									const uint32_t inst = src_program[execution_plan[k]].x;
									if ((inst & (0x20 << 8)) == 0)
									{
										execution_plan[j] = execution_plan[k];
										execution_plan[j + 1] = execution_plan[k + 1];
										if (first_instruction_slot == k) first_instruction_slot = j;
										if (first_instruction_slot == k + 1) first_instruction_slot = j + 1;
										slot_to_use = k;
										break;
									}
								}
							}

							if (slot_to_use < 0)
								slot_to_use = j;

							break;
						}
					}
				}
			}
			else
			{
				for (int32_t j = first_allowed_slot; j <= last_used_slot; ++j)
				{
					if (execution_plan[j] == 0)
					{
						slot_to_use = j;
						break;
					}
				}
			}

			if (i == 0)
			{
				first_instruction_slot = slot_to_use;
				first_instruction_fp = is_fp;
			}

			if (is_cfround)
				first_allowed_slot_cfround = slot_to_use - (slot_to_use % WORKERS_PER_HASH) + WORKERS_PER_HASH;

			++num_instructions;

			execution_plan[slot_to_use] = i;
			++num_slots_used;

			if (is_fp)
			{
				execution_plan[slot_to_use + 1] = i;
				++num_slots_used;
			}

			const uint32_t next_latency = (slot_to_use / WORKERS_PER_HASH) + 1;

			if (is_src_read)
			{
				int32_t value = get_byte(registerReadCycle, src);
				update_max(value, slot_to_use / WORKERS_PER_HASH);
				set_byte(registerReadCycle, src, value);
			}

			if (is_memory_op)
				update_max(last_memory_op_slot, slot_to_use);

			if (is_memory_store)
				update_max(last_memory_store_slot, slot_to_use);

			if (is_cfround)
			{
				const uint32_t t = next_latency | (next_latency << 8);
				registerLatencyFP = t | (t << 16);
				registerLatencyFP = registerLatencyFP | (registerLatencyFP << 32);
			}
			else if (is_fp)
			{
				set_byte(registerLatencyFP, dst, next_latency);

				int32_t value = get_byte(registerReadCycleFP, dst);
				update_max(value, slot_to_use / WORKERS_PER_HASH);
				set_byte(registerReadCycleFP, dst, value);
			}
			else
			{
				if (!is_memory_store && !is_nop)
				{
					set_byte(registerLatency, dst, next_latency);
					if (is_swap)
						set_byte(registerLatency, src, next_latency);

					int32_t value = get_byte(registerReadCycle, dst);
					update_max(value, slot_to_use / WORKERS_PER_HASH);
					set_byte(registerReadCycle, dst, value);
				}

				if (is_branch)
				{
					const uint32_t t = next_latency | (next_latency << 8);
					registerLatency = t | (t << 16);
					registerLatency = registerLatency | (registerLatency << 32);
				}

				if (is_memory_store)
				{
					int32_t value = get_byte(registerReadCycle, dst);
					update_max(value, slot_to_use / WORKERS_PER_HASH);
					set_byte(registerReadCycle, dst, value);
					ScratchpadLatency = slot_to_use / WORKERS_PER_HASH;
					if ((mod >> 4) >= randomx::StoreL3Condition)
						ScratchpadHighLatency = slot_to_use / WORKERS_PER_HASH;
				}
			}

			if (execution_plan[first_available_slot] || (first_available_slot == first_instruction_slot))
			{
				if (first_available_slot_is_branch_target)
				{
					src_program[i].x |= 0x40 << 8;
					first_available_slot_is_branch_target = false;
				}

				if (is_fp)
					++first_available_slot;

				do {
					++first_available_slot;
				} while ((first_available_slot < RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH) && (execution_plan[first_available_slot] != 0));
			}

			if (is_branch_target)
				update_max(first_available_slot, is_fp ? (slot_to_use + 2) : (slot_to_use + 1));

			update_max(last_used_slot, is_fp ? (slot_to_use + 1) : slot_to_use);
			while (execution_plan[last_used_slot] || (last_used_slot == first_instruction_slot) || ((last_used_slot == first_instruction_slot + 1) && first_instruction_fp))
				++last_used_slot;
			--last_used_slot;

			if (is_fp && (last_used_slot >= first_allowed_slot_cfround))
				first_allowed_slot_cfround = last_used_slot + 1;
		}

		uint32_t ma = static_cast<uint32_t>(entropy[8]) & CacheLineAlignMask;
		uint32_t mx = static_cast<uint32_t>(entropy[10]) & CacheLineAlignMask;

		uint32_t addressRegisters = static_cast<uint32_t>(entropy[12]);
		addressRegisters = ((addressRegisters & 1) | (((addressRegisters & 2) ? 3U : 2U) << 8) | (((addressRegisters & 4) ? 5U : 4U) << 16) | (((addressRegisters & 8) ? 7U : 6U) << 24)) * sizeof(uint64_t);

		uint32_t datasetOffset = (entropy[13] & randomx::DatasetExtraItems) * randomx::CacheLineSize;

		ulonglong2 eMask = *(ulonglong2*)(entropy + 14);
		eMask.x = getFloatMask(eMask.x);
		eMask.y = getFloatMask(eMask.y);

		((uint32_t*)(R + 16))[0] = ma;
		((uint32_t*)(R + 16))[1] = mx;
		((uint32_t*)(R + 16))[2] = addressRegisters;
		((uint32_t*)(R + 16))[3] = datasetOffset;
		((ulonglong2*)(R + 18))[0] = eMask;

		uint32_t* imm_buf = (uint32_t*)(R + REGISTERS_SIZE / sizeof(uint64_t));
		uint32_t imm_index = 0;
		int32_t imm_index_fscal_r = -1;
		uint32_t* compiled_program = (uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t));

		// Group control flags: clear before the emit loop ORs bits into them.
#ifndef RX_LEGACY_DISPATCH
		uint32_t* group_flags = (uint32_t*)((uint8_t*)R + RX_GROUP_FLAGS_OFFSET);
		#pragma unroll
		for (uint32_t gi = 0; gi < RX_GROUP_FLAGS_SIZE / 4; ++gi)
			group_flags[gi] = 0;
#endif

	int32_t branch_target_slot = -1;
	int32_t k = -1;

#ifndef RX_LEGACY_DISPATCH
	// Compiled plan v2 emitter: write one program word and derive its 4 flag
	// bits from the word itself (opcode/loc/dst/src fields). Centralizing the
	// flag logic keeps every opcode class consistent. INST_NOP decodes as
	// opcode 8 (degenerate ISWAP on r0); a live ISWAP_R additionally has
	// dst != src, which distinguishes it from a NOP word. Group-level cb/cf
	// header bits are folded in by the post-pass after the emit loop.
	auto rx_emit_word = [&](uint32_t word) {
		*(compiled_program++) = word;
		const uint32_t opc = (word >> OPCODE_OFFSET) & 15;
		uint32_t f = 0;
		if (opc == 9) f = 1;
		else if (opc == 13) f = 2;
		else if ((opc == 8) && (((word >> DST_OFFSET) ^ (word >> SRC_OFFSET)) & 7)) f = 4;
		if (word & (1u << LOC_OFFSET)) f |= 8;
		if (f) group_flags[k >> 3] |= f << ((k & 7) * 4);
	};
#endif
#ifdef RX_LEGACY_DISPATCH
	auto rx_emit_word = [&](uint32_t word) {
		*(compiled_program++) = word;
	};
#endif

	for (int32_t i = 0; i <= last_used_slot; ++i)
	{
		if (!(execution_plan[i] || (i == first_instruction_slot) || ((i == first_instruction_slot + 1) && first_instruction_fp)))
			continue;

			uint32_t num_workers = 1;
			uint32_t num_fp_insts = 0;
			while ((i + num_workers <= last_used_slot) && ((i + num_workers) % WORKERS_PER_HASH) && (execution_plan[i + num_workers] || (i + num_workers == first_instruction_slot) || ((i + num_workers == first_instruction_slot + 1) && first_instruction_fp)))
			{
				if ((num_workers & 1) && ((src_program[execution_plan[i + num_workers]].x & (0x20 << 8)) != 0))
					++num_fp_insts;
				++num_workers;
			}

			num_workers = ((num_workers - 1) << NUM_INSTS_OFFSET) | (num_fp_insts << NUM_FP_INSTS_OFFSET);

			const uint2 src_inst = src_program[execution_plan[i]];
			uint2 inst = src_inst;

			uint32_t opcode = inst.x & 0xff;
			const uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;
			const uint32_t mod = (inst.x >> 24);

			const bool is_fp = (src_inst.x & (0x20 << 8)) != 0;
			if (is_fp && ((i & 1) == 0))
				++i;

			const bool is_branch_target = (src_inst.x & (0x40 << 8)) != 0;
			if (is_branch_target && (branch_target_slot < 0))
				branch_target_slot = k;

			++k;

			inst.x = INST_NOP;

			if (opcode < RANDOMX_FREQ_IADD_RS)
			{
				const uint32_t shift = (mod >> 2) % 4;

				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (shift << SHIFT_OFFSET);

				if (dst != randomx::RegisterNeedsDisplacement)
				{
					inst.x |= (1 << OPCODE_OFFSET);
				}
				else
				{
					inst.x |= imm_index << IMM_OFFSET;
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_RS;

			if (opcode < RANDOMX_FREQ_IADD_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (1 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_M;

			if (opcode < RANDOMX_FREQ_ISUB_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_R;

			if (opcode < RANDOMX_FREQ_ISUB_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (1 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_M;

			if (opcode < RANDOMX_FREQ_IMUL_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (2 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_R;

			if (opcode < RANDOMX_FREQ_IMUL_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (2 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_M;

			if (opcode < RANDOMX_FREQ_IMULH_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (6 << OPCODE_OFFSET);

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_R;

			if (opcode < RANDOMX_FREQ_IMULH_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (6 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_M;

			if (opcode < RANDOMX_FREQ_ISMULH_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (4 << OPCODE_OFFSET);

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_R;

			if (opcode < RANDOMX_FREQ_ISMULH_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (4 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_M;

			if (opcode < RANDOMX_FREQ_IMUL_RCP)
			{
				const uint64_t r = imul_rcp_value(inst.y);
				if (r == 1)
				{
					rx_emit_word(INST_NOP | num_workers);
					continue;
				}

				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (2 << OPCODE_OFFSET);
				inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM64_OFFSET);

				if (imm_index < IMM_INDEX_COUNT - 1)
				{
					imm_buf[imm_index] = ((const uint32_t*)&r)[0];
					imm_buf[imm_index + 1] = ((const uint32_t*)&r)[1];
					imm_index += 2;
				}

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_RCP;

			if (opcode < RANDOMX_FREQ_INEG_R)
			{
				inst.x = (dst << DST_OFFSET) | (5 << OPCODE_OFFSET);

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_INEG_R;

			if (opcode < RANDOMX_FREQ_IXOR_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (3 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IXOR_R;

			if (opcode < RANDOMX_FREQ_IXOR_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (3 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IXOR_M;

			if (opcode < RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (7 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}
				if (opcode >= RANDOMX_FREQ_IROR_R)
					inst.x |= (1 << NEGATIVE_SRC_OFFSET);

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

			if (opcode < RANDOMX_FREQ_ISWAP_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (8 << OPCODE_OFFSET);

				rx_emit_word(((src != dst) ? inst.x : INST_NOP) | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_ISWAP_R;

			if (opcode < RANDOMX_FREQ_FSWAP_R)
			{
				inst.x = (dst << DST_OFFSET) | (11 << OPCODE_OFFSET);

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_FSWAP_R;

			if (opcode < RANDOMX_FREQ_FADD_R)
			{
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | ((src % randomx::RegisterCountFlt) << (SRC_OFFSET + 1)) | (12 << OPCODE_OFFSET);

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_FADD_R;

			if (opcode < RANDOMX_FREQ_FADD_M)
			{
				const uint32_t location = (mod % 4) ? 1 : 2;
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (12 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_FADD_M;

			if (opcode < RANDOMX_FREQ_FSUB_R)
			{
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | ((src % randomx::RegisterCountFlt) << (SRC_OFFSET + 1)) | (12 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_FSUB_R;

			if (opcode < RANDOMX_FREQ_FSUB_M)
			{
				const uint32_t location = (mod % 4) ? 1 : 2;
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (12 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_FSUB_M;

			if (opcode < RANDOMX_FREQ_FSCAL_R)
			{
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | (1 << SRC_IS_IMM64_OFFSET) | (3 << OPCODE_OFFSET);
				if (imm_index_fscal_r >= 0)
				{
					inst.x |= (imm_index_fscal_r << IMM_OFFSET);
				}
				else
				{
					imm_index_fscal_r = imm_index;
					inst.x |= (imm_index << IMM_OFFSET);

					if (imm_index < IMM_INDEX_COUNT - 1)
					{
						imm_buf[imm_index] = 0;
						imm_buf[imm_index + 1] = 0x80F00000UL;
						imm_index += 2;
					}
				}

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_FSCAL_R;

			if (opcode < RANDOMX_FREQ_FMUL_R)
			{
				inst.x = (((dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt) << DST_OFFSET) | ((src % randomx::RegisterCountFlt) << (SRC_OFFSET + 1)) | (1 << SHIFT_OFFSET) | (12 << OPCODE_OFFSET);

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_FMUL_R;

			if (opcode < RANDOMX_FREQ_FDIV_M)
			{
				const uint32_t location = (mod % 4) ? 1 : 2;
				inst.x = (((dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt) << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (15 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_FDIV_M;

			if (opcode < RANDOMX_FREQ_FSQRT_R)
			{
				inst.x = (((dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt) << DST_OFFSET) | (14 << OPCODE_OFFSET);

				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_FSQRT_R;

			if (opcode < RANDOMX_FREQ_CBRANCH)
			{
				inst.x = (dst << DST_OFFSET) | (9 << OPCODE_OFFSET);
				inst.x |= (imm_index << IMM_OFFSET);

				const uint32_t cshift = (mod >> 4) + randomx::ConditionOffset;

				uint32_t imm = inst.y | (1U << cshift);
				if (cshift > 0)
					imm &= ~(1U << (cshift - 1));

				if (imm_index < IMM_INDEX_COUNT - 1)
				{
					imm_buf[imm_index] = imm;
					imm_buf[imm_index + 1] = cshift | (static_cast<uint32_t>(branch_target_slot) << 5);
					imm_index += 2;
				}
				else
				{
					inst.x = INST_NOP;
				}

			branch_target_slot = -1;

			rx_emit_word(inst.x | num_workers);
			continue;
		}
		opcode -= RANDOMX_FREQ_CBRANCH;

		if (opcode < RANDOMX_FREQ_CFROUND)
		{
			inst.x = (src << SRC_OFFSET) | (13 << OPCODE_OFFSET) | ((inst.y & 63) << IMM_OFFSET);

			rx_emit_word(inst.x | num_workers);
			continue;
		}
		opcode -= RANDOMX_FREQ_CFROUND;

		if (opcode < RANDOMX_FREQ_ISTORE)
			{
				const uint32_t location = ((mod >> 4) >= randomx::StoreL3Condition) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (10 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;
				rx_emit_word(inst.x | num_workers);
				continue;
			}
			opcode -= RANDOMX_FREQ_ISTORE;

			rx_emit_word(inst.x | num_workers);
		}

#ifndef RX_LEGACY_DISPATCH
		// M2c post-pass: fold the per-word cb/cf flags into group header
		// words (GROUP_CB_BIT / GROUP_CF_BIT). A header is any word position
		// the executor can start a group from: the fall-through walk from 0
		// (ip += num_insts + 1) plus every branch landing
		// (branch_target_slot + 1, decoded from the CBRANCH immediates).
		// Walked with a fixed-point queue because branch-target groups may
		// themselves contain branches. The per-word flags were emitted with
		// correct positions, so this pass only needs to OR-reduce them over
		// each group span.
		{
			uint32_t* const prog_base = (uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t));
			uint64_t done[4] = { 0, 0, 0, 0 };
			int32_t queue[64];
			int32_t qn = 0;

			auto fold_header = [&](int32_t ip0) -> int32_t {
				done[ip0 >> 6] |= 1ull << (ip0 & 63);
				const uint32_t hdr = prog_base[ip0];
				const int32_t num_workers = (hdr >> NUM_INSTS_OFFSET) & (WORKERS_PER_HASH - 1);
				const int32_t num_fp = (hdr >> NUM_FP_INSTS_OFFSET) & (WORKERS_PER_HASH - 1);
				const int32_t num_insts = num_workers - num_fp;
				uint32_t bits = 0;
				for (int32_t w = ip0, w_end = ip0 + num_insts; (w <= w_end) && (w <= k); ++w)
				{
					const uint32_t f = (group_flags[w >> 3] >> ((w & 7) * 4)) & 3;
					if (f & 1) bits |= GROUP_CB_BIT;
					if (f & 2) bits |= GROUP_CF_BIT;
					if ((f & 1) && (qn < 64))
					{
						// Branch word: decode its target and queue it as a
						// header (executor lands at branch_target_slot + 1).
						const uint32_t ioff = (prog_base[w] >> IMM_OFFSET) & 255;
						const int32_t tgt = static_cast<int32_t>(imm_buf[ioff + 1] >> 5) + 1;
						if ((tgt >= 0) && (tgt <= k) && !(done[tgt >> 6] & (1ull << (tgt & 63))))
							queue[qn++] = tgt;
					}
				}
				prog_base[ip0] = hdr | bits;
				return num_insts + 1;
			};

			for (int32_t ip = 0; ip <= k; )
			{
				if (done[ip >> 6] & (1ull << (ip & 63))) break;
				ip += fold_header(ip);
			}
			for (int32_t qi = 0; qi < qn; ++qi)
				if (!(done[queue[qi] >> 6] & (1ull << (queue[qi] & 63))))
					fold_header(queue[qi]);
		}
#endif

		((uint32_t*)(R + 20))[0] = static_cast<uint32_t>(compiled_program - (uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t)));
	}
}

template<typename T, size_t N>
__device__ void load_buffer(T (&dst_buf)[N], const void* src_buf)
{
	uint32_t i = threadIdx.x * sizeof(T);
	const uint32_t step = blockDim.x * sizeof(T);
	const uint8_t* src = ((const uint8_t*) src_buf) + i;
	uint8_t* dst = ((uint8_t*) dst_buf) + i;
	while (i < sizeof(T) * N)
	{
		*(T*)(dst) = *(T*)(src);
		src += step;
		dst += step;
		i += step;
	}
}

template<typename T>
__device__ void load_buffer(T* dst_buf, size_t count, const void* src_buf)
{
	uint32_t i = threadIdx.x;
	const uint32_t step = blockDim.x;
	const uint8_t* src = ((const uint8_t*) src_buf) + i * sizeof(T);
	uint8_t* dst = ((uint8_t*) dst_buf) + i * sizeof(T);
	while (i < count)
	{
		*(T*)(dst) = *(T*)(src);
		src += step * sizeof(T);
		dst += step * sizeof(T);
		i += step;
	}
}

template<int> __device__ double fma_rnd(double a, double b, double c, uint32_t fprc);
template<int, bool> __device__ double div_rnd(double a, double b, uint32_t fprc);
template<int, bool> __device__ double sqrt_rnd(double x, uint32_t fprc);

// ===== Device Directed Rounding Helpers =====

__device__ __forceinline__ double hip_nextafter(double x, double y) {
    if (x != x || y != y) return x + y; // isnan
    if (x == y) return y;
    uint64_t ux = __double_as_longlong(x);
    if (x < y) {
        if (x >= 0.0) ux++; else ux--;
    } else {
        if (x >= 0.0) ux--; else ux++;
    }
    return __longlong_as_double(ux);
}

__device__ __forceinline__ void fma_error(double a, double b, double c, double& hi, double& lo) {
    hi = __fma_rn(a, b, c);
    if (b == 1.0) {
        // Addition (a + c), used by FADD_R/FADD_M/FSUB_R/FSUB_M. The naive
        // residual __fma_rn(a,1,c-hi) rounds (c-hi) and can flip the sign of the
        // true residual (a+c)-hi when the exact result is close to hi, so the
        // directed-rounding correction picks the wrong 1-ulp neighbour. TwoSum
        // yields s + lo = a + c exactly with s == hi, so lo is the exact residual.
        // (A multiply whose operand is exactly 1.0 also lands here and is exact.)
        double s = a + c;
        double v = s - a;
        lo = (a - (s - v)) + (c - v);
    } else {
        // Product path (FMUL uses c == 0), where __fma_rn(a,b,-hi) is exact.
        lo = __fma_rn(a, b, c - hi);
    }
}

__device__ __forceinline__ void div_error(double a, double b, double& hi, double& lo) {
    hi = a / b;
    lo = __fma_rn(-hi, b, a);
}

__device__ __forceinline__ void sqrt_error(double x, double& hi, double& lo) {
    hi = sqrt(x);
    if (hi == 0.0) { lo = 0.0; return; }
    lo = __fma_rn(-hi, hi, x);
}

__device__ __forceinline__ double hip_fma_ru(double a, double b, double c) {
    double hi, lo;
    fma_error(a, b, c, hi, lo);
    if (lo > 0.0) hi = hip_nextafter(hi, __builtin_inf());
    return hi;
}

__device__ __forceinline__ double hip_fma_rd(double a, double b, double c) {
    double hi, lo;
    fma_error(a, b, c, hi, lo);
    if (lo < 0.0) hi = hip_nextafter(hi, -__builtin_inf());
    return hi;
}

__device__ __forceinline__ double hip_fma_rz(double a, double b, double c) {
    double hi, lo;
    fma_error(a, b, c, hi, lo);
    if ((lo > 0.0 && hi < 0.0) || (lo < 0.0 && hi > 0.0)) hi = hip_nextafter(hi, (hi > 0.0) ? -__builtin_inf() : __builtin_inf());
    return hi;
}

// Correctly-rounded directed division using an EFT residual.
// Mode convention matches the CPU (RandomX RoundMode): 0=RN, 1=RD, 2=RU, 3=RZ.
// hi = RN(a/b); error_sign > 0 <=> hi < exact value; error_sign < 0 <=> hi > exact.
__device__ __forceinline__ double rx_ddiv(double a, double b, int mode) {
    double hi, lo;
    div_error(a, b, hi, lo);
    int error_sign = 0;
    if (lo > 0.0) error_sign = (b > 0.0) ? 1 : -1;
    else if (lo < 0.0) error_sign = (b > 0.0) ? -1 : 1;
    if (mode == 0) return hi;
    if (mode == 1) { if (error_sign < 0) hi = hip_nextafter(hi, -__builtin_inf()); }
    else if (mode == 2) { if (error_sign > 0) hi = hip_nextafter(hi, __builtin_inf()); }
    else if (mode == 3) { if (error_sign < 0) hi = hip_nextafter(hi, (hi > 0.0) ? -__builtin_inf() : __builtin_inf()); }
    return hi;
}

// Correctly-rounded directed sqrt (sqrt >= 0, so RZ == RD).
__device__ __forceinline__ double rx_dsqrt(double x, int mode) {
    double hi, lo;
    sqrt_error(x, hi, lo);
    if (mode == 0) return hi;
    if (mode == 1) { if (lo < 0.0) hi = hip_nextafter(hi, -__builtin_inf()); }
    else if (mode == 2) { if (lo > 0.0) hi = hip_nextafter(hi, __builtin_inf()); }
    else if (mode == 3) { if (lo < 0.0 && hi > 0.0) hi = hip_nextafter(hi, -__builtin_inf()); }
    return hi;
}

template<> __device__ double fma_rnd<-1>(double a, double b, double c, uint32_t fprc)
{
	// fprc convention matches the CPU (RandomX RoundMode): 0=RN, 1=RD, 2=RU, 3=RZ
	if (fprc == 0)
		return __fma_rn(a, b, c);
	else if (fprc == 1)
		return hip_fma_rd(a, b, c);
	else if (fprc == 2)
		return hip_fma_ru(a, b, c);
	else
		return hip_fma_rz(a, b, c);
}

template<> __device__ double div_rnd<-1, true>(double a, double b, uint32_t fprc)
{
	if (fprc == 0)
		return rx_ddiv(a, b, 0);
	else if (fprc == 1)
		return rx_ddiv(a, b, 1);
	else if (fprc == 2)
		return rx_ddiv(a, b, 2);
	else
		return rx_ddiv(a, b, 3);
}

template<> __device__ double sqrt_rnd<-1, true>(double a, uint32_t fprc)
{
	if (fprc == 0)
		return rx_dsqrt(a, 0);
	else if (fprc == 1)
		return rx_dsqrt(a, 1);
	else if (fprc == 2)
		return rx_dsqrt(a, 2);
	else
		return rx_dsqrt(a, 3);
}

template<> __device__ double div_rnd<-1, false>(double a, double b, uint32_t fprc)
{
	// Correctly-rounded directed division (bit-exact vs CPU fesetround division).
	return rx_ddiv(a, b, fprc);
}

template<> __device__ double sqrt_rnd<-1, false>(double a, uint32_t fprc)
{
	// Correctly-rounded directed sqrt (bit-exact vs CPU fesetround sqrt).
	return rx_dsqrt(a, fprc);
}

template<> __device__ double fma_rnd<0>(double a, double b, double c, uint32_t) { return __fma_rn(a, b, c); }
template<> __device__ double fma_rnd<1>(double a, double b, double c, uint32_t) { return hip_fma_rd(a, b, c); }
template<> __device__ double fma_rnd<2>(double a, double b, double c, uint32_t) { return hip_fma_ru(a, b, c); }
template<> __device__ double fma_rnd<3>(double a, double b, double c, uint32_t) { return hip_fma_rz(a, b, c); }

template<> __device__ double div_rnd<0, false>(double a, double b, uint32_t) { return rx_ddiv(a, b, 0); }
template<> __device__ double div_rnd<0, true>(double a, double b, uint32_t) { return rx_ddiv(a, b, 0); }
template<> __device__ double div_rnd<1, true>(double a, double b, uint32_t) { return rx_ddiv(a, b, 1); }
template<> __device__ double div_rnd<2, true>(double a, double b, uint32_t) { return rx_ddiv(a, b, 2); }
template<> __device__ double div_rnd<3, true>(double a, double b, uint32_t) { return rx_ddiv(a, b, 3); }

template<> __device__ double sqrt_rnd<0, false>(double a, uint32_t) { return rx_dsqrt(a, 0); }
template<> __device__ double sqrt_rnd<0, true>(double a, uint32_t) { return rx_dsqrt(a, 0); }
template<> __device__ double sqrt_rnd<1, true>(double a, uint32_t) { return rx_dsqrt(a, 1); }
template<> __device__ double sqrt_rnd<2, true>(double a, uint32_t) { return rx_dsqrt(a, 2); }
template<> __device__ double sqrt_rnd<3, true>(double a, uint32_t) { return rx_dsqrt(a, 3); }

#ifndef RX_TRACE_LO
#define RX_TRACE_LO 0
#endif
#ifndef RX_TRACE_HI
#define RX_TRACE_HI 40
#endif

#define ROUNDING_MODE (RANDOMX_FREQ_CFROUND ? -1 : 0)

#ifdef RX_M3_PROMOTE
__device__ __forceinline__ uint64_t rx_shfl_u64(uint64_t val, int srcLane)
{
	uint32_t lo = __shfl_sync(0xFFFFFFFFull, (uint32_t)val, srcLane);
	uint32_t hi = __shfl_sync(0xFFFFFFFFull, (uint32_t)(val >> 32), srcLane);
	return ((uint64_t)hi << 32) | lo;
}
#endif

// Wave-wide memory convergence for AMD GCN/RDNA. Waits until every outstanding
// vector/LDS memory operation of the whole wave (all lanes) has completed, so
// values stored by one lane become visible to loads issued by the other lanes
// afterwards. This replaces CUDA's `bar.warp.sync` for cross-lane handoff of
// the instruction pointer and the floating-point rounding mode.
__device__ __forceinline__ void rx_wave_sync()
{
#if defined(__HIP_DEVICE_COMPILE__)
	asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
#endif
}

__device__ __forceinline__ void rx_wave_sync_lds_only()
{
#if defined(__HIP_DEVICE_COMPILE__)
	asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#endif
}

#if (defined(RX_TRACE_VM) || defined(RX_TRACE_GROUPS)) && defined(__HIP_DEVICE_COMPILE__)
#define TRACE_RX_SYNC_BEFORE_TRACE() rx_wave_sync()
__device__ __forceinline__ void rx_trace_group(int32_t ic, int32_t ip, uint32_t fprc, const uint64_t* R)
{
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		printf("[GRP] ic=%d ip=%d fprc=%u %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx\n",
			(int)ic, (int)ip, (unsigned)fprc,
			(unsigned long long)R[0], (unsigned long long)R[1], (unsigned long long)R[2], (unsigned long long)R[3],
			(unsigned long long)R[4], (unsigned long long)R[5], (unsigned long long)R[6], (unsigned long long)R[7]);
		printf("[GRF] ic=%d ip=%d %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx\n",
			(int)ic, (int)ip,
			(unsigned long long)R[8], (unsigned long long)R[9], (unsigned long long)R[10], (unsigned long long)R[11],
			(unsigned long long)R[12], (unsigned long long)R[13], (unsigned long long)R[14], (unsigned long long)R[15],
			(unsigned long long)R[16], (unsigned long long)R[17], (unsigned long long)R[18], (unsigned long long)R[19],
			(unsigned long long)R[20], (unsigned long long)R[21], (unsigned long long)R[22], (unsigned long long)R[23]);
	}
}
#define RX_TRACE_GROUP(ic, ip, fprc, R) rx_trace_group(ic, ip, fprc, R)
__device__ __forceinline__ void rx_trace_post(uint32_t iter, int32_t ic, uint32_t fprc, uint32_t sp0, uint32_t sp1, const uint64_t* R)
{
	if (blockIdx.x == 0 && threadIdx.x == 0 && iter >= RX_TRACE_LO && iter < RX_TRACE_HI)
	{
		printf("[POST] ic=%d %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx sp0=%08x sp1=%08x fprc=%u\n",
			(int)ic,
			(unsigned long long)R[0], (unsigned long long)R[1], (unsigned long long)R[2], (unsigned long long)R[3],
			(unsigned long long)R[4], (unsigned long long)R[5], (unsigned long long)R[6], (unsigned long long)R[7],
			(unsigned long long)R[8], (unsigned long long)R[9], (unsigned long long)R[10], (unsigned long long)R[11],
			(unsigned long long)R[16], (unsigned long long)R[17], (unsigned long long)R[18], (unsigned long long)R[19],
			sp0, sp1, (unsigned)fprc);
	}
}
#define RX_TRACE_POST(iter, ic, fprc, sp0, sp1, R) rx_trace_post(iter, ic, fprc, sp0, sp1, R)
#else
#define TRACE_RX_SYNC_BEFORE_TRACE() do {} while (0)
#define RX_TRACE_GROUP(ic, ip, fprc, R) do {} while (0)
#define RX_TRACE_POST(iter, ic, fprc, sp0, sp1, R) do {} while (0)
#endif

template<int WORKERS_PER_HASH, bool HIGH_PRECISION>
__device__ void inner_loop(
	const uint32_t program_length,
	const uint32_t* compiled_program,
	const int32_t sub,
	uint8_t* scratchpad,
	const uint32_t fp_reg_offset,
	const uint32_t fp_reg_group_A_offset,
	uint64_t* R,
	uint32_t* imm_buf,
	const uint32_t batch_size,
	uint32_t& fprc,
	const uint64_t xexponentMask,
	const uint64_t workers_mask,
	const int32_t trace_ic
#ifdef RX_M3_PROMOTE
	,uint64_t& v_r
#ifdef RX_M3_PROMOTE_FE
	,uint64_t v_fe[4]
#endif
#endif
)
{
	const int32_t sub2 = sub >> 1;
	const bool trc = (trace_ic == RX_TRACE_LO);                              // full group walk
	const bool trc_entry = (trace_ic >= RX_TRACE_LO) && (trace_ic < RX_TRACE_HI); // START state only
#ifdef RX_LEGACY_DISPATCH
	imm_buf[IMM_INDEX_COUNT + 1] = fprc;
#endif
	if (trc_entry) { rx_wave_sync(); RX_TRACE_GROUP(trace_ic, 0, fprc, R); }

	#pragma unroll(1)
	for (int32_t ip = 0; ip < program_length;)
	{
#ifdef RX_LEGACY_DISPATCH
		imm_buf[IMM_INDEX_COUNT] = ip;
#endif

		uint32_t inst = compiled_program[ip];
		const int32_t num_workers = (inst >> NUM_INSTS_OFFSET) & (WORKERS_PER_HASH - 1);
		const int32_t num_fp_insts = (inst >> NUM_FP_INSTS_OFFSET) & (WORKERS_PER_HASH - 1);
		const int32_t num_insts = num_workers - num_fp_insts;
		const uint32_t group_bits = inst & (GROUP_CB_BIT | GROUP_CF_BIT);

		// Per-lane scratch participation: only lanes whose instruction touched
		// the scratchpad (M-variants, ISTORE) need the full vmcnt drain at the
		// slot boundary; their stores must retire before any lane's loads in
		// later groups. Lanes without outstanding global stores only need LDS
		// convergence (lgkmcnt). Lockstep execution guarantees ordering: every
		// storing lane drains at THIS slot's end, so by the next slot all
		// cross-lane global writes are visible through the write-through L1.
		bool my_loc = false;

		if (sub <= num_workers)
		{
			const int32_t inst_offset = sub - num_fp_insts;
			const bool is_fp = inst_offset < num_fp_insts;
			inst = compiled_program[ip + (is_fp ? sub2 : inst_offset)];

			uint32_t opcode = (inst >> OPCODE_OFFSET) & 15;
			const uint32_t location = (inst >> LOC_OFFSET) & 1;
			my_loc = (location != 0);

			const uint32_t reg_size_shift = is_fp ? 4 : 3;
			const uint32_t reg_base_offset = is_fp ? fp_reg_offset : 0;
			const uint32_t reg_base_src_offset = is_fp ? fp_reg_group_A_offset : 0;

			uint32_t dst_offset = (inst >> DST_OFFSET) & 7;
			dst_offset = reg_base_offset + (dst_offset << reg_size_shift);

			uint32_t src_offset = (inst >> SRC_OFFSET) & 7;
			src_offset = (src_offset << 3) + (location ? 0 : reg_base_src_offset);

			// M2a: INST_NOP decodes as ISWAP_R on r0 with dst == src (an
			// identity self-swap) but still pays the full register-file LDS
			// round trip. Skip it. Live ISWAP_R words always have dst != src
			// (the emitter NOPs the degenerate case), so this test is exact.
			if ((opcode == 8) && (dst_offset == src_offset))
				goto execution_end;

#ifdef RX_M3_PROMOTE
			const bool dst_is_mine_int = (!is_fp) && (dst_offset == (uint32_t)(sub << 3));
#ifdef RX_M3_PROMOTE_FE
			const bool dst_is_mine_fp = is_fp && ((dst_offset >> 4) < 4);
			const uint32_t fe_idx = dst_offset >> 4;
#endif
			const int32_t hash_base = threadIdx.x & ~(WORKERS_PER_HASH - 1);
#endif

			uint64_t* dst_ptr = (uint64_t*)((uint8_t*)(R) + dst_offset);
			uint64_t* src_ptr = (uint64_t*)((uint8_t*)(R) + src_offset);

			const uint32_t imm_offset = (inst >> IMM_OFFSET) & 255;
			const uint32_t* imm_ptr = imm_buf + imm_offset;

#ifdef RX_M3_PROMOTE
			uint64_t dst = dst_is_mine_int ? v_r : 
#ifdef RX_M3_PROMOTE_FE
				(dst_is_mine_fp ? v_fe[fe_idx] : 
#endif
				*dst_ptr
#ifdef RX_M3_PROMOTE_FE
				)
#endif
				;
			uint64_t src;
			if (!is_fp && !location && !(inst & (1 << SRC_IS_IMM32_OFFSET)) && !(inst & (1 << SRC_IS_IMM64_OFFSET)))
			{
				const int32_t src_lane = hash_base + (int32_t)(src_offset >> 3);
				src = rx_shfl_u64(v_r, src_lane);
			}
			else
			{
				src = *src_ptr;
			}
#else
			uint64_t dst = *dst_ptr;
			uint64_t src = *src_ptr;
#endif
			uint2 imm;
			imm.x = imm_ptr[0];
			imm.y = imm_ptr[1];

			if (location)
			{
				uint32_t loc_shift = (imm.x >> 21) & 0x1F;
				const uint32_t mask = (0xFFFFFFFFU >> loc_shift) - 7;

				const bool is_read = (opcode != 10);
				uint32_t addr = is_read ? ((loc_shift == LOC_L3) ? 0 : static_cast<uint32_t>(src)) : static_cast<uint32_t>(dst);
				addr += static_cast<int32_t>(imm.x);
				addr &= mask;

				uint64_t* ptr = (uint64_t*)(scratchpad + addr);

				if (is_read)
				{
					src = *ptr;
				}
				else
				{
					*ptr = src;
					goto execution_end;
				}
			}

			{
				if (inst & (1 << SRC_IS_IMM32_OFFSET)) src = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(imm.x)));

				if (opcode <= 3)
				{
					if (inst & (1 << NEGATIVE_SRC_OFFSET)) src = static_cast<uint64_t>(-static_cast<int64_t>(src));
					if (opcode == 0) dst += static_cast<int32_t>(imm.x);
					const uint32_t shift = (inst >> SHIFT_OFFSET) & 3;
					if (opcode < 2) dst += src << shift;
					const uint64_t imm64 = *((uint64_t*) &imm);
					if (inst & (1 << SRC_IS_IMM64_OFFSET)) src = imm64;
					if (opcode == 2) dst *= src;
					if (opcode == 3) dst ^= src;
				}
				else if (opcode == 12)
				{
					if (location) src = bit_cast<uint64_t>(__int2double_rn(static_cast<int32_t>(src >> ((sub & 1) * 32))));
					if (inst & (1 << NEGATIVE_SRC_OFFSET)) src ^= 0x8000000000000000ULL;

					const bool is_mul = (inst & (1 << SHIFT_OFFSET)) != 0;
					const double a = __longlong_as_double(dst);
					const double b = __longlong_as_double(src);

					dst = bit_cast<uint64_t>(fma_rnd<ROUNDING_MODE>(a, is_mul ? b : 1.0, is_mul ? 0.0 : b, fprc));
				}
				else if (opcode == 9)
				{
					// CBRANCH: add immediate to dst, then jump if the condition bits are all zero.
					dst += static_cast<int32_t>(imm.x);
					// imm.y encodes (branch_target_slot << 5) | condition_shift.
#ifdef RX_LEGACY_DISPATCH
					if ((static_cast<uint32_t>(dst) & (randomx::ConditionMask << (imm.y & 31))) == 0)
					{
						imm_buf[IMM_INDEX_COUNT] = static_cast<uint32_t>((static_cast<int32_t>(imm.y) >> 5) - num_insts);
					}
#else
					// Fast dispatch: publish the dispatch-time branch outcome. It must be
					// captured here rather than recomputed in the epilogue, because `dst`
					// (and the register file generally) may be overwritten by other
					// instructions of this same group before the epilogue runs. Store the
					// target offset when taken, else the group's ip (fall-through).
					{
						uint32_t jmp = static_cast<uint32_t>(ip);
						if ((static_cast<uint32_t>(dst) & (randomx::ConditionMask << (imm.y & 31))) == 0)
							jmp = static_cast<uint32_t>((static_cast<int32_t>(imm.y) >> 5) - num_insts);
						imm_buf[IMM_INDEX_COUNT] = jmp;
					}
#endif
				}
				else if (opcode == 7)
				{
					// IROR_R, IROL_R
					const uint32_t shift1 = src & 63;
#if RANDOMX_FREQ_IROL_R > 0
					const uint32_t shift2 = (64 - shift1) & 63;
					const bool is_rol = (inst & (1 << NEGATIVE_SRC_OFFSET));
					dst = (dst >> (is_rol ? shift2 : shift1)) | (dst << (is_rol ? shift1 : shift2));
#else
					dst = (dst >> shift1) | (dst << ((64 - shift1) & 63));
#endif
				}
				else if (opcode == 14)
				{
					// FSQRT_R
					dst = bit_cast<uint64_t>(sqrt_rnd<ROUNDING_MODE, HIGH_PRECISION>(__longlong_as_double(dst), fprc));
				}
				else if (opcode == 6)
				{
					// IMULH_R, IMULH_M
					dst = __umul64hi(dst, src);
				}
				else if (opcode == 4)
				{
					// ISMULH_R, ISMULH_M
					dst = static_cast<uint64_t>(__mul64hi(static_cast<int64_t>(dst), static_cast<int64_t>(src)));
				}
				else if (opcode == 11)
				{
					// FSWAP_R
					dst = *(uint64_t*)((uint8_t*)(R) + (dst_offset ^ 8));
				}
				else if (opcode == 8)
				{
					// ISWAP_R — cross-lane swap must go through LDS even with
					// M3 promotion; shfl cannot write to another lane's VGPR.
					*src_ptr = dst;
					dst = src;
				}
				else if (opcode == 15)
				{
					// FDIV_M
					src = bit_cast<uint64_t>(__int2double_rn(static_cast<int32_t>(src >> ((sub & 1) * 32))));
					src &= randomx::dynamicMantissaMask;
					src |= xexponentMask;
					dst = bit_cast<uint64_t>(div_rnd<ROUNDING_MODE, HIGH_PRECISION>(__longlong_as_double(dst), __longlong_as_double(src), fprc));
				}
				else if (opcode == 5)
				{
					// INEG_R
					dst = static_cast<uint64_t>(-static_cast<int64_t>(dst));
				}
				// CFROUND check will be skipped and removed entirely by the compiler if ROUNDING_MODE >= 0
				else if (ROUNDING_MODE < 0)
				{
					// CFROUND: rotate src right by imm_offset, new rounding mode = lowest 2 bits.
					// dst is intentionally NOT written back.
					//
					// Store (in BOTH dispatch modes): `src` is the dispatch-time value; another
					// instruction of this same group may overwrite that register before the
					// epilogue, so recomputing the mode from R there is not equivalent.
					imm_buf[IMM_INDEX_COUNT + 1] = ((src >> imm_offset) | (src << ((64 - imm_offset) & 63))) & 3;
					goto execution_end;
				}

#ifdef RX_M3_PROMOTE
				if (dst_is_mine_int)
					v_r = dst;
#ifdef RX_M3_PROMOTE_FE
				else if (dst_is_mine_fp)
					v_fe[fe_idx] = dst;
#endif
				else
#endif
				*dst_ptr = dst;
			}
		}

		execution_end:
		{
#ifdef RX_M3_PROMOTE
			R[sub] = v_r;
#endif
			if (my_loc)
				rx_wave_sync();
			else
				rx_wave_sync_lds_only();
#ifdef RX_M3_PROMOTE
			v_r = R[sub];
#endif

#ifdef RX_LEGACY_DISPATCH
			// Synchronize the instruction pointer and the rounding mode across all
			// lanes of the hash: CBRANCH/CFROUND above may have updated them from
			// a single lane via imm_buf.
			ip = imm_buf[IMM_INDEX_COUNT];
			fprc = imm_buf[IMM_INDEX_COUNT + 1];

			ip += num_insts + 1;
#else
			// Fast dispatch: ip/fprc are register-resident. M2c: cb/cf
			// presence bits live in the group header word, which every lane
			// already loaded at group start -- zero flag reads, zero scan.
			// The two imm_buf reads below are uniform across the hash's lanes
			// (no divergence) and are skipped entirely for non-control
			// groups. The executing lane published the dispatch-time
			// branch/rounding results into imm_buf above; reading them here
			// is correct even though other instructions of this group have
			// since overwritten the register file.
			if (group_bits & GROUP_CF_BIT)
				fprc = imm_buf[IMM_INDEX_COUNT + 1];

			if (group_bits & GROUP_CB_BIT)
				ip = imm_buf[IMM_INDEX_COUNT] + num_insts + 1;
			else
				ip += num_insts + 1;
#endif

#ifdef RX_FAST_DBG
			if (trace_ic == 0 && blockIdx.x == 0 && threadIdx.x == 0)
#ifndef RX_LEGACY_DISPATCH
				printf("[F ip=%d fprc=%u hdr=%08x]\n", (int)ip, (unsigned)fprc, (unsigned)group_bits);
#else
				printf("[F ip=%d fprc=%u]\n", (int)ip, (unsigned)fprc);
#endif
#endif

			if (trc) RX_TRACE_GROUP(trace_ic, ip, fprc, R);
		}
	}
}

template<int WORKERS_PER_HASH, bool HIGH_PRECISION>
__device__ void execute_vm_impl(void* vm_states, void* rounding, void* scratchpads, const void* dataset_ptr, uint32_t batch_size, uint32_t num_iterations, bool first, bool last)
{
	enum { IDX_WIDTH = (WORKERS_PER_HASH == 16) ? 16 : 8 };
	enum { HASHES_PER_BLOCK = RX_WAVE_SIZE / IDX_WIDTH };

	// Stage the block's VM states in LDS. All cross-lane register traffic
	// (R/F/E and the imm_buf ip/fprc handoff) runs through LDS: on AMD the
	// cross-lane visibility of global-memory stores is unreliable even with
	// s_waitcnt, while LDS writes become visible to the whole wave once
	// lgkmcnt drains (see rx_wave_sync). The state is written back to global
	// at the end of every kernel call, so it still survives the bfactor
	// kernel splits.
	//
	// With RX_PROGRAM_IN_GLOBAL the compiled program (read-only during
	// execute_vm -- written once by init_vm) is not staged: it is fetched
	// from global memory where it stays L1-resident. That halves the LDS
	// footprint per hash, doubling resident blocks per CU on wave64
	// (8 blocks/CU on gfx906 instead of 4).
#ifdef RX_PROGRAM_IN_GLOBAL
	enum { STAGED_SIZE = REGISTERS_SIZE + IMM_BUF_SIZE };
#else
	enum { STAGED_SIZE = VM_STATE_SIZE };
#endif

	__shared__ uint64_t vm_states_local[HASHES_PER_BLOCK * STAGED_SIZE / sizeof(uint64_t)];

#ifdef RX_PROGRAM_IN_GLOBAL
	// Strided stage: copy only the first STAGED_SIZE bytes of each hash's
	// global VM state into packed LDS.
	{
		enum { SQ = STAGED_SIZE / sizeof(uint64_t) };
		const uint64_t* g = ((const uint64_t*) vm_states) + blockIdx.x * HASHES_PER_BLOCK * (VM_STATE_SIZE / sizeof(uint64_t));
		for (uint32_t h = 0; h < HASHES_PER_BLOCK; ++h)
			for (uint32_t e = threadIdx.x; e < SQ; e += blockDim.x)
				vm_states_local[h * SQ + e] = g[h * (VM_STATE_SIZE / sizeof(uint64_t)) + e];
	}
	__syncthreads();
#else
	load_buffer(vm_states_local, ((const uint64_t*) vm_states) + blockIdx.x * HASHES_PER_BLOCK * (VM_STATE_SIZE / sizeof(uint64_t)));
	__syncthreads();
#endif

	const int32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t idx = global_index / IDX_WIDTH;
	const int32_t sub = global_index % IDX_WIDTH;

	if (idx >= batch_size)
		return;

	uint64_t* R = vm_states_local + (threadIdx.x / IDX_WIDTH) * STAGED_SIZE / sizeof(uint64_t);
	double* F = (double*)(R + 8);
	double* E = (double*)(R + 16);

	uint32_t ma = ((uint32_t*)(R + 16))[0];
	uint32_t mx = ((uint32_t*)(R + 16))[1];
	const uint32_t addressRegisters = ((uint32_t*)(R + 16))[2];
	const uint64_t* readReg0 = (uint64_t*)(((uint8_t*) R) + (addressRegisters & 0xff));
	const uint64_t* readReg1 = (uint64_t*)(((uint8_t*) R) + ((addressRegisters >> 8) & 0xff));
	const uint32_t* readReg2 = (uint32_t*)(((uint8_t*) R) + ((addressRegisters >> 16) & 0xff));
	const uint32_t* readReg3 = (uint32_t*)(((uint8_t*) R) + (addressRegisters >> 24));

	const uint32_t datasetOffset = ((uint32_t*)(R + 16))[3];
	const uint8_t* dataset = ((const uint8_t*) dataset_ptr) + datasetOffset;

	const uint32_t fp_reg_offset = 64 + ((global_index & 1) << 3);
	const uint32_t fp_reg_group_A_offset = 192 + ((global_index & 1) << 3);

	ulonglong2 eMask = ((ulonglong2*)(R + 18))[0];

	const uint32_t program_length = ((uint32_t*)(R + 20))[0];
	uint32_t fprc = ((uint32_t*) rounding)[idx];

	// CPU execute() starts each call with spAddr = {mx, ma}. Across bfactor
	// splits only the very first launch starts from {mx, ma}; every following
	// launch continues from 0 because the previous launch ended with both
	// scratchpad addresses reset to 0.
	uint32_t spAddr0 = first ? mx : 0;
	uint32_t spAddr1 = first ? ma : 0;

	uint8_t* scratchpad = ((uint8_t*) scratchpads) + idx * static_cast<uint64_t>(RANDOMX_SCRATCHPAD_L3 + 64);

	const bool f_group = (sub < 4);
	double* fe = f_group ? (F + sub * 2) : (E + (sub - 4) * 2);
	double* f = F + sub;
	double* e = E + sub;

	const uint64_t andMask = f_group ? uint64_t(-1) : randomx::dynamicMantissaMask;
	const uint64_t orMask1 = f_group ? 0 : eMask.x;
	const uint64_t orMask2 = f_group ? 0 : eMask.y;
	const uint64_t xexponentMask = (sub & 1) ? eMask.y : eMask.x;

	uint32_t* imm_buf = (uint32_t*)(R + REGISTERS_SIZE / sizeof(uint64_t));
#ifdef RX_PROGRAM_IN_GLOBAL
	// Read-only here (init_vm wrote it; the kernel boundary makes it
	// globally visible). L1-hot across all iterations of one program.
	const uint32_t* compiled_program = ((const uint32_t*) vm_states) + (uint64_t)idx * (VM_STATE_SIZE / sizeof(uint32_t)) + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint32_t);
#else
	const uint32_t* compiled_program = (const uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t));
#endif
#ifndef RX_LEGACY_DISPATCH
	// M2c: cb/cf presence moved into the group header word (GROUP_CB_BIT /
	// GROUP_CF_BIT); the per-word flag region is still emitted by init_vm for
	// the xlane/scratch milestones but no longer read here.
	(void) 0;
#endif

	const uint64_t workers_mask = ((1ull << WORKERS_PER_HASH) - 1) << ((threadIdx.x / IDX_WIDTH) * IDX_WIDTH);

	TRACE_VM_STATE("ENTER", 0, sub, R, fprc, ma, mx, spAddr0, spAddr1);

#ifdef RX_M3_PROMOTE
	uint64_t v_r = R[sub];
#ifdef RX_M3_PROMOTE_FE
	uint64_t v_fe[4];
	{
		const uint32_t fe_base = 64 + ((sub & 1) << 3);
		for (int k = 0; k < 4; k++)
			v_fe[k] = *(uint64_t*)((uint8_t*)R + fe_base + k * 16);
	}
#endif
#endif

	#pragma unroll(1)
	for (int ic = 0; ic < num_iterations; ++ic)
	{
		uint64_t *r, *p0, *p1;
		uint64_t dataset_line = 0;
		if ((WORKERS_PER_HASH <= 8) || (sub < 8))
		{
			dataset_line = *(const uint64_t*)(dataset + ma + sub * 8);

			rx_wave_sync();

			const uint64_t spMix = *readReg0 ^ *readReg1;
			spAddr0 ^= ((const uint32_t*)&spMix)[0];
			spAddr1 ^= ((const uint32_t*)&spMix)[1];
			spAddr0 &= ScratchpadL3Mask64;
			spAddr1 &= ScratchpadL3Mask64;

			TRACE_VM_STATE("SPADDR_UPDATED", ic, sub, R, fprc, ma, mx, spAddr0, spAddr1);

			p0 = (uint64_t*)(scratchpad + spAddr0 + sub * 8);
			p1 = (uint64_t*)(scratchpad + spAddr1 + sub * 8);

			r = R + sub;
#ifdef RX_M3_PROMOTE
			v_r ^= *p0;
#else
			*r ^= *p0;
#endif

			uint64_t global_mem_data = *p1;
			int32_t* q = (int32_t*)&global_mem_data;

			fe[0] = load_F_E_groups(q[0], andMask, orMask1);
			fe[1] = load_F_E_groups(q[1], andMask, orMask2);
		}

		TRACE_RX_SYNC_BEFORE_TRACE();
		TRACE_VM_STATE("BEFORE_INNER_LOOP", ic, sub, R, fprc, ma, mx, spAddr0, spAddr1);

		if ((WORKERS_PER_HASH == IDX_WIDTH) || (sub < WORKERS_PER_HASH))
			inner_loop<WORKERS_PER_HASH, HIGH_PRECISION>(program_length, compiled_program, sub, scratchpad, fp_reg_offset, fp_reg_group_A_offset, R, imm_buf, batch_size, fprc, xexponentMask, workers_mask			, ic
#ifdef RX_M3_PROMOTE
				, v_r
#ifdef RX_M3_PROMOTE_FE
				, v_fe
#endif
#endif
			);

		TRACE_VM_STATE("AFTER_INNER_LOOP", ic, sub, R, fprc, ma, mx, spAddr0, spAddr1);

		if ((WORKERS_PER_HASH <= 8) || (sub < 8))
		{
#ifdef RX_M3_PROMOTE
			R[sub] = v_r;
#ifdef RX_M3_PROMOTE_FE
			{
				const uint32_t fe_base = 64 + ((sub & 1) << 3);
				for (int k = 0; k < 4; k++)
					*(uint64_t*)((uint8_t*)R + fe_base + k * 16) = v_fe[k];
			}
#endif
#endif
			rx_wave_sync();

			mx ^= *readReg2 ^ *readReg3;
			mx &= CacheLineAlignMask;

#ifdef RX_M3_PROMOTE
			v_r ^= dataset_line;
			R[sub] = v_r;
#else
			const uint64_t next_r = *r ^ dataset_line;
			*r = next_r;
#endif

			*p1 = 
#ifdef RX_M3_PROMOTE
				v_r;
#else
				next_r;
#endif
			*p0 = bit_cast<uint64_t>(f[0]) ^ bit_cast<uint64_t>(e[0]);

			uint32_t tmp = ma;
			ma = mx;
			mx = tmp;

			spAddr0 = 0;
			spAddr1 = 0;

		TRACE_RX_SYNC_BEFORE_TRACE();
		TRACE_VM_STATE("DATASET_SWAP", ic, sub, R, fprc, ma, mx, spAddr0, spAddr1);
		RX_TRACE_POST(ic, ic, fprc, spAddr0, spAddr1, R);
		}
	}

	// Drain wave-wide LDS stores so the writeback reads the final register
	// file of all lanes (F[sub]/E[sub] may have been written by other lanes).
#ifdef RX_M3_PROMOTE
	R[sub] = v_r;
#endif
	rx_wave_sync();
#ifdef RX_M3_PROMOTE
	v_r = R[sub];
#endif

	if ((WORKERS_PER_HASH > 8) && (sub >= 8))
		return;

	uint64_t* p = ((uint64_t*) vm_states) + idx * (VM_STATE_SIZE / sizeof(uint64_t));
#ifdef RX_M3_PROMOTE
	p[sub] = v_r;
#else
	p[sub] = R[sub];
#endif

	if (sub == 0)
		((uint32_t*) rounding)[idx] = fprc;

	if (last)
	{
		p[sub +  8] = bit_cast<uint64_t>(F[sub]) ^ bit_cast<uint64_t>(E[sub]);
		p[sub + 16] = bit_cast<uint64_t>(E[sub]);
	}
	else if (sub == 0)
	{
		// Persist the control block for the next bfactor launch. With the LDS
		// staging only ma/mx can actually differ from the init_vm values kept
		// in global memory; the remaining fields are rewritten for parity.
		((uint32_t*)(p + 16))[0] = ma;
		((uint32_t*)(p + 16))[1] = mx;
		((uint32_t*)(p + 16))[2] = addressRegisters;
		((uint32_t*)(p + 16))[3] = datasetOffset;
		((ulonglong2*)(p + 18))[0] = eMask;
		((uint32_t*)(p + 20))[0] = program_length;
	}
}

template<int WORKERS_PER_HASH, bool HIGH_PRECISION>
__global__ void __launch_bounds__(RX_WAVE_SIZE, RX_VM_BLOCKS_PER_CU) execute_vm(void* vm_states, void* rounding, void* scratchpads, const void* dataset_ptr, uint32_t batch_size, uint32_t num_iterations, bool first, bool last)
{
	execute_vm_impl<WORKERS_PER_HASH, HIGH_PRECISION>(vm_states, rounding, scratchpads, dataset_ptr, batch_size, num_iterations, first, last);
}

template<int WORKERS_PER_HASH, bool HIGH_PRECISION>
__global__ void __launch_bounds__(RX_WAVE_SIZE, RX_VM_BLOCKS_PER_CU) execute_vm_dbg(void* vm_states, void* rounding, void* scratchpads, const void* dataset_ptr, uint32_t batch_size, uint32_t num_iterations, bool first, bool last, uint64_t* dbg, uint32_t* dbg_idx)
{
	execute_vm_impl<WORKERS_PER_HASH, HIGH_PRECISION>(vm_states, rounding, scratchpads, dataset_ptr, batch_size, num_iterations, first, last);

	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx = global_index / 8;
	const uint32_t sub = global_index % 8;

	if (idx >= batch_size)
		return;

	// Read final state from global memory (written by execute_vm_impl when last=true)
	uint64_t R[24];
	load_buffer(R, 24, ((const uint64_t*) vm_states) + idx * VM_STATE_SIZE / sizeof(uint64_t));

	uint32_t old_idx = atomicAdd(dbg_idx, 1);
	if (old_idx < 4096)
	{
		for (int i = 0; i < 8; ++i)
			dbg[old_idx * 24 + i] = R[i];
		for (int i = 0; i < 8; ++i)
			dbg[old_idx * 24 + 8 + i] = R[8 + i];
		for (int i = 0; i < 8; ++i)
			dbg[old_idx * 24 + 16 + i] = R[16 + i];
	}
}