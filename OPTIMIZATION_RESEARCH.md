# xmrig-cuda → HIP: Optimization Research Guide

This document is a structured set of research questions and direction notes for
optimizing our CUDA→HIP port of xmrig (RandomX / KawPow) on AMD RDNA 3 (RX 7900 XT).

Use it to drive Q-research / literature / spec reads. Each section lists:
- **Current approach** — what the code does today
- **Why it may be suboptimal** — the open question
- **Research questions** — concrete things to investigate
- **Optimization directions** — candidate changes to evaluate

Legend: VM = RandomX virtual machine emulated on GPU; LDS = local data share
(shared memory); wavefront = AMD's 32-lane execution unit (NVIDIA "warp").

---

## 0. Validation & Measurement (do this FIRST)

**Current**: We mine on the donation pool (diff 1000K). At that difficulty a valid
share needs `hash < target` with `target ≈ 1.84e13`; per-hash probability is
~1.6e-64, so we essentially never see a share. We cannot tell correct from
incorrect output. Build/launch crashes are visible; *wrong hashes are not*.

**Why it matters**: Every deeper optimization (kernel fusion, precision changes,
memory layout) risks silent correctness regressions. We are currently flying
blind.

**Research questions**:
- What is the minimal local setup to get *measurable* valid shares?
  (local `monerod` + `xmrig` with `daemon=true`, or a private pool with diff 1)
- Can we add a self-check mode: feed a known seed/nonce with a known-good
  RandomX hash (e.g. reference `randomx` CLI output) and assert equality on-device?
- How expensive is it to run a CPU RandomX hash for a few nonces per batch as a
  golden-reference compare?

**Optimization directions**:
- Add a `test_target` / `verify_every` config knob: override pool target with an
  easy target so shares are found and the full pipeline (blake2b → fill → init_vm
  → execute_vm → blake2b_hash_registers → share check) is exercised end-to-end.
- Add a `--self-test` path that hashes a fixed vector and compares against a
  baked-in expected digest.
- Wire up `hipEvent` timing around `hash()` to get a real H/s number instead of
  inferring from kernel-launch rate.

---

## 1. Kernel Launch Overhead & Fusion

**Current**: One `hash()` call for RandomX issues roughly:
`blake2b_initial_hash`, `fillAes1Rx4`, (×2 programs) `fillAes4Rx4`, `init_vm`,
`execute_vm` (single launch after bfactor collapse), `blake2b_hash_registers`,
+ `hashAes1Rx4` on the last program. (~9 launches + 1 D→H copy.)
Already done: collapsed `bfactor` loop (was 64 launches/program → 1) and folded
`find_shares` into `blake2b_hash_registers`.

**Why it may be suboptimal**: Each launch has fixed ~5–10 µs host/command-processor
overhead independent of work. RandomX is launch-dominated, not compute-dominated.

**Research questions**:
- For RDNA 3, what is the measured cost of a zero-arg kernel launch vs a launch
  with our grid/register footprint? (rocprof / `hipEvent` around empty launches)
- Does ROCm support **CUDA Graphs** (`hipGraph`) for this workload, and does it
  actually cut launch overhead when grids/params are static per batch?
- What is the cost/benefit of **cooperative groups** to replace the
  init→execute split?
- Is CPU-side per-batch submission itself the bottleneck (one `hash()` per batch
  serialized behind the previous `hipMemcpy`)?

**Optimization directions**:
- **Fuse `init_vm` + `execute_vm`**: `init_vm` writes the compiled program into
  global `vm_states`; `execute_vm` reloads it via `load_buffer` into LDS. If
  `init_vm` instead writes into LDS and `execute_vm` reads LDS directly (single
  kernel), we remove 1 launch/program + the reload. *High risk: `init_vm` is the
  ~1200-line program compiler — extract its `sub==0` scheduling body into a
  reusable `__device__ compile_vm_program(R, plan_buf, entropy)` so both the
  standalone and fused kernels share it. ONLY do this once §0 validation exists.*
- **Graph capture** of the static per-batch launch sequence.
- **Persistent kernels / stream-ordered launch** to overlap submission.

---

## 2. RandomX VM on SIMT Hardware

**Current**: `WORKERS_PER_HASH = 8`, `IDX_WIDTH = 8`, 4 groups of 8 lanes per
32-lane wavefront. Each lane is one "worker" (register/FP unit) of the RandomX VM.
The program is pre-scheduled (`execution_plan`) so integer + FP instructions
issue in parallel across the 8 workers.

**Why it may be suboptimal**: RandomX was designed for superscalar OoO CPUs. On a
GPU, the 8-worker packing assumes the program uses all 8 integer + 8 FP registers
uniformly. Real programs may under-utilize lanes → wasted occupancy.

**Research questions**:
- Is `WORKERS_PER_HASH = 8` optimal for RDNA 3, or does `= 4` (2 groups/wavefront,
  less LDS, higher occupancy) or `= 16` (1 group/wavefront, simpler indexing)
  benchmark better? What does the upstream CUDA build use and why?
- How much does the pre-scheduling (`init_vm`) cost vs. a JIT-on-GPU or a
  CPU-compiled program cache reused across batches?
- Can we specialize the kernel per-program (template on program shape) to let the
  compiler unroll the inner loop?

**Optimization directions**:
- Sweep `WORKERS_PER_HASH ∈ {4, 8, 16}` with §0 measurement.
- Cache compiled programs on the CPU (they depend only on entropy = block header
  program bytes) and upload the *compiled* plan, skipping `init_vm` entirely for
  repeated program shapes.

---

## 3. Memory Hierarchy (Registers / LDS / HBM)

**Current**:
- `execute_vm` uses `vm_states_local` in LDS: `(VM_STATE_SIZE * 4)/8` uint64 =
  10 KB/block (4 VM states for 4 groups). This caps blocks/CU from LDS alone.
- `init_vm` uses a second LDS buffer `execution_plan_buf` (~8–16 KB depending on
  `RANDOMX_PROGRAM_SIZE`).
- Scratchpad (`d_long_state`) and dataset (`d_rx_dataset`) live in HBM.

**Why it may be suboptimal**: 10 KB LDS/block on a CU with 64 KB LDS → ≤6 blocks/CU
from LDS; register pressure may drop this further (occupancy warnings already
show 3–8 waves vs 16 target). HBM bandwidth is the dominant cost for scratchpad
and dataset accesses (2048 KB scratchpad/hash, 2 MB dataset region/hash).

**Research questions**:
- What is the register count of `execute_vm` and `init_vm` at `-O3` for gfx1100,
  and the resulting max occupancy? (check `-Rpass=...` / `roc-obj-ltx` /
  `hipOccupancyMaxActiveBlocksPerMultiprocessor`)
- Can we shrink `vm_states_local` (e.g. keep only the live register file + FP
  regs in LDS, spill the imm_buf to a compact form) to raise blocks/CU?
- Is there a software-managed LDS **dataset cache** (keep the last N dataset
  cache-lines in LDS) that measurably helps, given RandomX's access pattern?

**Optimization directions**:
- Use `hipOccupancyMaxPotentialBlockSize` at runtime to pick `batch_size` blocks.
- Reduce LDS footprint; move `execution_plan` to registers where possible.
- Experiment with `-mllvm -amdgpu-lds-size` / bank-conflict-free indexing.

---

## 4. Wavefront Sizing & `__launch_bounds__`

**Current**: `execute_vm` and `init_vm` are `__launch_bounds__(32, 16)` (max 32
threads, min 16 warps/CU). Launch grid is `batch_size/4` blocks × 32 threads.
We fixed a prior crash where `vm_states_local` was sized for 2 states but 32
threads need 4 — now correctly 4.

**Why it may be suboptimal**: `min 16 warps` forces the compiler to keep register
usage very low, which can *increase* spills. The right bound depends on register
pressure, not a fixed number.

**Research questions**:
- What occupancy does ROCm actually achieve for `execute_vm` (rocprof
  `WavefrontsPerSimd` / `Occupancy`)? Is 16 waves/CU achievable or is it
  register-limited to 3–8 as the warnings suggest?
- Does dropping the `min` (or raising it) improve or hurt, given real register use?

**Optimization directions**:
- Replace the hard-coded `__launch_bounds__` with a measured value, or template
  the kernel on block-size and autotune once.
- Consider 1D launch of full wavefronts only (`blockDim.x` multiple of 32).

---

## 5. HIP / ROCm-Specific Tuning

**Current**: Ported CUDA APIs to HIP; `--offload-arch=gfx1100 -O3`; runtime libs
from ROCm 7.1/7.2. No ROCm-specific occupancy or graph APIs used yet.

**Research questions**:
- Should we call `hipFuncSetAttribute` for `hipFuncAttributeMaxDynamicSharedMemorySize`
  / preferred cache config (`hipDeviceSetCacheConfig(L1?)`)?
- Does `hipExtLaunchKernel` / multi-stream submission reduce tail latency?
- Are we paying for `hipDeviceSynchronize`'s host stall anywhere still? (Removed
  one already; audit `cuda_core.cu` for others — e.g. `hipModuleUnload` return
  ignored is a warning, not a stall.)

**Optimization directions**:
- `hipOccupancyMaxPotentialBlockSize` → runtime block/grid selection.
- Prefer L1 cache over shared for the dataset-read-heavy phases.
- Replace per-batch explicit syncs with stream-ordered copies.

---

## 6. Numerical Precision & Rounding Modes

**Current**: FP ops use directed rounding (`fma_rnd`, `div_rnd`, `sqrt_rnd` with
`fprc` = rounding mode per program). `RANDOMX_FREQ_CFROUND` switches modes mid-program
via `cfround`. This is required for RandomX correctness.

**Why it may be suboptimal**: Directed FP rounding is expensive on GPUs (often
serialized / emulated). The `HIGH_PRECISION` template path and `cfround` handling
may force slow paths.

**Research questions**:
- On RDNA 3, what is the cost of `fma_rnd`/`div_rnd`/`sqrt_rnd` vs native
  (`__fma_rn` etc.)? Is the directed-rounding emulation the dominant FP cost?
- Can we batch rounding-mode switches (group instructions by mode) to reduce
  mode-change frequency, or pre-expand the program into mode-homogeneous chunks?

**Optimization directions**:
- Measure FP rounding cost in isolation; consider a kernel variant that avoids
  mid-program `cfround` by splitting the program at mode boundaries.
- Only keep directed rounding where RandomX actually requires it.

---

## 7. Dataset Access Pattern

**Current**: `execute_vm` reads the RandomX dataset (`d_rx_dataset`, 2 GB) at
`datasetOffset + ma/mx` computed per iteration. Access is pseudo-random but
`CacheLineAlignMask`-constrained.

**Why it may be suboptimal**: 2 GB can't fit in L2; HBM random access is the
throughput ceiling for RandomX. Cache-line reuse across iterations/hashes may be
poor.

**Research questions**:
- What is the L2 hit rate for dataset accesses on gfx1100 (rocprof `L2CacheHit`)?
- Is there locality we can exploit with a small LDS/register dataset prefetch
  window (read ahead N cache-lines)?
- Does `hipDeviceSetCacheConfig(hipFuncCachePreferL1)` help the dataset-read
  phase specifically?

**Optimization directions**:
- Software prefetch of dataset cache-lines into LDS ring buffer.
- Stride/hint accesses to improve L2 coalescing where the algorithm permits.

---

## 8. CPU↔GPU Overlap & Argon2 Bottleneck

**Current**: Dataset is built on CPU (`randomx_dataset` init, 16 threads, ~2.3 s)
and uploaded once. Per batch, the CPU builds the *program* (entropy) — actually
NO: the program is derived from the block header and compiled on-GPU in `init_vm`.
But the CPU still does Argon2 / MSR and submits batches.

**Why it may be suboptimal**: With `<1%` CPU mining and a fast GPU, the CPU may
starve the GPU if batch submission isn't pipelined. (Note: MSR mod failed —
"HASHRATE WILL BE LOW" is about CPU only, not GPU.)

**Research questions**:
- Is the GPU ever idle waiting on host submission? (Trace host time vs
  `hipDeviceSynchronize` waits.)
- Can we double-buffer: build/submit batch N+1 on the CPU while batch N runs?

**Optimization directions**:
- Multi-stream / async memcpy + compute overlap.
- Pipeline batch preparation with kernel execution (persistent worker thread).

---

## 9. KawPow Status

**Current**: KawPow (ETH-style, DAG) converted (CUDA→HIPRTC) but **untested** on
this GPU (no active KawPow pools). Different memory profile (large DAG, lighter
VM).

**Research questions**:
- Does the same `bfactor`/fusion approach apply? KawPow has no RandomX-style
  program compiler, so `init_vm`-style fusion is N/A, but launch-count reduction
  still applies.
- Does the DAG fit in VRAM at current epoch, and what's the optimal DAG cache
  placement (L2 vs HBM)?

**Optimization directions**:
- Once testable, apply the same launch-collapse + share-fold patterns.
- Consider kernel fusion of DAG fetch + hash.

---

## 10. Multi-GPU

**Current**: Context-per-device model exists (`nvid_ctx`), but no automatic
work distribution across multiple GPUs in one process.

**Research questions**:
- Is the `xmrig` host layer already splitting work across GPUs, or is the DLL
  limited to one device?
- What's the scaling efficiency of adding a second GPU (PCIe vs bridge, memory
  per-device dataset duplication)?

**Optimization directions**:
- Expose per-device contexts to the host for N-way mining.
- NUMA/affinity pinning for multi-GPU AMD setups.

---

## Suggested Execution Order

1. **§0 Validation** — without this, nothing deeper is safe.
2. **§5 / §4** runtime occupancy + launch-bounds autotuning (low risk, measurable).
3. **§2 `WORKERS_PER_HASH` sweep** (low risk, high potential).
4. **§1 Graph capture / persistent kernels** (medium risk, cuts launch overhead).
5. **§1 `init_vm`+`execute_vm` fusion** — ONLY after §0 exists (high risk).
6. **§3 / §7 memory + dataset cache** (medium risk, needs profiling).
