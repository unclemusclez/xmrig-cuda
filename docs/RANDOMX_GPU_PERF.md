# RandomX GPU Performance Optimization — Design & Roadmap

Date: 2026-09-14
Scope: HIP backend RandomX kernels (`src/RandomX/`), targets gfx1100 (7900 XT, wave32) and gfx906 (MI50, wave64).

## 1. Current architecture

Pipeline per job, per hash in the batch (`src/RandomX/hash.hpp`):

```
blake2b_initial_hash
fillAes1Rx4                    -> scratchpad (2 MB per hash)
memset rounding/result
for p in 0..RANDOMX_PROGRAM_COUNT(8):
    fillAes4Rx4                -> entropy + program (from previous hash)
    init_vm<8>                 -> per-hash compiled program + VM state
    execute_vm<8,false> x 2^bfactor
    blake2b_hash_registers     -> chained hash feeds program p+1
```

Key constants (Monero `configuration.h`): PROGRAM_SIZE=256, PROGRAM_ITERATIONS=2048,
PROGRAM_COUNT=8, scratchpad L3=2 MB, dataset 2 GB + 32 MB extra.

`init_vm` is already a scheduler + compiler: it analyzes the random program,
dependency-schedules instructions into worker groups (WORKERS_PER_HASH=8),
folds NOPs/IMUL_RCP, and emits packed 32-bit instruction words + an immediate
table into the per-hash VM state (2560 B). `execute_vm` runs an interpreter
over those words; the integer register file, ip and fprc travel through LDS.

## 2. The "per-job GPU JIT" verdict: NOT feasible as originally framed

Original idea: at each new job, AOT-compile the program into straight-line
native code with hiprtc (the CN-R / KawPow generators prove the plumbing
works: `hiprtcCreateProgram` -> compile -> `hipModuleLoadDataEx`).

Blocker: **the program is not per-job, it is per-hash and chained.**
Every nonce in the batch has its own seed (previous hash), therefore its own
8 programs. A job of 8k nonces would need ~64k hiprtc compiles before any
hashing starts. Compiling per program-step per hash would serialize the GPU
for seconds per job. Dead end. Rejected.

What JIT-ification cannot remove here (per-hash randomness is the point of
RandomX): instruction decode of the packed words, register-file LDS traffic,
data-dependent scratchpad addressing.

## 3. Lever 1 — hot-loop dispatch and control-propagation reduction

Cost model of `inner_loop` (per slot): instruction word fetch (LDS) + decode
(bit extracts + 16-way opcode if-chain) + per-slot control overhead:

```
imm_buf[IMM_INDEX_COUNT] = ip          // LDS store, EVERY slot
... execute ...
rx_wave_sync()                          // s_waitcnt vmcnt/lgkm, EVERY slot
ip   = imm_buf[IMM_INDEX_COUNT]         // LDS load, EVERY slot
fprc = imm_buf[IMM_INDEX_COUNT + 1]     // LDS load, EVERY slot
```

Only CBRANCH (~10% of slots) and CFROUND (~0.4%) mutate ip/fprc cross-lane;
the other ~90% of slots pay the full LDS round-trip for nothing.

### Phase 1a — dataset line prefetch (DONE, 2026-09-14)
The end-of-iteration dataset read `dataset[ma]` is hoisted to the top of the
iteration so its ~300–500 ns global latency overlaps the inner loop instead
of stalling the wave after it (`execute_vm_impl`, `dataset_line`). Expected:
a few percent, grows with memory latency (MI50).

### Phase 1b — eliminate per-slot ip/fprc LDS round-trip
- Keep `ip`/`fprc` in registers on every lane.
- The executing lane of a CBRANCH/CFROUND publishes the new value via
  `__ballot(taken) & workers_mask` (already computed, currently unused) +
  exactly one `__shfl` from the owning lane; all other slots advance
  `ip += num_insts + 1` with **no LDS traffic and no extra waitcnt** beyond
  what register/scratchpad coherence requires.
- Fallback for architectures where cross-lane vote semantics are unsafe at
  32-thread-blocks-on-wave64 (gfx906 packs two blocks per wavefront): a
  compile-time switch keeps the current LDS protocol.
- Gate behind `RX_FAST_DISPATCH` (default ON on wave32 builds, opt-in on
  wave64 until validated).

Correctness gate: `tools/rx_selftest`, RX_TRACE_VM group traces vs CPU
reference, and at least one accepted pool share at diff >= 10k.

Expected: removes the dominant fixed per-slot overhead; est. 1.15–1.4x on
total hashrate (inner loop is the majority of kernel time; fillAes/blake2b
phases are unchanged).

### Phase 2 — selective megakernel specialization (only if 1b underdelivers)
Per-*job-structure* (not per-hash) template variants: number of slots,
branch count, and CFROUND presence are statistically stable; generate a small
set of precompiled template kernels and pick the closest at init. No runtime
compilation needed. Large effort, modest extra gain; do only with profiling
evidence.

## 4. Lever 3 — memory subsystem

- **1a (done)**: dataset prefetch.
- **Dataset nontemporal loads**: the 2 GB dataset stream cannot live in L2
  (MI50 4 MB / RDNA3 6 MB); marking dataset loads nontemporal (`slc` bit via
  inline asm or `__builtin_nontemporal_load`) stops them evicting scratchpad
  lines from L2. Scratchpad working set per CU block set fits partially in L2
  and benefits more. Test with rocprof L2-hit counters before/after.
- **Scratchpad access**: already 8-byte aligned 64-bit ops; address masks are
  precomputed in the imm table. No further structure to exploit without
  changing RandomX itself (impossible).
- **L1 policy**: keep scratchpad cacheable-L1 (L1 subsets are reused via the
  L1/L2/L3 loc selection).

## 5. Lever 2 — MI50 (gfx906) occupancy, separate track

Compile warnings show target-16 occupancy landing at 1–2 waves/EU on wave64
builds (32-thread blocks = half a wavefront; LDS 10 KB/block + VGPRs).
Options, in order of risk:
1. Raise effective waves by dropping per-block hash count (2 hashes/block,
   WORKERS_PER_HASH=16 template exists; validate scheduler assumptions on
   wave64 first).
2. Reduce VGPR pressure in `inner_loop` (live-range review of the decode
   temporaries; `#pragma unroll(1)` is already set).
3. Accept the ceiling: MI50 OpenCL (~1389 H/s) remains the port's parity
   target; wave64 kernel surgery beyond occupancy is not planned.

## 6. Validation & measurement protocol

1. `tools/rx_selftest` byte-exact vs CPU after every change.
2. RX_TRACE_VM / RX_TRACE_GROUPS traces for the first ~40 iterations must
   match the reference traces.
3. Mine against local p2pool; acceptance must stay ~100% over >= 2h and
   >= 1000 accepted shares (rejects = wrong hashes or stale; stales ruled
   out via job-id correlation).
4. Profile with rocprof (`SQ_WAVES`, `TCC_HIT/MISS`, `SQ_INSTS_VALU`,
   kernel time) to confirm each phase's mechanism before claiming gains.

## 7. Expectations (honest)

RandomX per-hash work is a crypto lower bound; no 10x exists on any GPU.
Stacked realistic ceiling of this roadmap: ~1.3–1.6x current hashrate on
RDNA3, MI50 parity with OpenCL (~1389 H/s). CPUs remain ~20–40x ahead by
design.
