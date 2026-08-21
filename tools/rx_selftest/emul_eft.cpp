// Host emulation of the GPU's execute_vm semantics.
//
// Runs the GPU-compiled program (from gpu_vm_post_init.bin = d_rx_vm_states item 0
// right after init_vm) with EXACTLY the semantics of inner_loop/execute_vm_impl in
// src/RandomX/randomx_cuda.hpp, on the CPU reference scratchpad and dataset.
//
// Compares the resulting register file against cpu_rf_p0.bin (tevador reference).
// If this MATCHES, the GPU kernel semantics + init_vm compilation are correct and
// any remaining mismatch is a GPU-hardware/intrinsic issue. If this DIFFERS, the
// divergence is in the compiled-program path and can be localized here.
//
// Usage: emul_vm.exe  (reads local .bin files from cwd)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfenv>
#include <vector>
#include <intrin.h>

static std::vector<uint8_t> rd(const char* n) {
    FILE* f = fopen(n, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", n); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(sz);
    if (fread(v.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", n); exit(1); }
    fclose(f);
    return v;
}

// ---- RandomX monero constants ----
static const uint64_t SCRATCHPAD_L3 = 2097152;
static const int      ScratchpadL3Mask64 = (int)(SCRATCHPAD_L3 - 64);
static const uint32_t CacheLineAlignMask = (2147483648u - 1) & ~63u;
static const uint64_t dynamicMantissaMask = (1ULL << 56) - 1;
static const uint32_t ConditionMask = 0xFF;
static const int DATASET_ITEM_SIZE = 64;

// instruction word field offsets (must match randomx_cuda.hpp)
static const int DST_OFFSET = 0, SRC_OFFSET = 3, IMM_OFFSET = 6, LOC_OFFSET = 14;
static const int SHIFT_OFFSET = 15, SRC_IS_IMM32_OFFSET = 17, SRC_IS_IMM64_OFFSET = 18;
static const int NEGATIVE_SRC_OFFSET = 19, OPCODE_OFFSET = 20;
static const int NUM_INSTS_OFFSET = 24, NUM_FP_INSTS_OFFSET = 28;
static const int LOC_L1 = 32 - 14, LOC_L2 = 32 - 18, LOC_L3 = 32 - 21; // scratchpad 16K, 256K, 2M
static const int IMM_INDEX_COUNT = (768 / 4) - 2; // IMM_BUF_SIZE=768

static inline double load_F_E_groups(int32_t value, uint64_t andMask, uint64_t orMask) {
    uint64_t x = (uint64_t)(int64_t)(int32_t)value;
    // numeric int->double conversion (bit pattern of the converted double)
    double d = (double)(int32_t)value;
    uint64_t bits; memcpy(&bits, &d, 8);
    bits &= andMask;
    bits |= orMask;
    memcpy(&d, &bits, 8);
    (void)x;
    return d;
}

static inline uint64_t f2b(double d) { uint64_t b; memcpy(&b, &d, 8); return b; }
static inline double b2f(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }

// ===== Host port of the GPU's EFT directed-rounding code (intrin_cuda.h /
// randomx_cuda.hpp). Must mirror the device code EXACTLY so this emulator
// reproduces what execute_vm computes on the GPU. =====

static inline double h_nextafter(double x, double y) {
    if (x != x || y != y) return x + y;
    if (x == y) return y;
    uint64_t ux = f2b(x);
    if (x < y) { if (x >= 0.0) ux++; else ux--; }
    else       { if (x >= 0.0) ux--; else ux++; }
    return b2f(ux);
}
static inline double h_fma_rn(double a, double b, double c) { return fma(a, b, c); }

struct EftRes { double hi, lo; };
static inline EftRes fma_error_h(double a, double b, double c) {
    double hi = h_fma_rn(a, b, c);
    double lo;
    if (b == 1.0) {
        // FADD: a + c. The naive residual fma(a,1,c-hi) loses the sign of the
        // true residual (a+c)-hi because c-hi is rounded. TwoSum(a,c) gives
        // s + e = a + c EXACTLY with s = RN(a+c) = hi, so e is the exact
        // residual and its sign is always correct.
        double s = a + c;
        double v = s - a;
        lo = (a - (s - v)) + (c - v);
        (void)s;
    } else {
        lo = h_fma_rn(a, b, c - hi);
    }
    return { hi, lo };
}
static inline EftRes div_error_h(double a, double b) {
    double hi = a / b;
    double lo = h_fma_rn(-hi, b, a);
    return { hi, lo };
}
static inline EftRes sqrt_error_h(double x) {
    double hi = sqrt(x);
    if (hi == 0.0) return { 0.0, 0.0 };
    double lo = h_fma_rn(-hi, hi, x);
    return { hi, lo };
}

// device hip_fma_rd/ru/rz
static inline double eft_fma(double a, double b, double c, int mode) {
    EftRes r = fma_error_h(a, b, c);
    double hi = r.hi, lo = r.lo;
    if (mode == 1) { if (lo < 0.0) hi = h_nextafter(hi, -HUGE_VAL); }
    else if (mode == 2) { if (lo > 0.0) hi = h_nextafter(hi, HUGE_VAL); }
    else if (mode == 3) { if ((lo > 0.0 && hi < 0.0)) hi = h_nextafter(hi, HUGE_VAL);
                          else if ((lo < 0.0 && hi > 0.0)) hi = h_nextafter(hi, -HUGE_VAL); }
    return hi;
}

// device rx_ddiv (mode: 0=RN 1=RD 2=RU 3=RZ)
static inline double eft_div(double a, double b, int mode) {
    EftRes r = div_error_h(a, b);
    double hi = r.hi, lo = r.lo;
    int error_sign = 0;
    if (lo > 0.0) error_sign = (b > 0.0) ? 1 : -1;
    else if (lo < 0.0) error_sign = (b > 0.0) ? -1 : 1;
    if (mode == 0) return hi;
    if (mode == 1) { if (error_sign < 0) hi = h_nextafter(hi, -HUGE_VAL); }
    else if (mode == 2) { if (error_sign > 0) hi = h_nextafter(hi, HUGE_VAL); }
    else if (mode == 3) { if (error_sign < 0) hi = h_nextafter(hi, (hi > 0.0) ? -HUGE_VAL : HUGE_VAL); }
    return hi;
}

// device rx_dsqrt
static inline double eft_sqrt(double x, int mode) {
    EftRes r = sqrt_error_h(x);
    double hi = r.hi, lo = r.lo;
    if (mode == 0) return hi;
    if (mode == 1) { if (lo < 0.0) hi = h_nextafter(hi, -HUGE_VAL); }
    else if (mode == 2) { if (lo > 0.0) hi = h_nextafter(hi, HUGE_VAL); }
    else if (mode == 3) { if (lo < 0.0 && hi > 0.0) hi = h_nextafter(hi, -HUGE_VAL); }
    return hi;
}

// CPU-exact directed FP ops via fesetround (ground truth for fma/div/sqrt).
static const int fes_modes[4] = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
static inline void set_rnd(uint32_t fprc) { fesetround(fes_modes[fprc & 3]); }

static uint64_t cnt_fma_diff = 0, cnt_div_diff = 0, cnt_sqrt_diff = 0;
static uint64_t cnt_fma = 0, cnt_div = 0, cnt_sqrt = 0;
static int dbg_fp_ops = 0; // set to ic==0 group tracing when needed

static inline uint64_t umulhi(uint64_t a, uint64_t b) {
    uint64_t hi; _umul128(a, b, &hi); return hi;
}
static inline uint64_t mulhi(int64_t a, int64_t b) {
    int64_t hi; _mul128(a, b, &hi); return (uint64_t)hi;
}

int main() {
    auto vmstate = rd("gpu_vmstate_p0.bin");     // 2048 bytes: registers+imm_buf+compiled program (post-init_vm, in hash()'s real flow)
    auto scratch = rd("gpu_scratch_p0.bin");     // 2 MB scratchpad item 0 (post-fillAes1Rx4, pre-execute)
    auto rfCPU   = rd("cpu_rf_p0.bin");          // 256 bytes: CPU reference register file after prog 0
    if (scratch.size() < SCRATCHPAD_L3) { fprintf(stderr, "scratch too small\n"); return 1; }

    // dataset: full 2GB+extra via randomx lib is heavy; instead read dataset slices on demand
    // from cpu_dataset only if small. We need the full dataset: load cpu_dataset_full.bin if present.
    std::vector<uint8_t> dataset;
    {
        FILE* f = fopen("cpu_dataset_full.bin", "rb");
        if (!f) { fprintf(stderr, "cpu_dataset_full.bin missing - dump it from the randomx dataset\n"); return 1; }
        _fseeki64(f, 0, SEEK_END); __int64 sz = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
        fprintf(stderr, "[emul] dataset file size %lld\n", sz);
        if (sz <= 0) { fprintf(stderr, "bad dataset size\n"); return 1; }
        dataset.resize((size_t)sz);
        if (fread(dataset.data(), 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "dataset short read\n"); return 1; }
        fclose(f);
        fprintf(stderr, "[emul] dataset loaded\n");
    }

    uint64_t* R = (uint64_t*)vmstate.data();
    uint32_t* imm_buf = (uint32_t*)(vmstate.data() + 256);
    uint32_t* compiled = (uint32_t*)(vmstate.data() + 256 + 768);

    // control block (matches execute_vm_impl reads)
    uint32_t ma = ((uint32_t*)(R + 16))[0];
    uint32_t mx = ((uint32_t*)(R + 16))[1];
    uint32_t addressRegisters = ((uint32_t*)(R + 16))[2];
    uint32_t datasetOffset = ((uint32_t*)(R + 16))[3];
    uint64_t eMask0 = R[18], eMask1 = R[19];
    uint32_t program_length = ((uint32_t*)(R + 20))[0];

    printf("init state: ma=%08x mx=%08x addrRegs=%08x dsOff=%08x proglen=%u\n", ma, mx, addressRegisters, datasetOffset, program_length);
    printf("eMask0=%016llx eMask1=%016llx\n", (unsigned long long)eMask0, (unsigned long long)eMask1);

    // readReg byte offsets within the register file
    uint32_t rr0 = addressRegisters & 0xff;
    uint32_t rr1 = (addressRegisters >> 8) & 0xff;
    uint32_t rr2 = (addressRegisters >> 16) & 0xff;
    uint32_t rr3 = (addressRegisters >> 24) & 0xff;

    double* F = (double*)(R + 8);
    double* E = (double*)(R + 16);

    uint32_t fprc = 0; // rounding buffer memset to 0
    uint32_t spAddr0 = mx; // first=true -> {mx, ma}
    uint32_t spAddr1 = ma;

    const int NUM_ITER = 2048;
    FILE* iterf = fopen("emul_iter_states.txt", "w");
    FILE* g0f = fopen("emu_g0_groups.txt", "w");
    for (int ic = 0; ic < NUM_ITER; ++ic) {
        // --- iteration prologue (scratchpad XOR + F/E load) ---
        uint64_t spMix = *(uint64_t*)((uint8_t*)R + rr0) ^ *(uint64_t*)((uint8_t*)R + rr1);
        spAddr0 ^= ((uint32_t*)&spMix)[0];
        spAddr1 ^= ((uint32_t*)&spMix)[1];
        spAddr0 &= ScratchpadL3Mask64;
        spAddr1 &= ScratchpadL3Mask64;
        for (int sub = 0; sub < 8; ++sub) {
            uint64_t* r = R + sub;
            *r ^= *(uint64_t*)(scratch.data() + spAddr0 + sub * 8);
            uint64_t gmd = *(uint64_t*)(scratch.data() + spAddr1 + sub * 8);
            int32_t q0 = (int32_t)(gmd & 0xFFFFFFFFu), q1 = (int32_t)(gmd >> 32);
            double* fe = (sub < 4) ? (F + sub * 2) : (E + (sub - 4) * 2);
            if (sub < 4) {
                fe[0] = load_F_E_groups(q0, ~0ULL, 0);
                fe[1] = load_F_E_groups(q1, ~0ULL, 0);
            } else {
                fe[0] = load_F_E_groups(q0, dynamicMantissaMask, eMask0);
                fe[1] = load_F_E_groups(q1, dynamicMantissaMask, eMask1);
            }
        }

        // dump iteration-start state (after prologue, before inner_loop)
        fprintf(iterf, "%d", ic);
        for (int k = 0; k < 8; ++k) fprintf(iterf, " %016llx", (unsigned long long)R[k]);
        for (int k = 0; k < 8; ++k) fprintf(iterf, " %016llx", (unsigned long long)f2b(F[k]));
        for (int k = 0; k < 8; ++k) fprintf(iterf, " %016llx", (unsigned long long)f2b(E[k]));
        fprintf(iterf, "\n");
        if (ic == 1) {
            // state at end of iteration 0 (after its dataset swap + ic=1 prologue)
            fprintf(g0f, "ENDITER0");
            for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)R[k]);
            for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)f2b(F[k]));
            for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)f2b(E[k]));
            fprintf(g0f, "\n");
            // continue through all iterations
        }
        if (ic == 0) {
            fprintf(stderr, "POST_PROLOGUE_R: ");
            for (int k = 0; k < 8; ++k) fprintf(stderr, "%016llx ", (unsigned long long)R[k]);
            fprintf(stderr, "\n");
        }

        // --- inner_loop emulation ---
        int32_t ip = 0;
        imm_buf[IMM_INDEX_COUNT + 1] = fprc;   // matches inner_loop entry
        while (ip < (int32_t)program_length) {
            bool is_last_group = false;
            {
                uint32_t h = compiled[ip];
                int32_t nw = (h >> NUM_INSTS_OFFSET) & 7;
                int32_t nf = (h >> NUM_FP_INSTS_OFFSET) & 7;
                int32_t ni = nw - nf;
                is_last_group = (ip + ni + 1 >= (int32_t)program_length);
            }
            if (ic == 0 && is_last_group) {
                fprintf(stderr, "EMU_LAST_GROUP ic=0 ip=%d R: ", ip);
                for (int k = 0; k < 8; ++k) fprintf(stderr, "%016llx ", (unsigned long long)R[k]);
                fprintf(stderr, "\n");
            }
            if (ic == 0) {
                fprintf(g0f, "%d", ip);
                for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)R[k]);
                for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)f2b(F[k]));
                for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)f2b(E[k]));
                fprintf(g0f, "\n");
            }
            imm_buf[IMM_INDEX_COUNT] = ip;
            uint32_t inst = compiled[ip];
            int32_t num_workers = (inst >> NUM_INSTS_OFFSET) & 7;
            int32_t num_fp_insts = (inst >> NUM_FP_INSTS_OFFSET) & 7;
            int32_t num_insts = num_workers - num_fp_insts;

            // two-phase group execution: all lanes read/compute first, writes
            // applied afterwards (GPU lanes read in lockstep before any write)
            struct PW { uint64_t* a; uint64_t v; };
            PW pw[24]; int npw = 0;
            PW spw[8]; int nspw = 0;

            for (int32_t sub = 0; sub <= num_workers; ++sub) {
                int32_t inst_offset = sub - num_fp_insts;
                bool is_fp = inst_offset < num_fp_insts;
                uint32_t w = compiled[ip + (is_fp ? (sub >> 1) : inst_offset)];
                uint32_t opcode = (w >> OPCODE_OFFSET) & 15;
                uint32_t location = (w >> LOC_OFFSET) & 1;
                uint32_t reg_size_shift = is_fp ? 4 : 3;
                uint32_t fp_reg_offset = 64 + ((sub & 1) << 3);
                uint32_t fp_reg_group_A_offset = 192 + ((sub & 1) << 3);
                uint32_t reg_base_offset = is_fp ? fp_reg_offset : 0;
                uint32_t reg_base_src_offset = is_fp ? fp_reg_group_A_offset : 0;
                uint32_t dst_offset = reg_base_offset + (((w >> DST_OFFSET) & 7) << reg_size_shift);
                uint32_t src_off = ((w >> SRC_OFFSET) & 7) << 3;
                uint32_t src_offset = src_off + (location ? 0 : reg_base_src_offset);
                uint64_t* dst_ptr = (uint64_t*)((uint8_t*)R + dst_offset);
                uint64_t* src_ptr = (uint64_t*)((uint8_t*)R + src_offset);
                uint32_t imm_offset = (w >> IMM_OFFSET) & 255;
                uint32_t* imm_ptr = imm_buf + imm_offset;
                uint64_t dst = *dst_ptr;
                uint64_t src = *src_ptr;
                uint32_t immx = imm_ptr[0], immy = imm_ptr[1];
                bool skip_dst_write = false;
                uint64_t dst_before = dst, src_before = src;
                bool trc = (ic == 0 && ip >= 10 && ip <= 14);

                if (location) {
                    uint32_t loc_shift = (immx >> 21) & 0x1F;
                    uint32_t mask = (0xFFFFFFFFu >> loc_shift) - 7;
                    bool is_read = (opcode != 10);
                    uint32_t addr = is_read ? ((loc_shift == LOC_L3) ? 0 : (uint32_t)src) : (uint32_t)dst;
                    addr += (int32_t)immx;
                    addr &= mask;
                    if (is_read) src = *(uint64_t*)(scratch.data() + addr);
                    else { spw[nspw].a = (uint64_t*)(scratch.data() + addr); spw[nspw].v = src; nspw++; skip_dst_write = true; }
                }

                if (!skip_dst_write) {
                    if (w & (1 << SRC_IS_IMM32_OFFSET)) src = (uint64_t)(int64_t)(int32_t)immx;
                    if (opcode <= 3) {
                        if (w & (1 << NEGATIVE_SRC_OFFSET)) src = (uint64_t)(-(int64_t)src);
                        if (opcode == 0) dst += (int32_t)immx;
                        uint32_t shift = (w >> SHIFT_OFFSET) & 3;
                        if (opcode < 2) dst += src << shift;
                        uint64_t imm64 = ((uint64_t)immy << 32) | immx;
                        if (w & (1 << SRC_IS_IMM64_OFFSET)) src = imm64;
                        if (opcode == 2) dst *= src;
                        if (opcode == 3) dst ^= src;
                    } else if (opcode == 12) {
                        if (location) src = f2b((double)(int32_t)(uint32_t)(src >> ((sub & 1) * 32)));
                        if (w & (1 << NEGATIVE_SRC_OFFSET)) src ^= 0x8000000000000000ULL;
                        bool is_mul = (w & (1 << SHIFT_OFFSET)) != 0;
                        double a = b2f(dst), b = b2f(src);
                        double ba = a, bb = is_mul ? b : 1.0, bc = is_mul ? 0.0 : b;
                        set_rnd(fprc);
                        double exact = fma(ba, bb, bc);
                        fesetround(FE_TONEAREST);
                        double eft = eft_fma(ba, bb, bc, fprc & 3);
                        cnt_fma++;
                        if (f2b(eft) != f2b(exact)) {
                            cnt_fma_diff++;
                            if (cnt_fma_diff <= 30)
                                fprintf(stderr, "FMA_DIFF ic=%d ip=%d sub=%d mode=%u a=%016llx b=%016llx c=%016llx eft=%016llx exact=%016llx\n",
                                        ic, ip, (int)sub, fprc & 3, f2b(ba), f2b(bb), f2b(bc), f2b(eft), f2b(exact));
                        }
                        dst = f2b(eft);
                    } else if (opcode == 9) {
                        dst += (int32_t)immx;
                        if (((uint32_t)dst & (ConditionMask << (immy & 31))) == 0)
                            imm_buf[IMM_INDEX_COUNT] = (uint32_t)((int32_t)immy >> 5) - num_insts;
                    } else if (opcode == 7) {
                        uint32_t shift1 = (uint32_t)(src & 63);
                        uint32_t shift2 = (64 - shift1) & 63;
                        bool is_rol = (w & (1 << NEGATIVE_SRC_OFFSET)) != 0; // RANDOMX_FREQ_IROL_R>0
                        dst = (dst >> (is_rol ? shift2 : shift1)) | (dst << (is_rol ? shift1 : shift2));
                    } else if (opcode == 14) {
                        double xa = b2f(dst);
                        set_rnd(fprc);
                        double exact = sqrt(xa);
                        fesetround(FE_TONEAREST);
                        double eft = eft_sqrt(xa, fprc & 3);
                        cnt_sqrt++;
                        if (f2b(eft) != f2b(exact)) {
                            cnt_sqrt_diff++;
                            if (cnt_sqrt_diff <= 30)
                                fprintf(stderr, "SQRT_DIFF ic=%d ip=%d sub=%d mode=%u a=%016llx eft=%016llx exact=%016llx\n",
                                        ic, ip, (int)sub, fprc & 3, f2b(xa), f2b(eft), f2b(exact));
                        }
                        dst = f2b(eft);
                    } else if (opcode == 6) {
                        dst = umulhi(dst, src);
                    } else if (opcode == 4) {
                        dst = mulhi((int64_t)dst, (int64_t)src);
                    } else if (opcode == 11) {
                        dst = *(uint64_t*)((uint8_t*)R + (dst_offset ^ 8));
                    } else if (opcode == 8) {
                        pw[npw].a = src_ptr; pw[npw].v = dst; npw++;
                        dst = src;
                    } else if (opcode == 15) {
                        src = f2b((double)(int32_t)(uint32_t)(src >> ((sub & 1) * 32)));
                        src &= dynamicMantissaMask;
                        src |= (sub & 1) ? eMask1 : eMask0;
                        double da = b2f(dst), db = b2f(src);
                        set_rnd(fprc);
                        double exact = da / db;
                        fesetround(FE_TONEAREST);
                        double eft = eft_div(da, db, fprc & 3);
                        cnt_div++;
                        if (f2b(eft) != f2b(exact)) {
                            cnt_div_diff++;
                            if (cnt_div_diff <= 30)
                                fprintf(stderr, "DIV_DIFF ic=%d ip=%d sub=%d mode=%u a=%016llx b=%016llx eft=%016llx exact=%016llx\n",
                                        ic, ip, (int)sub, fprc & 3, f2b(da), f2b(db), f2b(eft), f2b(exact));
                        }
                        dst = f2b(eft);
                    } else if (opcode == 5) {
                        dst = (uint64_t)(-(int64_t)dst);
                    } else if (opcode == 13) {
                        // CFROUND
                        imm_buf[IMM_INDEX_COUNT + 1] = ((src >> imm_offset) | (src << ((64 - imm_offset) & 63))) & 3;
                        skip_dst_write = true;
                    }
                }

                if (!skip_dst_write) { pw[npw].a = dst_ptr; pw[npw].v = dst; npw++; }
                if (trc)
                    fprintf(stderr, "GRP ip=%d sub=%d op=%u dso=%u sro=%u dst %016llx->%016llx src %016llx->%016llx skip=%d\n",
                            ip, (int)sub, opcode, dst_offset, src_offset,
                            dst_before, skip_dst_write ? 0ULL : dst, src_before, src, (int)skip_dst_write);
            }
            for (int i = 0; i < npw; ++i) *pw[i].a = pw[i].v;
            for (int i = 0; i < nspw; ++i) *spw[i].a = spw[i].v;

            // execution_end resync
            ip = (int32_t)imm_buf[IMM_INDEX_COUNT];
            fprc = imm_buf[IMM_INDEX_COUNT + 1];
            ip += num_insts + 1;
        }

        // --- dataset swap ---
        mx ^= *(uint32_t*)((uint8_t*)R + rr2) ^ *(uint32_t*)((uint8_t*)R + rr3);
        mx &= CacheLineAlignMask;
        for (int sub = 0; sub < 8; ++sub) {
            uint64_t* r = R + sub;
            uint64_t next_r = *r ^ *(uint64_t*)(dataset.data() + datasetOffset + ma + sub * 8);
            *r = next_r;
            *(uint64_t*)(scratch.data() + spAddr1 + sub * 8) = next_r;
            *(uint64_t*)(scratch.data() + spAddr0 + sub * 8) = f2b(F[sub]) ^ f2b(E[sub]);
        }
        uint32_t tmp = ma; ma = mx; mx = tmp;
        spAddr0 = 0; spAddr1 = 0;

        if (ic == 0) {
            // post-iteration-0, post-dataset-swap, pre-prologue state (matches GPU DATASET_SWAP ic=0)
            fprintf(g0f, "POSTSWAP0");
            for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)R[k]);
            for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)f2b(F[k]));
            for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)f2b(E[k]));
            fprintf(g0f, "\n");
        }

        if (ic == 0 || ic == NUM_ITER - 1) {
            printf("iter %d done: R0=%016llx fprc=%u ma=%08x mx=%08x\n", ic,
                   (unsigned long long)R[0], fprc, ma, mx);
        }
    }

    // compare against cpu_rf_p0.bin
    // CPU final register file: r[0..7], f^e[0..7], e[0..7], a[0..7]
    std::vector<uint8_t> emuRF(256);
    memcpy(emuRF.data(), R, 64);
    for (int k = 0; k < 8; ++k) {
        uint64_t fe = f2b(F[k]) ^ f2b(E[k]);
        uint64_t e = f2b(E[k]);
        memcpy(emuRF.data() + 64 + k * 8, &fe, 8);
        memcpy(emuRF.data() + 128 + k * 8, &e, 8);
    }
    memcpy(emuRF.data() + 192, (uint8_t*)R + 192, 64);
    int diff = 0, firstdiff = -1;
    for (int i = 0; i < 256; ++i) {
        if (emuRF[i] != rfCPU[i]) { if (firstdiff < 0) firstdiff = i; ++diff; }
    }
    printf("emulated register file vs cpu_rf_p0: %d/256 differ (first @%d)\n", diff, firstdiff);
    printf("EFT-vs-exact op diffs: fma=%llu/%llu div=%llu/%llu sqrt=%llu/%llu\n",
           (unsigned long long)cnt_fma_diff, (unsigned long long)cnt_fma,
           (unsigned long long)cnt_div_diff, (unsigned long long)cnt_div,
           (unsigned long long)cnt_sqrt_diff, (unsigned long long)cnt_sqrt);
    if (diff) {
        for (int i = 0; i < 32; ++i) {
            uint64_t g = ((uint64_t*)emuRF.data())[i], c = ((uint64_t*)rfCPU.data())[i];
            printf("[%02d] emul=%016llx cpu=%016llx %s\n", i,
                    (unsigned long long)g, (unsigned long long)c, g == c ? "" : "  <== DIFF");
        }
    }
    return 0;
}
