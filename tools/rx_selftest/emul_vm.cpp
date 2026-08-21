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

static inline void chk_fp(FILE* evf, int ic, int ip, int sub, uint32_t opcode, uint32_t fprc, uint64_t bits) {
    uint64_t exp = (bits >> 52) & 0x7FF;
    uint64_t man = bits & 0xFFFFFFFFFFFFFull;
    if (exp == 0 && man != 0) fprintf(evf, "SUBNORMAL op=%u ic=%d ip=%d sub=%d fprc=%u bits=%016llx\n", opcode, ic, ip, sub, fprc, (unsigned long long)bits);
    else if (exp == 0x7FF) fprintf(evf, "NANINF op=%u ic=%d ip=%d sub=%d fprc=%u bits=%016llx\n", opcode, ic, ip, sub, fprc, (unsigned long long)bits);
    else if (exp == 0 && man == 0) fprintf(evf, "ZERO op=%u ic=%d ip=%d sub=%d fprc=%u bits=%016llx\n", opcode, ic, ip, sub, fprc, (unsigned long long)bits);
}

// GPU div_rnd<-1,false>: Newton-Raphson reciprocal-sqrt style division
static inline double gpu_div(double a, double b) {
    double y0 = 1.0 / b;
    double y1 = fma(y0, fma(-b, y0, 1.0), y0);
    const double t0 = a * y1; (void)t0;
    double y2 = fma(y1, fma(-b, y1, 1.0), y1);
    const double t1 = a * y2; (void)t1;
    double y3 = fma(y2, fma(-b, y2, 1.0), y2);
    return a * y3;   // fprc==0 -> round-to-nearest result used as-is
}

// GPU sqrt_rnd<-1,false>
static inline double gpu_sqrt(double a) {
    double y0 = 1.0 / sqrt(a);
    double y1 = y0 * fma(0.5, fma(-a, y0 * y0, 1.0), 1.0);
    double y2 = y1 * fma(0.5, fma(-a, y1 * y1, 1.0), 1.0);
    return a * y2;
}

// CPU-exact directed FP ops via fesetround (ground truth for fma/div/sqrt).
static const int fes_modes[4] = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
static inline void set_rnd(uint32_t fprc) { fesetround(fes_modes[fprc & 3]); }

static inline uint64_t umulhi(uint64_t a, uint64_t b) {
    uint64_t hi; _umul128(a, b, &hi); return hi;
}
static inline uint64_t mulhi(int64_t a, int64_t b) {
    int64_t hi; _mul128(a, b, &hi); return (uint64_t)hi;
}

int main(int argc, char** argv) {
    const char* vmf = (argc > 1) ? argv[1] : "gpu_vmstate_p0.bin";
    const char* scf = (argc > 2) ? argv[2] : "gpu_scratch_p0.bin";
    const char* rff = (argc > 3) ? argv[3] : "cpu_rf_p0.bin";
    char tag[64] = "p0";
    {
        const char* s = strstr(vmf, "_p");
        if (s && s[2]) { snprintf(tag, sizeof(tag), "p%c", s[2]); }
    }
    auto vmstate = rd(vmf);     // 2048 bytes: registers+imm_buf+compiled program (post-init_vm, in hash()'s real flow)
    auto scratch = rd(scf);     // 2 MB scratchpad item 0
    auto rfCPU   = rd(rff);     // 256 bytes: CPU reference register file after this program
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

    uint32_t fprc = (argc > 4) ? (uint32_t)strtoul(argv[4], nullptr, 10) : 0; // carry-in rounding mode
    uint32_t spAddr0 = mx; // first=true -> {mx, ma}
    uint32_t spAddr1 = ma;

    const int NUM_ITER = 2048;
    int g_trace_ic = -1;
    {
        const char* e = getenv("EMUL_TRACE_IC");
        if (e) g_trace_ic = atoi(e);
    }
    FILE* gtrf = (g_trace_ic >= 0) ? fopen("emu_ic_boundary_states.txt", "w") : nullptr;
    FILE* iterf = fopen("emul_iter_states.txt", "w");
    FILE* g0f = fopen("emu_g0_groups.txt", "w");
    FILE* g1f = fopen("emu_g1_groups.txt", "w");
    FILE* evf = fopen("emu_events.txt", "w");
    int first_ic[16]; for (int i = 0; i < 16; ++i) first_ic[i] = -1;
    uint64_t total_ops[16] = {0};
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
        if (ic == g_trace_ic) {
            FILE* spf = fopen("emu_scratch_icN.bin", "wb");
            if (spf) { fwrite(scratch.data(), 1, scratch.size(), spf); fclose(spf); }
            fprintf(stderr, "[emul] scratch snapshot at ic=%d (%zu bytes)\n", ic, scratch.size());
        }
        fprintf(iterf, "%d", ic);
        for (int k = 0; k < 8; ++k) fprintf(iterf, " %016llx", (unsigned long long)R[k]);
        for (int k = 0; k < 8; ++k) fprintf(iterf, " %016llx", (unsigned long long)f2b(F[k]));
        for (int k = 0; k < 8; ++k) fprintf(iterf, " %016llx", (unsigned long long)f2b(E[k]));
        fprintf(iterf, " ma=%08x mx=%08x sp0=%08x sp1=%08x fprc=%u\n", ma, mx, spAddr0, spAddr1, fprc);
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
        long long insts_done = 0; // CPU-instructions executed so far this iteration
        while (ip < (int32_t)program_length) {
            bool is_last_group = false;
            {
                uint32_t h = compiled[ip];
                int32_t nw = (h >> NUM_INSTS_OFFSET) & 7;
                int32_t nf = (h >> NUM_FP_INSTS_OFFSET) & 7;
                int32_t ni = nw - nf;
                is_last_group = (ip + ni + 1 >= (int32_t)program_length);
            }
            if (ic == g_trace_ic && gtrf) {
                fprintf(gtrf, "ip=%d insts=%lld", ip, insts_done);
                for (int k = 0; k < 8; ++k) fprintf(gtrf, " %016llx", (unsigned long long)R[k]);
                for (int k = 0; k < 8; ++k) fprintf(gtrf, " %016llx", (unsigned long long)f2b(F[k]));
                for (int k = 0; k < 8; ++k) fprintf(gtrf, " %016llx", (unsigned long long)f2b(E[k]));
                fprintf(gtrf, "\n");
            }
            if (ic == 0 && is_last_group) {
                fprintf(stderr, "EMU_LAST_GROUP ic=0 ip=%d R: ", ip);
                for (int k = 0; k < 8; ++k) fprintf(stderr, "%016llx ", (unsigned long long)R[k]);
                fprintf(stderr, "\n");
            }
            if (ic == 0 || ic == 1) {
                FILE* gf = (ic == 0) ? g0f : g1f;
                fprintf(gf, "%d", ip);
                for (int k = 0; k < 8; ++k) fprintf(gf, " %016llx", (unsigned long long)R[k]);
                for (int k = 0; k < 8; ++k) fprintf(gf, " %016llx", (unsigned long long)f2b(F[k]));
                for (int k = 0; k < 8; ++k) fprintf(gf, " %016llx", (unsigned long long)f2b(E[k]));
                fprintf(gf, "\n");
            }
            imm_buf[IMM_INDEX_COUNT] = ip;
            uint32_t inst = compiled[ip];
            int32_t num_workers = (inst >> NUM_INSTS_OFFSET) & 7;
            int32_t num_fp_insts = (inst >> NUM_FP_INSTS_OFFSET) & 7;
            int32_t num_insts = num_workers - num_fp_insts;
            if (ic == g_trace_ic) insts_done += (num_insts + 1); // CPU instructions in this group

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
                total_ops[opcode]++;
                if (first_ic[opcode] < 0) { first_ic[opcode] = ic; fprintf(evf, "FIRST op=%u ic=%d ip=%d sub=%d w=%08x loc=%u\n", opcode, ic, ip, (int)sub, w, location); }
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
                        set_rnd(fprc);
                        dst = f2b(fma(a, is_mul ? b : 1.0, is_mul ? 0.0 : b));
                        fesetround(FE_TONEAREST);
                        chk_fp(evf, ic, ip, (int)sub, opcode, fprc, dst);
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
                        set_rnd(fprc);
                        dst = f2b(sqrt(b2f(dst)));
                        fesetround(FE_TONEAREST);
                        chk_fp(evf, ic, ip, (int)sub, opcode, fprc, dst);
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
                        set_rnd(fprc);
                        dst = f2b(b2f(dst) / b2f(src));
                        fesetround(FE_TONEAREST);
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
            if (ic == g_trace_ic && gtrf) {
                for (int i = 0; i < npw; ++i) {
                    long off = (long)((uint8_t*)pw[i].a - (uint8_t*)R);
                    fprintf(gtrf, "R off=%ld old=%016llx new=%016llx ip=%d\n", off,
                            (unsigned long long)*pw[i].a, (unsigned long long)pw[i].v, ip);
                }
                for (int i = 0; i < nspw; ++i) {
                    uint64_t addr = (uint64_t)((uint8_t*)spw[i].a - (uint8_t*)scratch.data());
                    fprintf(gtrf, "S addr=%08llx val=%016llx ip=%d\n", (unsigned long long)addr,
                            (unsigned long long)spw[i].v, ip);
                }
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

        if (ic < NUM_ITER) {
            fprintf(iterf, "%d END %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx sp0=%08x sp1=%08x fprc=%u\n",
                    ic,
                    (unsigned long long)R[0], (unsigned long long)R[1], (unsigned long long)R[2], (unsigned long long)R[3],
                    (unsigned long long)R[4], (unsigned long long)R[5], (unsigned long long)R[6], (unsigned long long)R[7],
                    (unsigned long long)f2b(F[0]), (unsigned long long)f2b(F[1]), (unsigned long long)f2b(F[2]), (unsigned long long)f2b(F[3]),
                    (unsigned long long)f2b(E[0]), (unsigned long long)f2b(E[1]), (unsigned long long)f2b(E[2]), (unsigned long long)f2b(E[3]),
                    spAddr0, spAddr1, fprc);
            fflush(iterf);
        }

        if (ic == 0) {
            // post-iteration-0, post-dataset-swap, pre-prologue state (matches GPU DATASET_SWAP ic=0)
            fprintf(g0f, "POSTSWAP0");
            for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)R[k]);
            for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)f2b(F[k]));
            for (int k = 0; k < 8; ++k) fprintf(g0f, " %016llx", (unsigned long long)f2b(E[k]));
            fprintf(g0f, "\n");
        }

        if (ic == NUM_ITER - 1 || ic == g_trace_ic) {
            printf("iter %d done: R0=%016llx fprc=%u ma=%08x mx=%08x insts_done=%lld\n", ic,
                   (unsigned long long)R[0], fprc, ma, mx, insts_done);
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
    printf("emulated register file vs %s: %d/256 differ (first @%d)\n", rff, diff, firstdiff);
    if (diff) {
        for (int i = 0; i < 32; ++i) {
            uint64_t g = ((uint64_t*)emuRF.data())[i], c = ((uint64_t*)rfCPU.data())[i];
            printf("[%02d] emul=%016llx cpu=%016llx %s\n", i,
                    (unsigned long long)g, (unsigned long long)c, g == c ? "" : "  <== DIFF");
        }
    }
    return 0;
}
