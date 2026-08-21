// Debug: dump the first parsed CPU state and first emulator state, word by word.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>

int main() {
    FILE* f = fopen("stagecmp_nt.err", "rb");
    char line[512];
    uint64_t cpu[24]; int rw=0, fw=0, ew=0; bool have=false; int seen_pc0=0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "[CPU_TRACE] EXEC pc=", 20) == 0) {
            if (have) break;
            int pc = atoi(line+20);
            if (pc == 0) seen_pc0++;
            continue;
        }
        if (seen_pc0 != 1) continue;
        unsigned long long v;
        if (sscanf(line, "  R[%*d]=%llx", &v) == 1) { if (rw<8) cpu[rw++]=v; continue; }
        unsigned long long fh, fl, eh, el;
        if (sscanf(line, "  F[%*d]=%16llx%16llx E[%*d]=%16llx%16llx", &fh,&fl,&eh,&el) == 4) {
            if (fw<7) { cpu[8+fw]=fl; cpu[9+fw]=fh; fw+=2; }
            if (ew<7) { cpu[16+ew]=el; cpu[17+ew]=eh; ew+=2; }
            have = (rw==8 && fw==8 && ew==8);
            continue;
        }
    }
    fclose(f);
    printf("CPU pc=0 (rw=%d fw=%d ew=%d):\n", rw, fw, ew);
    printf("  R:"); for (int i=0;i<8;++i) printf(" %016llx",(unsigned long long)cpu[i]); printf("\n");
    printf("  F:"); for (int i=8;i<16;++i) printf(" %016llx",(unsigned long long)cpu[i]); printf("\n");
    printf("  E:"); for (int i=16;i<24;++i) printf(" %016llx",(unsigned long long)cpu[i]); printf("\n");

    FILE* g = fopen("emu_g0_groups.txt", "r");
    char gl[4096];
    if (fgets(gl, sizeof gl, g)) {
        unsigned long long vals[24]; int n=0;
        char* sp = strchr(gl, ' ');
        char* p = sp;
        while (n<24) { char* end=nullptr; unsigned long long v=strtoull(p,&end,16); if(end==p)break; vals[n++]=v; p=end; }
        printf("EMU group0 (%d words):\n", n);
        printf("  R:"); for (int i=0;i<8;++i) printf(" %016llx", vals[i]); printf("\n");
        printf("  F:"); for (int i=8;i<16;++i) printf(" %016llx", vals[i]); printf("\n");
        printf("  E:"); for (int i=16;i<24;++i) printf(" %016llx", vals[i]); printf("\n");
        printf("DIFF words:");
        for (int i=0;i<24;++i) if (cpu[i]!=vals[i]) printf(" %d", i);
        printf("\n");
    }
    fclose(g);
    return 0;
}
