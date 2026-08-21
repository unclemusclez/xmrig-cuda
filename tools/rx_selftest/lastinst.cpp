// Extract the last EXEC block of iteration 0 from stagecmp_nt.err:
// finds the second "[CPU_TRACE] EXEC pc=0" and prints the block immediately before it.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>

int main() {
    FILE* f = fopen("stagecmp_nt.err", "rb");
    if (!f) return 1;
    char line[512];
    std::string block[13];
    int pc = -1;
    std::string lastblock[13];
    int lastpc = -1;
    int n = 0;
    int pc0_count = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "[CPU_TRACE] EXEC pc=", 20) == 0) {
            if (pc0_count == 1) {  // we're inside iteration 0 (after first pc=0)
                lastpc = pc;
                for (int i = 0; i < n; ++i) lastblock[i] = block[i];
            }
            pc = atoi(line + 20);
            if (pc == 0) { pc0_count++; if (pc0_count >= 2) break; }
            n = 0;
            continue;
        }
        if (n < 13) block[n++] = std::string(line);
    }
    fclose(f);
    printf("last EXEC of iteration 0: pc=%d\n", lastpc);
    for (int i = 0; i < 13 && !lastblock[i].empty(); ++i) fputs(lastblock[i].c_str(), stdout);
    return 0;
}
