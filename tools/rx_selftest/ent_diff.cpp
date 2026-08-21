#include <cstdio>
#include <cstdint>
#include <vector>
int main(int argc, char** argv) {
    const char* a = argv[1]; const char* b = argv[2];
    FILE* fa = fopen(a, "rb"); FILE* fb = fopen(b, "rb");
    std::vector<uint8_t> x(2176), y(2176);
    size_t na = fread(x.data(), 1, 2176, fa); size_t nb = fread(y.data(), 1, 2176, fb);
    fclose(fa); fclose(fb);
    printf("sizes: %zu %zu\n", na, nb);
    size_t n = na < nb ? na : nb;
    long long diffs = 0, first = -1;
    long long reg_diffs = 0, prog_diffs = 0;
    for (size_t i = 0; i < n; ++i) {
        if (x[i] != y[i]) { ++diffs; if (first < 0) first = (long long)i; if (i < 128) reg_diffs++; else prog_diffs++; }
    }
    printf("total diffs=%lld first@%lld reg_diffs(0-127)=%lld prog_diffs(128+)=%lld\n", diffs, first, reg_diffs, prog_diffs);
    // show first few program-region diffs in detail
    int shown = 0;
    for (size_t i = 128; i < n && shown < 6; ++i) if (x[i] != y[i]) {
        size_t s = i, e = i; while (e + 1 < n && x[e + 1] != y[e + 1]) ++e;
        printf("diff run @%zu..%zu\n", s, e);
        printf("  gpu: "); for (size_t k = s; k <= e && k - s < 24; ++k) printf("%02x", x[k]); printf("\n");
        printf("  cpu: "); for (size_t k = s; k <= e && k - s < 24; ++k) printf("%02x", y[k]); printf("\n");
        i = e; ++shown;
    }
    return 0;
}
