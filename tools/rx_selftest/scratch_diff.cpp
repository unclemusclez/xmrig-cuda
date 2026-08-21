// Diff two 2MB scratchpads: count differing bytes, list contiguous runs, and
// report the address pattern (mod 64) of the first differing bytes.
#include <cstdio>
#include <cstdint>
#include <vector>

int main(int argc, char** argv)
{
    const char* a = argc > 1 ? argv[1] : "gpu_scratch_p2.bin";
    const char* b = argc > 2 ? argv[2] : "cpu_scratch_p2.bin";
    FILE* fa = fopen(a, "rb"); FILE* fb = fopen(b, "rb");
    if (!fa || !fb) { fprintf(stderr, "open failed\n"); return 1; }
    std::vector<uint8_t> x(1 << 21), y(1 << 21);
    if (fread(x.data(), 1, x.size(), fa) != x.size()) return 1;
    if (fread(y.data(), 1, y.size(), fb) != y.size()) return 1;
    fclose(fa); fclose(fb);

    long long diffs = 0;
    long long runs = 0, inRun = 0, runStart = -1;
    std::vector<long long> runStarts, runLens;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] != y[i]) {
            ++diffs;
            if (!inRun) { inRun = 1; runStart = (long long)i; runStarts.push_back(runStart); }
            runLens.push_back(0);
            runLens.back()++;
        } else {
            if (inRun) { inRun = 0; runs++; }
        }
    }
    if (inRun) runs++;
    printf("differing bytes: %lld, contiguous runs: %lld\n", diffs, runs);
    int shown = 0;
    long long pos = 0; long long cur = 0; long long len = 0; int idx = 0;
    // re-walk to print first runs cleanly
    inRun = 0; long long s = -1; long long l = 0;
    for (size_t i = 0; i < x.size() && shown < 12; ++i) {
        if (x[i] != y[i]) { if (!inRun) { inRun = 1; s = i; l = 1; } else l++; }
        else if (inRun) { printf("run @ 0x%07llx len %lld (mod64=%lld)\n", s, l, s % 64); inRun = 0; shown++; }
    }
    if (inRun && shown < 12) printf("run @ 0x%07llx len %lld (mod64=%lld)\n", s, l, s % 64);
    // mod-64 histogram of all differing byte addresses
    long long hist[64] = {0};
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) hist[i % 64]++;
    printf("mod64 histogram of differing bytes (nonzero):\n");
    for (int m = 0; m < 64; ++m) if (hist[m]) printf("  mod64=%2d : %lld\n", m, hist[m]);
    return 0;
}
