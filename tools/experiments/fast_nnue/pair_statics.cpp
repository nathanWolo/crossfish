// nnue2 stage 4, isolation test (ii): in a two-engine build (fast_pair.py), each engine evaluates with
// its own net and nothing else.
//
//   pair_statics POSITIONS.cfdg N DEPTH OUT.txt
//
// One Dev and one Prev engine, created once and used for every position, run on each of the first N
// records (DgRec, 128 bytes, the UTTTAI state in the first 93) exactly what `datagen label IN OUT DEPTH 1`
// runs per record: evaluate_hce, evaluate (the static eval, from scratch) and, with DEPTH > 0,
// search_fixed_depth(DEPTH) (the incremental accumulator stack, the eval cache, the TT). Dev and Prev
// alternate on every position, so any state the two engines shared would show as a difference from the
// labels of the single-net builds. OUT gets one line per record:
//   index dev_static prev_static dev_search prev_search      (search 99999999: not searched)
// Compile in the pairing directory (it includes that directory's test_bots.cpp, which includes its
// crossfish_prev.hpp and crossfish_dev.hpp); pair_statics_check.py compares OUT with the references.
#define main tb_main
#include "test_bots.cpp"
#undef main

#include <memory>

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: pair_statics POSITIONS.cfdg N DEPTH OUT.txt\n");
        return 2;
    }
    const char *in_path = argv[1];
    const long long n = std::atoll(argv[2]);
    const int depth = std::atoi(argv[3]);
    const char *out_path = argv[4];
    constexpr int NONE = 99999999;
    CrossfishDev::init_mini_lut();
    CrossfishPrev::init_mini_lut();
    auto dev = std::make_unique<CrossfishDev>();
    auto prev = std::make_unique<CrossfishPrev>();
    FILE *in = std::fopen(in_path, "rb");
    FILE *out = std::fopen(out_path, "w");
    if (!in || !out) {
        std::fprintf(stderr, "cannot open %s or %s\n", in_path, out_path);
        return 1;
    }
    unsigned char rec[128];
    GlobalBoard b;
    long long i = 0, bad = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (; i < n && std::fread(rec, sizeof rec, 1, in) == 1; i++) {
        char s[94];
        std::memcpy(s, rec, 93);
        s[93] = 0;
        int dev_static = 0, prev_static = 0, dev_search = NONE, prev_search = NONE, score = 0;
        if (!prepare_board_for_search(b, s)) {
            bad++;
            std::fprintf(out, "%lld bad\n", i);
            continue;
        }
        (void)dev->evaluate_hce(b);
        dev_static = dev->evaluate(b);
        if (depth > 0 && dev->search_fixed_depth(b, depth, score)) dev_search = score;
        prepare_board_for_search(b, s);
        (void)prev->evaluate_hce(b);
        prev_static = prev->evaluate(b);
        if (depth > 0 && prev->search_fixed_depth(b, depth, score)) prev_search = score;
        std::fprintf(out, "%lld %d %d %d %d\n", i, dev_static, prev_static, dev_search, prev_search);
    }
    std::fclose(out);
    std::fclose(in);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("pair_statics: %lld positions (%lld unusable), depth %d, %.1f s -> %s\n", i, bad, depth, sec, out_path);
    return 0;
}
