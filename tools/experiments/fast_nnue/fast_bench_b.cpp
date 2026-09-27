// Micro-benchmark of fast_nnue_b.hpp's pieces on real positions (single thread), for the BGN1 net
// named by FASTNNUE_PATH:
//
//   fast_bench_b POSITIONS.cfdg [N_POS] [REPS]
//
// For N_POS records of POSITIONS (DgRec) and every legal move from each:
//   load      reading + quantizing the net (and baking it from the generator with FASTNNUE_BAKE=1)
//   refresh   both perspectives' accumulators from scratch (root refresh / evaluate_board)
//   update    one ply of the lazy stack from a hot parent: a non-deciding move (pattern row out,
//             pattern row in, per perspective) and a deciding move (pattern row out, decided row in)
//   eval      the evaluation-time rows (constraint rows, the forced board's pattern rows) plus
//             eval_avx on the child, and the scalar reference eval_ref, plus the mean number of
//             nonzero activation pairs
//   cold      update + eval of every child right after evicting the caches (a 256 MB sweep), i.e.
//             with the table rows coming from DRAM, against the same work hot
// Work runs in chunks of 32 positions so parents and children stay cache-resident as in a search;
// each chunk is timed as a whole (steady_clock) and repeated REPS times.
#define main tb_main
#include "test_bots.cpp"
#undef main

#include <chrono>

#pragma pack(push, 1)
struct BenchRec {
    char s[93];
    uint8_t rest[35];
};
#pragma pack(pop)

struct LBoard {  // what fast_nnue's board templates read
    std::array<MiniBoard, 9> mini_boards;
    std::array<int, 3> mini_board_states;
    int n_moves;
};

struct Child {
    int parent;      // position index within the chunk
    fnnue::Dirty d;
    int constraint;  // the child's constraint
    LBoard b;        // the child position
};

static bool wins(int m) {
    static const int W[8] = {7, 56, 448, 73, 146, 292, 273, 84};
    for (int w : W)
        if ((m & w) == w) return true;
    return false;
}

static std::vector<uint8_t> g_evict(256u << 20, 1);
static long long evict() {
    long long s = 0;
    for (size_t i = 0; i < g_evict.size(); i += 64) s += g_evict[i]++;
    return s;
}

template <class N>
static int run(const N &net, const std::vector<LBoard> &boards, const std::vector<int> &cons, int reps) {
    using Stk = fnnue::bnn::Stack<N>;
    using Entry = typename Stk::Entry;
    constexpr int A = N::A;
    const int CH = 32;
    std::vector<Entry> parents(CH), kids(CH * 81);
    double t_refresh = 0, t_upd_n = 0, t_upd_d = 0, t_eval = 0, t_ref = 0, t_cold = 0, t_hot = 0;
    long long n_refresh = 0, n_upd_n = 0, n_upd_d = 0, n_eval = 0, n_ref = 0, pairs = 0, n_cold = 0, forced = 0;
    long long sink = 0;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    Stk st(net);  // acc[0] parent, acc[1] child
    for (size_t base = 0; base < boards.size(); base += CH) {
        const int nb = (int)std::min<size_t>(CH, boards.size() - base);
        std::vector<Child> ch;  // non-deciding first, then deciding
        std::vector<Child> chd;
        for (int k = 0; k < nb; k++) {
            const LBoard &b = boards[base + k];
            const int stm = b.n_moves & 1, c = cons[base + k];
            const int dec = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
            for (int mb = 0; mb < 9; mb++) {
                if ((c < 9 && mb != c) || (dec >> mb & 1)) continue;
                const int mine = b.mini_boards[mb].markers[stm], theirs = b.mini_boards[mb].markers[stm ^ 1];
                for (int sq = 0; sq < 9; sq++) {
                    if (((mine | theirs) >> sq) & 1) continue;
                    Child x;
                    x.parent = k;
                    x.d.mb = (uint8_t)mb;
                    x.d.sq = (uint8_t)sq;
                    x.d.stm = (uint8_t)stm;
                    x.d.before[stm] = (uint16_t)mine;
                    x.d.before[stm ^ 1] = (uint16_t)theirs;
                    const int nm = mine | (1 << sq);
                    x.d.decided = (int8_t)(wins(nm) ? stm : ((nm | theirs) == 511 ? 2 : -1));
                    x.b = b;
                    x.b.mini_boards[mb].markers[stm] = nm;
                    if (x.d.decided >= 0) x.b.mini_board_states[x.d.decided] |= 1 << mb;
                    x.b.n_moves++;
                    const int dec2 = dec | (x.d.decided >= 0 ? 1 << mb : 0);
                    x.constraint = (dec2 >> sq & 1) ? 9 : sq;
                    (x.d.decided >= 0 ? chd : ch).push_back(x);
                }
            }
        }
        const size_t nn = ch.size();
        ch.insert(ch.end(), chd.begin(), chd.end());
        const size_t nk = ch.size();
        auto eval_child = [&](size_t i) {
            const Child &x = ch[i];
            const int stm = x.b.n_moves & 1;
            const fnnue::bnn::EvalRows<N, LBoard> er(net, x.b, stm, x.constraint);
            const Entry &e = kids[i];
            return fnnue::bnn::eval_avx(net, e.v[stm], e.v[stm ^ 1], er.cu, er.ct, er.fu, er.ft,
                                        e.ps[stm] - e.ps[stm ^ 1] + er.ps_extra);
        };
        for (int rep = 0; rep <= reps; rep++) {  // rep 0: cold (after eviction), not in the hot means
            if (rep == 0) sink += evict();
            auto t0 = clk::now();
            for (int k = 0; k < nb; k++) {
                fnnue::bnn::scratch_avx(net, boards[base + k], 0, parents[k].v[0], parents[k].ps[0]);
                fnnue::bnn::scratch_avx(net, boards[base + k], 1, parents[k].v[1], parents[k].ps[1]);
            }
            auto t1 = clk::now();
            if (rep) { t_refresh += secs(t0, t1); n_refresh += nb; }
            if (rep == 0) {  // cold: update + eval with rows from DRAM (parents were just rebuilt: hot)
                auto c0 = clk::now();
                for (size_t i = 0; i < nk; i++) {
                    st.acc[0] = parents[ch[i].parent];
                    st.dirty[1] = ch[i].d;
                    st.update(1);
                    kids[i] = st.acc[1];
                    sink += eval_child(i);
                }
                auto c1 = clk::now();
                // the same work again, hot
                for (size_t i = 0; i < nk; i++) {
                    st.acc[0] = parents[ch[i].parent];
                    st.dirty[1] = ch[i].d;
                    st.update(1);
                    kids[i] = st.acc[1];
                    sink += eval_child(i);
                }
                auto c2 = clk::now();
                t_cold += secs(c0, c1);
                t_hot += secs(c1, c2);
                n_cold += (long long)nk;
                continue;
            }
            for (int pass = 0; pass < 2; pass++) {
                const size_t lo = pass ? nn : 0, hi = pass ? nk : nn;
                auto u0 = clk::now();
                for (size_t i = lo; i < hi; i++) {
                    st.acc[0] = parents[ch[i].parent];
                    st.dirty[1] = ch[i].d;
                    st.update(1);
                    kids[i] = st.acc[1];
                }
                auto u1 = clk::now();
                for (size_t i = lo; i < hi; i++) {  // the copies alone, subtracted
                    st.acc[0] = parents[ch[i].parent];
                    st.dirty[1] = ch[i].d;
                    kids[i] = st.acc[1];
                }
                auto u2 = clk::now();
                const double dt = secs(u0, u1) - secs(u1, u2);
                if (pass) { t_upd_d += dt; n_upd_d += (long long)(hi - lo); }
                else { t_upd_n += dt; n_upd_n += (long long)(hi - lo); }
            }
            for (size_t i = 0; i < nk; i++) {  // restore every child's accumulator
                st.acc[0] = parents[ch[i].parent];
                st.dirty[1] = ch[i].d;
                st.update(1);
                kids[i] = st.acc[1];
            }
            auto e0 = clk::now();
            for (size_t i = 0; i < nk; i++) sink += eval_child(i);
            auto e1 = clk::now();
            t_eval += secs(e0, e1);
            n_eval += (long long)nk;
            if (rep == 1) {
                auto r0 = clk::now();
                for (size_t i = 0; i < nk; i++) {
                    const Child &x = ch[i];
                    const int stm = x.b.n_moves & 1;
                    const fnnue::bnn::EvalRows<N, LBoard> er(net, x.b, stm, x.constraint);
                    int32_t us[A], them[A];
                    for (int l = 0; l < A; l++) {
                        us[l] = kids[i].v[stm][l];
                        them[l] = kids[i].v[stm ^ 1][l];
                    }
                    const int32_t psd = kids[i].ps[stm] - kids[i].ps[stm ^ 1] + er.ps_extra;
                    const int e = fnnue::bnn::eval_ref(net, us, them, er.cu, er.ct, er.fu, er.ft, psd);
                    sink += e;
                    if (e != eval_child(i)) {
                        std::fprintf(stderr, "eval_ref != eval_avx\n");
                        return 1;
                    }
                    // scratch of the child == incremental
                    int32_t ref[2][A], ps[2];
                    fnnue::bnn::scratch_ref(net, x.b, 0, ref[0], ps[0]);
                    fnnue::bnn::scratch_ref(net, x.b, 1, ref[1], ps[1]);
                    for (int P = 0; P < 2; P++) {
                        if (ps[P] != kids[i].ps[P]) { std::fprintf(stderr, "psqt mismatch\n"); return 1; }
                        for (int l = 0; l < A; l++)
                            if (ref[P][l] != kids[i].v[P][l]) { std::fprintf(stderr, "acc mismatch\n"); return 1; }
                    }
                    forced += x.constraint < 9;
                    const int16_t *cc[2] = {er.cu, er.ct}, *ff[2] = {er.fu, er.ft};
                    for (int v = 0; v < 2; v++) {
                        const int16_t *a = kids[i].v[v == 0 ? stm : stm ^ 1];
                        for (int l = 0; l < A; l += 2)
                            pairs += (a[l] + cc[v][l] + ff[v][l] > 0) || (a[l + 1] + cc[v][l + 1] + ff[v][l + 1] > 0);
                    }
                }
                auto r1 = clk::now();
                t_ref += secs(r0, r1);
                n_ref += (long long)nk;
            }
        }
    }
    std::printf("fast_bench_b: A=%d L1=%d L2=%d, %zu positions, %lld children (%.0f%% forced) x%d reps\n", A, N::L1,
                N::L2, boards.size(), n_ref, 100.0 * forced / std::max(1LL, n_ref), reps);
    std::printf("  refresh (both perspectives)   %7.1f ns\n", 1e9 * t_refresh / n_refresh);
    std::printf("  update non-deciding (both)    %7.1f ns   (%lld)\n", 1e9 * t_upd_n / n_upd_n, n_upd_n / reps);
    std::printf("  update deciding (both)        %7.1f ns   (%lld)\n", 1e9 * t_upd_d / std::max(1LL, n_upd_d), n_upd_d / reps);
    std::printf("  eval (rows + eval_avx)        %7.1f ns   (%.1f nonzero pairs of %d)\n", 1e9 * t_eval / n_eval,
                (double)pairs / n_ref, N::NPAIR);
    std::printf("  eval_ref (scalar reference)   %7.1f ns\n", 1e9 * t_ref / n_ref);
    std::printf("  update+eval cold (DRAM rows)  %7.1f ns   vs the same hot %.1f ns\n", 1e9 * t_cold / n_cold,
                1e9 * t_hot / n_cold);
    std::printf("  (sink %lld)\n", sink);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fast_bench_b POSITIONS.cfdg [N_POS] [REPS]\n");
        return 2;
    }
    FILE *f = std::fopen(argv[1], "rb");
    if (!f) return 1;
    std::vector<BenchRec> recs;
    BenchRec r;
    while (std::fread(&r, sizeof(r), 1, f) == 1) recs.push_back(r);
    std::fclose(f);
    const int npos = std::min<int>(argc > 2 ? std::atoi(argv[2]) : 20000, (int)recs.size());
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const auto l0 = std::chrono::steady_clock::now();
    const int kind = fnnue::any_kind();
    std::printf("fast_bench_b: load %.0f ms (kind %d)\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - l0).count(), kind);
    std::vector<LBoard> boards;
    std::vector<int> cons;
    for (int i = 0; i < npos; i++) {
        GlobalBoard g;
        if (!prepare_board_for_search(g, recs[i].s)) continue;
        LBoard b;
        b.mini_boards = g.mini_boards;
        b.mini_board_states = g.mini_board_states;
        b.n_moves = g.n_moves;
        boards.push_back(b);
        cons.push_back(d16_mini_board_constraint(g));
    }
    switch (kind) {
    case fnnue::KIND_B64: return run(fnnue::bnn::g_bnet<fnnue::B64>, boards, cons, reps);
    case fnnue::KIND_B128: return run(fnnue::bnn::g_bnet<fnnue::B128>, boards, cons, reps);
    case fnnue::KIND_B64S: return run(fnnue::bnn::g_bnet<fnnue::B64S>, boards, cons, reps);
    case fnnue::KIND_B128S: return run(fnnue::bnn::g_bnet<fnnue::B128S>, boards, cons, reps);
    default: std::fprintf(stderr, "fast_bench_b: FNN1 nets: use stage 1's fast_bench\n"); return 2;
    }
}
