// Micro-benchmark of fast_nnue.hpp's pieces on real positions (single thread):
//
//   fast_bench POSITIONS.cfdg [N_POS] [REPS]
//
// For N_POS records of POSITIONS (DgRec) and every legal move from each:
//   refresh   both perspectives' accumulators from scratch (root refresh / evaluate_board)
//   update    one ply of the lazy stack from a hot parent: a non-deciding move (one row
//             per perspective) and a deciding move (stones removed + decided row)
//   eval      eval_avx on the child (crelu + constraint row, sparse dense, output), the
//             alternative index-list kernel eval_avx_list, and the
//             scalar reference eval_ref, plus the mean number of nonzero activation pairs.
// Work runs in chunks of 32 positions so parents and children stay cache-resident as in
// a search; each chunk is timed as a whole (steady_clock) and repeated REPS times.
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
    int stm;         // the child's side to move
};

static bool wins(int m) {
    static const int W[8] = {7, 56, 448, 73, 146, 292, 273, 84};
    for (int w : W)
        if ((m & w) == w) return true;
    return false;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fast_bench POSITIONS.cfdg [N_POS] [REPS]\n");
        return 2;
    }
    FILE *f = std::fopen(argv[1], "rb");
    if (!f) return 1;
    std::vector<BenchRec> recs;
    BenchRec r;
    while (std::fread(&r, sizeof(r), 1, f) == 1) recs.push_back(r);
    std::fclose(f);
    const int npos = std::min<int>(argc > 2 ? std::atoi(argv[2]) : 20000, (int)recs.size());
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    const fnnue::Net &net = fnnue::net();

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
    const int CH = 32;
    std::vector<fnnue::AccEntry> parents(CH), kids(CH * 81);
    double t_refresh = 0, t_upd_n = 0, t_upd_d = 0, t_eval = 0, t_ref = 0, t_list = 0;
    long long n_refresh = 0, n_upd_n = 0, n_upd_d = 0, n_eval = 0, n_ref = 0, removed = 0, pairs = 0;
    long long sink = 0;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    for (size_t base = 0; base < boards.size(); base += CH) {
        const int nb = (int)std::min<size_t>(CH, boards.size() - base);
        // children of this chunk
        std::vector<Child> ch_n, ch_d;
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
                    const int dec2 = dec | (x.d.decided >= 0 ? 1 << mb : 0);
                    x.constraint = (dec2 >> sq & 1) ? 9 : sq;
                    x.stm = stm ^ 1;
                    (x.d.decided >= 0 ? ch_d : ch_n).push_back(x);
                }
            }
        }
        fnnue::Stack st;  // uses acc[0] as parent, acc[1] as child
        for (int rep = 0; rep < reps; rep++) {
            auto t0 = clk::now();
            for (int k = 0; k < nb; k++) {
                fnnue::scratch_avx(net, boards[base + k], 0, parents[k].v[0]);
                fnnue::scratch_avx(net, boards[base + k], 1, parents[k].v[1]);
            }
            auto t1 = clk::now();
            t_refresh += secs(t0, t1);
            n_refresh += nb;
            for (int pass = 0; pass < 2; pass++) {
                const std::vector<Child> &v = pass ? ch_d : ch_n;
                auto u0 = clk::now();
                for (size_t i = 0; i < v.size(); i++) {
                    // update() reads acc[j-1] and dirty[j]; point them at this child
                    st.acc[0] = parents[v[i].parent];  // not timed separately: subtract below
                    st.dirty[1] = v[i].d;
                    st.update(net, 1);
                    kids[i + (pass ? ch_n.size() : 0)] = st.acc[1];
                }
                auto u1 = clk::now();
                // the same loop without update(): copy overhead to subtract
                for (size_t i = 0; i < v.size(); i++) {
                    st.acc[0] = parents[v[i].parent];
                    st.dirty[1] = v[i].d;
                    kids[i + (pass ? ch_n.size() : 0)] = st.acc[1];
                }
                auto u2 = clk::now();
                const double dt = secs(u0, u1) - secs(u1, u2);
                if (pass) { t_upd_d += dt; n_upd_d += (long long)v.size(); }
                else { t_upd_n += dt; n_upd_n += (long long)v.size(); }
                if (pass && rep == 0)
                    for (const Child &x : v) removed += __builtin_popcount(x.d.before[0] | x.d.before[1]);
            }
            // evaluation of every child (children stored n then d)
            const size_t nk = ch_n.size() + ch_d.size();
            auto e0 = clk::now();
            for (size_t i = 0; i < nk; i++) {
                const Child &x = i < ch_n.size() ? ch_n[i] : ch_d[i - ch_n.size()];
                sink += fnnue::eval_avx(net, kids[i].v[x.stm], kids[i].v[x.stm ^ 1], x.constraint);
            }
            auto e1 = clk::now();
            t_eval += secs(e0, e1);
            n_eval += (long long)nk;
            for (size_t i = 0; i < nk; i++) {  // the index-list kernel, for comparison
                const Child &x = i < ch_n.size() ? ch_n[i] : ch_d[i - ch_n.size()];
                sink += fnnue::eval_avx_list(net, kids[i].v[x.stm], kids[i].v[x.stm ^ 1], x.constraint);
            }
            t_list += secs(e1, clk::now());
            if (rep == 0) {
                auto r0 = clk::now();
                for (size_t i = 0; i < nk; i++) {
                    const Child &x = i < ch_n.size() ? ch_n[i] : ch_d[i - ch_n.size()];
                    int32_t us[fnnue::A], them[fnnue::A];
                    for (int l = 0; l < fnnue::A; l++) {
                        us[l] = kids[i].v[x.stm][l];
                        them[l] = kids[i].v[x.stm ^ 1][l];
                    }
                    const int e = fnnue::eval_ref(net, us, them, x.constraint);
                    sink += e;
                    if (e != fnnue::eval_avx(net, kids[i].v[x.stm], kids[i].v[x.stm ^ 1], x.constraint)
                        || e != fnnue::eval_avx_list(net, kids[i].v[x.stm], kids[i].v[x.stm ^ 1], x.constraint)) {
                        std::fprintf(stderr, "eval_ref != eval_avx\n");
                        return 1;
                    }
                    const int16_t *con = net.W0[fnnue::ROW_CON + x.constraint];
                    for (int v = 0; v < 2; v++) {
                        const int16_t *a = kids[i].v[v == 0 ? x.stm : x.stm ^ 1];
                        for (int l = 0; l < fnnue::A; l += 2)
                            pairs += (a[l] + con[l] > 0) || (a[l + 1] + con[l + 1] > 0);
                    }
                }
                auto r1 = clk::now();
                t_ref += secs(r0, r1);
                n_ref += (long long)nk;
            }
        }
    }
    std::printf("fast_bench: %zu positions, %lld evals x%d reps\n", boards.size(), n_ref, reps);
    std::printf("  refresh (both perspectives)   %7.1f ns\n", 1e9 * t_refresh / n_refresh);
    std::printf("  update non-deciding (both)    %7.1f ns   (%lld)\n", 1e9 * t_upd_n / n_upd_n, n_upd_n / reps);
    std::printf("  update deciding (both)        %7.1f ns   (%lld, %.2f stones removed)\n", 1e9 * t_upd_d / n_upd_d,
                n_upd_d / reps, n_upd_d ? (double)removed / (n_upd_d / reps) : 0.0);
    std::printf("  eval_avx                      %7.1f ns   (%.1f nonzero pairs of %d)\n", 1e9 * t_eval / n_eval,
                (double)pairs / n_ref, fnnue::NPAIR);
    std::printf("  eval_avx_list (index list)    %7.1f ns\n", 1e9 * t_list / n_eval);
    std::printf("  eval_ref (scalar reference)   %7.1f ns\n", 1e9 * t_ref / n_ref);
    std::printf("  (sink %lld)\n", sink);
    return 0;
}
