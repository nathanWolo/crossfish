// Stage 3: searches entered WITHOUT fnnue_stack.refresh_root must still evaluate correctly (the stage-1/2
// stack trusted stale per-ply "computed" flags left by an earlier search; the correctness review's
// review_fallback.cpp reproduced that). Built in a candidate directory next to its
// patched crossfish_dev.hpp (build_cand.sh builds fallback_repro[_check].exe), with or without -DFASTNNUE_CHECK:
//
//   fallback_repro POSITIONS.cfdg [N]        (FASTNNUE_PATH = any FNN1/BGN1 net the build accepts)
//
// For N (default 200) positions A of POSITIONS, each on a fresh engine that first searches A the way
// getMove does (refresh_root, then plies made and evaluated), four entries without refresh_root:
//   S1  an unrelated position B two plies deeper than A, evaluated directly (the review's case);
//   S2  A's search makes a second move and is cut before evaluating it; then B (two plies deeper, so
//       at the cut ply) is entered, one move made from it and the child evaluated: the walk back
//       reaches a ply whose recorded move belongs to the old search (a parent-key-only check would
//       accept the stale chain below B);
//   S3  the same position as A's evaluated grandchild, built afresh from a GlobalBoard: its entry
//       legitimately holds it, so it must be reused (no refresh) and be correct;
//   S4  after S1-S3, a normal search_fixed_depth(4) from A on the same engine (refresh_root): its
//       evaluations must not need any fallback refresh (counted in -DFASTNNUE_CHECK builds).
// Every eval is compared with the from-scratch fnnue::evaluate_any. Exit status 3 if any is wrong.
#include <memory>
#define private public
#define main tb_main
#include "test_bots.cpp"
#undef main
#undef private

#pragma pack(push, 1)
struct Rec {
    char s[93];
    uint8_t rest[35];
};
#pragma pack(pop)
static_assert(sizeof(Rec) == 128, "DgRec is 128 bytes");

using Dev = CrossfishDev;

static void enter(Dev &e, Dev::FastBoard &fb) {  // what getMove does, minus refresh_root
    e.init_hce_acc(fb);
    e.init_macro_key(fb);
    e.sync_terminal(fb);
}

static bool first_legal(Dev &e, Dev::FastBoard &fb, Move &m) {
    Move mv[81];
    if (e.check_winner_fast(fb) != -1 || e.fill_legal_moves_fast(fb, mv) <= 0) return false;
    m = mv[0];
    return true;
}

static int eval_here(Dev &e, Dev::FastBoard &fb) { return e.fnnue_stack.evaluate(fb, Dev::active_board_index(fb)); }

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fallback_repro POSITIONS.cfdg [N]\n");
        return 2;
    }
    const int want = argc > 2 ? std::atoi(argv[2]) : 200;
    FILE *f = std::fopen(argv[1], "rb");
    std::vector<Rec> all;
    Rec r;
    while (f && std::fread(&r, sizeof(r), 1, f) == 1) all.push_back(r);
    if (f) std::fclose(f);
    if (all.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    Dev::init_mini_lut();
    int tried[4] = {0, 0, 0, 0}, wrong[4] = {0, 0, 0, 0};
    long long s4_nodes = 0;
    auto report = [&](int s, int got, int truth, const GlobalBoard &g, const char *what) {
        tried[s]++;
        if (got != truth) {
            if (wrong[s] < 3) std::printf("S%d wrong: %s at %d plies: eval %d, from scratch %d\n", s + 1, what, g.n_moves, got, truth);
            wrong[s]++;
        }
    };
    const size_t step = std::max<size_t>(1, all.size() / (size_t)(want * 1.1 + 1));
    for (size_t i = 0; i < all.size() && tried[0] < want; i += step) {
        GlobalBoard A;
        if (!prepare_board_for_search(A, all[i].s)) continue;
        // an unrelated position B two plies deeper than A
        GlobalBoard B;
        bool found = false;
        for (size_t k = 0, j = (i * 7919 + 13) % all.size(); k < all.size() && !found; k++, j = (j + 1) % all.size()) {
            GlobalBoard t;
            if (prepare_board_for_search(t, all[j].s) && t.n_moves == A.n_moves + 2 && t.checkWinner() == -1) {
                B = t;
                found = true;
            }
        }
        if (!found) continue;
        // search 1 from A (each scenario on its own fresh engine): refresh, two plies made and
        // evaluated; returns false when A has fewer than two plies to play
        GlobalBoard G = A;
        auto search1 = [&](Dev &e, bool record) {
            Dev::FastBoard fa(A);
            enter(e, fa);
            e.fnnue_stack.refresh_root(fa);
            for (int p = 0; p < 2; p++) {
                Move m;
                if (!first_legal(e, fa, m)) return false;
                e.make_move_fast(fa, m);
                if (record) G.makeMove(m);
                (void)eval_here(e, fa);
            }
            return true;
        };
        auto e1 = std::make_unique<Dev>(), e2 = std::make_unique<Dev>(), e3 = std::make_unique<Dev>();
        if (!search1(*e1, true)) continue;
        // S1: B entered directly, no refresh
        {
            Dev::FastBoard fb(B);
            enter(*e1, fb);
            report(0, eval_here(*e1, fb), fnnue::evaluate_any(B, d16_mini_board_constraint(B)), B, "unrelated entry");
        }
        // S2: a search from A cut after an unevaluated make at A+2; then B (at A+2) entered, one move, child evaluated
        {
            Dev &e = *e2;
            Dev::FastBoard fa(A);
            enter(e, fa);
            e.fnnue_stack.refresh_root(fa);
            Move m1, m2, m3;
            if (first_legal(e, fa, m1)) {
                e.make_move_fast(fa, m1);
                (void)eval_here(e, fa);  // ply A+1 evaluated
                if (first_legal(e, fa, m2)) {
                    e.make_move_fast(fa, m2);  // ply A+2 made, never evaluated (a cut)
                    Dev::FastBoard fb(B);
                    enter(e, fb);
                    if (first_legal(e, fb, m3)) {
                        GlobalBoard Bc = B;
                        Bc.makeMove(m3);
                        e.make_move_fast(fb, m3);
                        report(1, eval_here(e, fb), fnnue::evaluate_any(Bc, d16_mini_board_constraint(Bc)), Bc,
                               "child of an unrelated entry above a cut");
                    }
                }
            }
        }
        // S3: search 1's evaluated grandchild G, entered afresh from a GlobalBoard
        if (G.checkWinner() == -1 && search1(*e3, false)) {
            Dev::FastBoard fg(G);
            enter(*e3, fg);
            report(2, eval_here(*e3, fg), fnnue::evaluate_any(G, d16_mini_board_constraint(G)), G, "re-entry at a held position");
        }
        // S4: a normal search on S1's engine afterwards
        {
            GlobalBoard c = A;
            int score = 0;
            e1->search_fixed_depth(c, 4, score);
            s4_nodes += e1->nodes;
            tried[3]++;
        }
    }
    std::printf("fallback_repro: S1 unrelated entry: %d of %d wrong | S2 entry above a cut: %d of %d wrong | "
                "S3 re-entry at a held position: %d of %d wrong | S4 %d normal searches after them (%lld nodes)\n",
                wrong[0], tried[0], wrong[1], tried[1], wrong[2], tried[2], tried[3], s4_nodes);
#ifdef FASTNNUE_CHECK
    std::printf("fallback_repro: fallback refreshes %lld (S1 and S2 need one each; S3 and S4 none)\n",
                fnnue::g_check.fallback_refresh.load());
#endif
    std::fflush(stdout);
    return (wrong[0] + wrong[1] + wrong[2]) ? 3 : 0;
}
