// Regression driver for the empty-board key of the accumulator stacks (review 3, fixed in stage 5): searches
// entered without refresh_root at, or with their chain reaching, the empty board. Until the fix an unset
// entry's key was 0, which is also the empty board's tt_hash, so such an entry looked like it held the empty
// board; the stacks now use the sentinel kNoKey. fallback_repro.cpp covers the other entry cases (S1-S4).
// Compile in a fast candidate directory (build_cand.sh builds stack_repro[_check].exe) so "test_bots.cpp"
// resolves there:
//
//   stack_repro [POSITIONS.cfdg]        (FASTNNUE_PATH, or the candidate's compiled-in net; POSITIONS unused)
//
// Z1  a fresh engine evaluates the EMPTY board (ply 0) through its stack, no refresh_root.
// Z2  a fresh engine makes one move from the empty board and evaluates the child (ply 1), no refresh_root.
// Z3  a fresh engine makes k = 1..6 moves from the empty board (first legal each time) and evaluates each
//     node on the way down, no refresh_root.
// Z4  an engine that already ran a normal search_fixed_depth from a real position (refreshing at that
//     root), then enters the empty board and one child without refresh_root (the ply-0 entry was never
//     written).
// Z5  control: the same as Z2 but with refresh_root at the empty board first (the getMove path).
// Every stack eval is compared with the from-scratch evaluation of the same board; exit status 3 if any is
// wrong. Before the fix Z1-Z4 were wrong on every entry (81/81, 81/81, 485-486/486, 162/162) for all three
// nets; with the kNoKey headers all are 0.
#include <memory>
#define private public
#define main tb_main
#include "test_bots.cpp"
#undef main
#undef private

using Dev = CrossfishDev;

static void enter(Dev &e, Dev::FastBoard &fb) {  // what getMove does, minus refresh_root
    e.init_hce_acc(fb);
    e.init_macro_key(fb);
    e.sync_terminal(fb);
}
static int stack_eval(Dev &e, Dev::FastBoard &fb) { return e.fnnue_stack.evaluate(fb, Dev::active_board_index(fb)); }
static int scratch_eval(GlobalBoard &g) { return fnnue::evaluate_any(g, d16_mini_board_constraint(g)); }
static bool first_legal(Dev &e, Dev::FastBoard &fb, Move &m, int pick) {
    Move mv[81];
    if (e.check_winner_fast(fb) != -1) return false;
    const int n = e.fill_legal_moves_fast(fb, mv);
    if (n <= 0) return false;
    m = mv[pick % n];
    return true;
}

int main(int argc, char **argv) {
    Dev::init_mini_lut();
    int tried[5] = {0}, wrong[5] = {0};
    auto rep = [&](int s, int got, int truth, int ply) {
        tried[s]++;
        if (got != truth) {
            if (wrong[s] < 2) std::printf("Z%d wrong at ply %d: stack %d, from scratch %d\n", s + 1, ply, got, truth);
            wrong[s]++;
        }
    };
    {
        GlobalBoard empty;
        Dev::FastBoard fb(empty);
        std::printf("empty board: FastBoard.tt_hash = %llu (an unset stack entry's key must differ from it)\n",
                    (unsigned long long)fb.tt_hash);
    }
    for (int pick = 0; pick < 81; pick++) {
        // Z1
        {
            auto e = std::make_unique<Dev>();
            GlobalBoard g;
            Dev::FastBoard fb(g);
            enter(*e, fb);
            rep(0, stack_eval(*e, fb), scratch_eval(g), 0);
        }
        // Z2
        {
            auto e = std::make_unique<Dev>();
            GlobalBoard g;
            Dev::FastBoard fb(g);
            enter(*e, fb);
            Move m;
            if (first_legal(*e, fb, m, pick)) {
                e->make_move_fast(fb, m);
                g.makeMove(m);
                rep(1, stack_eval(*e, fb), scratch_eval(g), 1);
            }
        }
        // Z3
        {
            auto e = std::make_unique<Dev>();
            GlobalBoard g;
            Dev::FastBoard fb(g);
            enter(*e, fb);
            for (int k = 1; k <= 6; k++) {
                Move m;
                if (!first_legal(*e, fb, m, pick + 7 * k)) break;
                e->make_move_fast(fb, m);
                g.makeMove(m);
                rep(2, stack_eval(*e, fb), scratch_eval(g), k);
            }
        }
        // Z4
        {
            auto e = std::make_unique<Dev>();
            GlobalBoard a;
            a.makeMove(Move{4, 4});
            a.makeMove(Move{4, (pick % 8) < 4 ? pick % 8 : pick % 8 + 1});
            int score = 0;
            e->search_fixed_depth(a, 5, score);
            GlobalBoard g;
            Dev::FastBoard fb(g);
            enter(*e, fb);
            rep(3, stack_eval(*e, fb), scratch_eval(g), 0);
            Move m;
            if (first_legal(*e, fb, m, pick)) {
                e->make_move_fast(fb, m);
                g.makeMove(m);
                rep(3, stack_eval(*e, fb), scratch_eval(g), 1);
            }
        }
        // Z5 control
        {
            auto e = std::make_unique<Dev>();
            GlobalBoard g;
            Dev::FastBoard fb(g);
            enter(*e, fb);
            e->fnnue_stack.refresh_root(fb);
            rep(4, stack_eval(*e, fb), scratch_eval(g), 0);
            Move m;
            if (first_legal(*e, fb, m, pick)) {
                e->make_move_fast(fb, m);
                g.makeMove(m);
                rep(4, stack_eval(*e, fb), scratch_eval(g), 1);
            }
        }
    }
    std::printf("stack_repro: Z1 empty board %d/%d wrong | Z2 child of the empty board %d/%d wrong | "
                "Z3 plies 1-6 from the empty board %d/%d wrong | Z4 after a normal search %d/%d wrong | "
                "Z5 control with refresh_root %d/%d wrong\n",
                wrong[0], tried[0], wrong[1], tried[1], wrong[2], tried[2], wrong[3], tried[3], wrong[4], tried[4]);
    return (wrong[0] + wrong[1] + wrong[2] + wrong[3] + wrong[4]) ? 3 : 0;
}
