import sys
f=sys.argv[1]; s=open(f).read()
def rep(old,new):
    global s; assert s.count(old)==1,old[:70]; s=s.replace(old,new)
rep('#include "macro_eval.hpp"', '#include "macro_eval.hpp"\n#include "full_nnue.hpp"')
rep("""            int hce = evaluate_hce_incremental(board)
                    + qstruct.applied;
            if (hce - QHCE_FAIL_HIGH_MARGIN >= beta) {
                return beta;
            }
            int stand_pat;
            if (hce + MINI_MAX + MACRO_CLIP < alpha) {
                stand_pat = hce + MINI_MAX + MACRO_CLIP;
            } else {
                stand_pat = hce + evaluate_mini_cached(board)
                          + evaluate_macro_cached(board);
            }""","""            // EXPERIMENT: the full NNUE replaces HCE + MiniNet + macro.
            int stand_pat = FullNnueFloat::evaluate_board(
                                board, active_board_index(board))
                          + qstruct.applied;""")
rep("""                static_eval = corrected_eval(
                    evaluate_hce_incremental(board), static_corr_refs);""","""                static_eval = corrected_eval(
                    FullNnueFloat::evaluate_board(
                        board, active_board_index(board)),
                    static_corr_refs);""")
rep("""                if (depth == 1
                    && static_eval + 2500 - reverse_futility_margin >= beta
                    && static_eval + evaluate_mini_cached(board)
                       - reverse_futility_margin >= beta) {
                    return beta;
                }
""","")
open(f,"w").write(s); print("ok")
